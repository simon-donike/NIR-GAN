import os
import zipfile
import subprocess
from tqdm import tqdm
from rasterio.warp import transform



def deep_folder_search(directory,file_type=".zip"):
    """Recursively finds all .zip files in the given directory and subdirectories."""
    zip_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(file_type):
                zip_files.append(os.path.join(root, file))
    return zip_files

def unzip_and_delete(zip_files):
    for zip_path in zip_files:
        extract_dir = os.path.dirname(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        os.remove(zip_path)
        print(f"Unzipped and deleted: {zip_path}")

# go over all zip files until none left
def unzip_deep(file_path):
    zip_files = deep_folder_search(file_path, ".zip")
    print("Found zip files: ", len(zip_files))
    while len(zip_files) > 0:
        unzip_and_delete(zip_files)
        zip_files = deep_folder_search(file_path, ".zip")
        print("Found zip files: ", len(zip_files))
    print("Finished unzipping all files.")

# define archive
archive_path = "/data2/simon/spot6_archive"
# unzip iteratively all files
unzip_deep(archive_path)
# get image file paths
image_files = deep_folder_search(archive_path, ".jp2")
print("Amount of SPOT6 Scenes: ", len(image_files))



"""
Create Dataset
"""

import rasterio
import pandas as pd
import numpy as np
from rasterio.windows import Window

def create_windows(image_path, window_size=512):
    """Generates 512x512 windows for a given JP2 image."""
    windows = []
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        for y in range(0, height, window_size):
            for x in range(0, width, window_size):
                w = Window(x, y, min(window_size, width - x), min(window_size, height - y))
                windows.append(w)
    return windows

def process_images(jp2_files):
    """Processes a list of JP2 files and extracts 512x512 windows."""
    data = []
    for img_path in tqdm(jp2_files,desc="Creating Images..."):
        windows = create_windows(img_path)
        with rasterio.open(img_path) as src:
            transform_matrix = src.transform
            src_crs = src.crs
            dst_crs = "EPSG:4326"
            for w in windows:
                img_array = src.read(window=w)
                zero_ratio = np.mean(img_array == 0)
                if zero_ratio < 0.1:
                    # Compute centroid coordinates
                    x_center = w.col_off + w.width / 2
                    y_center = w.row_off + w.height / 2
                    lon, lat = rasterio.transform.xy(transform_matrix, y_center, x_center)
                    lon, lat = transform(src_crs, dst_crs, [lon], [lat])
                    
                    data.append({"file": img_path, "window": w,"zero_ratio":zero_ratio, "lon": lon[0], "lat": lat[0]})
    
    df = pd.DataFrame(data)
    return df

# Example usage:
jp2_files = deep_folder_search(archive_path, ".jp2")[:2]
df = process_images(jp2_files)
df.to_csv("playground/spot_dataset.csv")
print(df.head())
