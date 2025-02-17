import numpy as np
import torch
import rasterio
import os,glob
import os
import shutil
import glob
import random
from tqdm import tqdm
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.windows import transform as window_transform
from rasterio.transform import Affine

"""
Code to prepare dataset form a HR version only
"""


def prepare_dataset(input_folder: str, hr_output_folder: str, lr_output_folder: str, scale_factor: int, window_size: int):
    """
    Args:
        input_folder : str, path to folder containing HR images
        hr_output_folder : str, path to folder where HR images will be saved
        lr_output_folder : str, path to folder where LR images will be saved
        scale_factor : int, scale factor to downsample HR images
        window_size : int, size of the window to read the images
    """
    # Create output folders
    os.makedirs(hr_output_folder, exist_ok=True)
    os.makedirs(lr_output_folder, exist_ok=True)

    window_size = window_size*2 # account for 1.25 instead of 2.5m pixel size
    
    # Get list of HR images
    hr_images = glob.glob(os.path.join(input_folder, "*.tif"))
    
    # Loop over HR images
    for hr_image in tqdm(hr_images,desc="images..."):
        if "vietnam_hr_27.tif" in hr_image:
            continue
        # Open HR image
        with rasterio.open(hr_image) as src:
            # Get image name
            image_name = os.path.basename(hr_image)

            # Get image shape
            height, width = src.height, src.width

            for i in range(0, height, window_size):
                for j in range(0, width, window_size):
                    # Define window
                    window = Window(j, i, window_size, window_size)

                    # Read HR image window and save geotransform
                    hr_data = src.read(window=window)
                    window_gt = window_transform(window, src.transform)
                    hr_transform = window_gt * Affine.scale(2, 2)

                    # Save the window data with geotransform
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "count": 3,
                        "driver": "GTiff",
                        "height": window.height//2,
                        "width": window.width//2,
                        "transform": hr_transform
                    })
                    

                    hr_data = hr_data[:3,:,:]
                    #hr_data = hr_data.astype(np.int8)
                    
                    if hr_data.shape != (3,600,600) or hr_data.mean() <= 0.05:
                        print("Error:",hr_data.shape, " - Mean:",hr_data.mean())
                        continue
                    else:
                        pass

                    # turn into Torch
                    hr_data = torch.Tensor(hr_data)

                    # interpolate down factor of 2, then create LR version
                    hr_data = torch.nn.functional.interpolate(hr_data.unsqueeze(0),size=(300,300),mode="bilinear",antialias=True).squeeze(0)
                    lr_data = torch.nn.functional.interpolate(hr_data.unsqueeze(0),size=(75,75),mode="bilinear",antialias=True).squeeze(0)
                    hr_data = hr_data.numpy()
                    lr_data = lr_data.numpy()
                    #print(lr_data.shape,hr_data.shape)
                    
                    # SAVE IMAGES
                    im_name_str = f"{os.path.splitext(image_name)[0]}_{i}_{j}.tif"
                    hr_image_path = os.path.join(hr_output_folder, im_name_str)
                    lr_image_path = os.path.join(lr_output_folder, im_name_str)


                    with rasterio.open(hr_image_path, "w", **out_meta) as dest:
                        dest.write(hr_data)

                    lr_transform = hr_transform * Affine.scale(4, 4)
                    
                    out_meta.update({
                        "driver": "GTiff",
                        "height": window.height//8,
                        "width": window.width//8,
                        "transform": lr_transform
                    })
                    with rasterio.open(lr_image_path, "w", **out_meta) as dest:
                        dest.write(lr_data)




# path
input_path = "/data1/simon/datasets/vietnam_google/raw"
lr_path = "/data1/simon/datasets/vietnam_google/train/LR"
hr_path = "/data1/simon/datasets/vietnam_google/train/HR"
prepare_dataset(input_folder=input_path, hr_output_folder=hr_path, lr_output_folder=lr_path, scale_factor=4, window_size=300)



def split_dataset(hr_folder: str, lr_folder: str, test_folder: str, test_percentage: float):
    """
    Args:
        hr_folder : str, path to folder containing HR images
        lr_folder : str, path to folder containing LR images
        test_folder : str, path to folder where test set will be saved
        test_percentage : float, percentage of images to move to test set
    """
    # Create test folder and subfolders
    hr_test_folder = os.path.join(test_folder, "HR")
    lr_test_folder = os.path.join(test_folder, "LR")
    os.makedirs(hr_test_folder, exist_ok=True)
    os.makedirs(lr_test_folder, exist_ok=True)
    
    # Get list of HR and LR images
    hr_images = glob.glob(os.path.join(hr_folder, "*.tif"))
    lr_images = glob.glob(os.path.join(lr_folder, "*.tif"))
    
    # Ensure we have the same number of HR and LR images
    assert len(hr_images) == len(lr_images), "Mismatch in number of HR and LR images"
    
    # Calculate number of images to move
    num_images = len(hr_images)
    num_test_images = int(num_images * test_percentage)
    print("Total Amount of images",num_images,"Amount of images to move",num_test_images)
    
    # Select random images to move
    test_indices = random.sample(range(num_images), num_test_images)
    
    for idx in tqdm(test_indices,desc="Moving..."):
        hr_image = hr_images[idx]
        lr_image = lr_images[idx]
        print(hr_image,lr_image)
        
        # Move HR image to test folder
        hr_image_name = os.path.basename(hr_image)
        hr_test_image_path = os.path.join(hr_test_folder, hr_image_name)
        shutil.move(hr_image, hr_test_image_path)
        
        # Move LR image to test folder
        lr_image_name = os.path.basename(lr_image)
        lr_test_image_path = os.path.join(lr_test_folder, lr_image_name)
        shutil.move(lr_image, lr_test_image_path)

# Example usage
test_path = "/data1/simon/datasets/vietnam_google/val"
split_dataset(hr_folder=hr_path, lr_folder=lr_path,test_folder=test_path , test_percentage=0.1)


