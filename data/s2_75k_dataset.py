import os
import random
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pyproj import Transformer
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import Point
from tqdm import tqdm

class S2_75k(Dataset):
    def __init__(self, config, phase="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
            patch_size (int): Size of the patch to be extracted (default: 512).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.config = config
        self.root_dir = self.config.Data.S2_75k_settings.base_path
        self.patch_size = self.config.Data.S2_75k_settings.image_size
        self.phase=phase
        
        # read all files in directory
        self.image_paths = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                # assert file is tiff
                if filename.endswith(".tif"):
                    self.image_paths.append(os.path.join(dirpath, filename))
        assert len(self.image_paths) > 0, "No images found in the directory"
        
        # Create DataFrame
        df = pd.DataFrame(self.image_paths, columns=["image_path"])

        # Rabdom shuffle after split
        if self.phase == "train":
            df = df.iloc[:-100]
        elif self.phase == "val":
            df = df.iloc[-100:]

        # Create Dataframe and get metadata information
        self.image_df = df.sample(frac=1).reset_index(drop=True)  # Shuffle after split
        self.build_metadata()
        
        # keep only europe if CLC mask is enabled
        if self.config.Data.S2_75k_settings.return_clc_mask:
            print("Filtering to only include Europe...")
            self.image_df["continent"] = self.image_df["continent"].astype(str).str.strip()
            self.image_df = self.image_df[self.image_df["continent"]=="Europe"]
        print(f"Instantiated S2_75k dataset with {len(self.image_df)} datapoints for phase: {self.phase}")
                    
        # set padding settings
        if self.config.Data.padding:
            self.pad = torch.nn.ReflectionPad2d(self.config.Data.padding_amount)
            
    def build_metadata(self,force_rebuild=False):
        # check weather metadata pkl exists
        metadata_file = os.path.join(self.root_dir, "metadata.pkl")
        if os.path.exists(metadata_file) and not force_rebuild:
            print("Loading metadata from file...")
            self.image_df = pd.read_pickle(metadata_file)
        else:
            print("No metadata file found, creating new metadata...")
        
            continents_file = self.config.Data.S2_75k_settings.continent_geojson
            continents = gpd.read_file(continents_file)
            continents = continents.to_crs("EPSG:4326")  # Ensure it's in lat/lon
            
            geometries = []
            for path in tqdm(self.image_df["image_path"], desc="Extracting centroids"):
                with rasterio.open(path) as src:
                    bounds = src.bounds
                    src_crs = src.crs
                    center_x = (bounds.left + bounds.right) / 2
                    center_y = (bounds.top + bounds.bottom) / 2
                    pt = gpd.GeoSeries([Point(center_x, center_y)], crs=src_crs).to_crs("EPSG:4326").iloc[0]
                    geometries.append(pt)

            # Build a GeoDataFrame with transformed centroids
            centroid_gdf = gpd.GeoDataFrame(
                self.image_df.copy(),
                geometry=geometries,
                crs="EPSG:4326"
            )

            # Spatial join
            joined = gpd.sjoin(centroid_gdf, continents, how="left", predicate="intersects")
            joined = joined.drop(columns=["index_right"])
            joined = joined.rename(columns={"CONTINENT": "continent"})
            self.image_df = joined
            # save to file
            self.image_df.to_pickle(metadata_file)

            
    def __len__(self):
        return len(self.image_df)
    
    def __getitem__(self, idx):
        
        try:
            im_path =  self.image_df.loc[idx, "image_path"]

            with rasterio.open(im_path) as src:
                # Get image dimensions
                width = src.width
                height = src.height
                
                # Ensure the patch fits within the image boundaries
                x_max = width - self.patch_size
                y_max = height - self.patch_size

                # Randomly select from top-left corner of the patch
                x = random.randint(0, x_max)
                y = random.randint(0, y_max)
                
                patch = src.read(window=((y, y + self.patch_size), (x, x + self.patch_size)))

                if self.config.Data.S2_75k_settings.return_coords:
                    # get centroid lon/lat
                    crs = src.crs
                    trans = src.transform
                    centroid_x = x + self.patch_size / 2
                    centroid_y = y + self.patch_size / 2
                    geo_x, geo_y = trans * (centroid_x, centroid_y)
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    lon, lat = transformer.transform(geo_x, geo_y)
                    coords = torch.Tensor([lon,lat])
        except TypeError:
            print("Error reading file:",self.image_paths[idx], "Skipping...")
            rand_idx = random.randint(0,len(self.image_paths)-1)
            return self.__getitem__(rand_idx)

        # turn to 0..1
        patch = patch / 10000.0
        patch = torch.from_numpy(patch).float()
        patch = torch.clamp(patch, 0, 1)
        rgb = patch[:3,:,:]
        nir = patch[3:,:,:]
        if self.config.Data.padding:
            rgb = self.pad(rgb)
            nir = self.pad(nir)
            
        # extract RGB and NIR bands
        batch = {"rgb": rgb, "nir": nir}
        if self.config.Data.S2_75k_settings.return_coords:
            batch["coords"] = coords
        if self.config.Data.S2_75k_settings.return_clc_mask:
            pass

        return(batch)
                
                
                
class S2_75k_datamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Initialize the dataset
        self.dataset_train = S2_75k(self.config,phase="train")
        self.dataset_val = S2_75k(self.config,phase="val")
        

    def train_dataloader(self):        
        return DataLoader(self.dataset_train,batch_size=self.config.Data.train_batch_size,
                          shuffle=True, num_workers=self.config.Data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,batch_size=self.config.Data.val_batch_size,
                          shuffle=False, num_workers=self.config.Data.num_workers,drop_last=True)



if __name__=="__main__":
    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
    ds = S2_75k(config)
    
    #m = ds.build_metadata()
    #dm = S2_75k_datamodule(config)
    #batch = next(iter(dm.train_dataloader()))
    #coords = batch["coords"]

    """
    from tqdm import tqdm
    for i in tqdm(dm.train_dataloader()):
        pass

    for i in ds.image_paths:
        if type(i) != str:
            print(i)
    """
        