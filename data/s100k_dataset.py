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
from tqdm import tqdm

class S2_100k(Dataset):
    def __init__(self, config, phase="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
            patch_size (int): Size of the patch to be extracted (default: 512).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.config = config
        self.root_dir = self.config.Data.S2_100k.base_path
        self.patch_size = self.config.Data.S2_100k.image_size
        self.phase=phase
        assert self.patch_size == 256, "Only 512x512 patches are supported for Sentinel-2 100k dataset."
        
        # read all files in directory
        self.image_paths = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                # assert file is tiff
                if filename.endswith(".tif"):
                    self.image_paths.append(os.path.join(dirpath, filename))
        assert len(self.image_paths) > 0, "No images found in the directory"
        
        # Set train/test/split
        if self.phase=="train":
            self.image_paths = self.image_paths[:-5000]
        if self.phase=="val":
            self.image_paths = self.image_paths[-5000:]
        print(f"Instanciated S2_100k dataset with {len(self.image_paths)} datapoints for phase: {self.phase}")
        
        # shuffle list after split
        random.shuffle(self.image_paths)
            
        # set padding settings
        if self.config.Data.padding:
            self.pad = torch.nn.ReflectionPad2d(self.config.Data.padding_amount)

        # set up error list that will hold the lists of indices that failed to load
        self.error_idx = []
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # If this idx is in the error list, return a random image
        if idx in self.error_idx:
            print("Error cought, returning random image")
            rand_idx = random.randint(0,len(self.image_paths)-1)
            return self.__getitem__(rand_idx)

        # try reading image, if it fails, add to error list and return random image
        try:
            with rasterio.open(self.image_paths[idx]) as src:
                # Get image dimensions
                rgb_nir_bands = [2, 3, 4, 8]
                patch = src.read(rgb_nir_bands)

                if self.config.Data.S2_100k.return_coords:
                    # get centroid lon/lat
                    crs = src.crs
                    trans = src.transform
                    centroid_x = self.patch_size / 2
                    centroid_y = self.patch_size / 2
                    geo_x, geo_y = trans * (centroid_x, centroid_y)
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    lon, lat = transformer.transform(geo_x, geo_y)
                    coords = torch.Tensor([lon,lat])
        except Exception as e:
            print("Error reading file:",self.image_paths[idx], "Error:",e)
            self.error_idx.append(idx)
            rand_idx = random.randint(0,len(self.image_paths)-1)
            return self.__getitem__(rand_idx)
        

        # After successful reading, do preprocessing and return
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
        if self.config.Data.S2_rand_settings.return_coords:
            batch["coords"] = coords
            batch["path"] = self.image_paths[idx]

        # do percentage 
        do_percenteage_check = False
        if do_percenteage_check:
            num_zeros = torch.sum(rgb == 0.00).item()
            total_elements = rgb.numel()
            percentage_zeros = (num_zeros / total_elements) * 100
            if percentage_zeros > 15.:
                print("Error, too many 0.0s :",self.image_paths[idx])
                self.error_idx.append(idx)
                rand_idx = random.randint(0,len(self.image_paths)-1)
                return self.__getitem__(rand_idx)
        
        return(batch)
                
                
                
class S2_100k_datamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Initialize the dataset
        self.dataset_train = S2_100k(self.config,phase="train")
        self.dataset_val = S2_100k(self.config,phase="val")
        

    def train_dataloader(self):        
        return DataLoader(self.dataset_train,batch_size=self.config.Data.train_batch_size,
                          shuffle=True, num_workers=self.config.Data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,batch_size=self.config.Data.val_batch_size,
                          shuffle=True, num_workers=self.config.Data.num_workers,drop_last=True)



if __name__=="__main__":
    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
    ds = S2_100k(config)
    dm = S2_100k_datamodule(config)
    batch = next(iter(dm.train_dataloader()))
    coords = batch["coords"]


    for batch in tqdm(dm.train_dataloader()):
        pass