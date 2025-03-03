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

class S2_75k(Dataset):
    def __init__(self, config, phase="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
            patch_size (int): Size of the patch to be extracted (default: 512).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.config = config
        self.root_dir = self.config.Data.S2_75k.base_path
        self.patch_size = self.config.Data.S2_75k.image_size
        self.phase=phase
        
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
            self.image_paths = self.image_paths[:-100]
        if self.phase=="val":
            self.image_paths = [self.image_paths[-100:]]
        print(f"Instanciated S2_75k dataset with {len(self.image_paths)} datapoints for phase: {self.phase}")
        
        # shuffle list after split
        random.shuffle(self.image_paths)
            
        # set padding settings
        if self.config.Data.padding:
            self.pad = torch.nn.ReflectionPad2d(self.config.Data.padding_amount)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        with rasterio.open(self.image_paths[idx]) as src:
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

            if self.config.Data.S2_75k.return_coords:
                # get centroid lon/lat
                crs = src.crs
                trans = src.transform
                centroid_x = x + self.patch_size / 2
                centroid_y = y + self.patch_size / 2
                geo_x, geo_y = trans * (centroid_x, centroid_y)
                transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(geo_x, geo_y)
                coords = torch.Tensor([lon,lat])
                
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
                          shuffle=True, num_workers=self.config.Data.num_workers,drop_last=True)



if __name__=="__main__":
    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
    ds = S2_75k(config)
    dm = S2_75k_datamodule(config)
    batch = next(iter(dm.train_dataloader()))
    coords = batch["coords"]