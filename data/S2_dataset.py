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


class S2_rand_dataset(Dataset):
    def __init__(self, config, phase="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
            patch_size (int): Size of the patch to be extracted (default: 512).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.config = config
        self.root_dir = self.config.Data.S2_rand_settings.base_path
        self.patch_size = self.config.Data.S2_rand_settings.image_size
        self.no = self.config.Data.S2_rand_settings.no_images
        self.phase=phase
        
        # Collect all the paths of the stacked_10m.tif files in the directory and subdirectories
        self.image_paths = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename == "stacked_10m.tif":
                    self.image_paths.append(os.path.join(dirpath, filename))
        assert len(self.image_paths) > 0, "No images found in the directory"
        if len(self.image_paths) ==1:
            print("Warning: Train/test split depends on multiple files. Only one has been found.")
        # order file names alphabetically
        self.image_paths.sort()
        if phase=="train":
            if len(self.image_paths)==1:
                pass
            else:
                self.image_paths = self.image_paths[:-1]
        elif phase=="val":
            if len(self.image_paths)==1:
                self.no = 500
            else:
                self.image_paths = [self.image_paths[-1]]
                self.no = 500
        else:
            raise ValueError("phase must be either 'train' or 'val'")
        
        # set padding settings
        if self.config.Data.padding:
            self.pad = torch.nn.ReflectionPad2d(self.config.Data.padding_amount)

        print("Instanciated S2_random dataset with",self.no,"datapoints for phase: ",self.phase)


    def __len__(self):
        return self.no

    def __getitem__(self, idx):
        # Select a random image
        if self.phase=="train":
            img_path = self.image_paths[random.randint(0, len(self.image_paths) - 1)] # rand select of img
        if self.phase=="val":
            img_path = self.image_paths[0] # only 1 if val, get this one
        
        # Open the image using rasterio
        with rasterio.open(img_path) as dataset:

            # Get image dimensions
            width = dataset.width
            height = dataset.height
            
            # Ensure the patch fits within the image boundaries
            x_max = width - self.patch_size
            y_max = height - self.patch_size

            # Randomly select from top-left corner of the patch
            x = random.randint(0, x_max)
            y = random.randint(0, y_max)
            
            # Read the patch from the image
            patch = dataset.read(window=((y, y + self.patch_size), (x, x + self.patch_size)))
            
            if self.config.Data.S2_rand_settings.return_coords:
                # get centroid lon/lat
                crs = dataset.crs
                trans = dataset.transform
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

        return batch



class S2_datamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        # Initialize the dataset
        self.dataset_train = S2_rand_dataset(self.config,phase="train")
        self.dataset_val = S2_rand_dataset(self.config,phase="val")
        

    def train_dataloader(self):        
        return DataLoader(self.dataset_train,batch_size=self.config.Data.train_batch_size,
                          shuffle=True, num_workers=self.config.Data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,batch_size=self.config.Data.val_batch_size,
                          shuffle=True, num_workers=self.config.Data.num_workers,drop_last=True)


if __name__ == "__main__":
    # Example usage
    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")

    ds = S2_rand_dataset(config,phase="train")
    ds_v = S2_rand_dataset(config,phase="val")

    
    data_module = S2_datamodule(config)

    # get a batch
    a = next(iter(data_module.train_dataloader()))

    from tqdm import tqdm
    val,inval = 0,0
    for i in tqdm(data_module.train_dataloader()):
        if i["rgb"].shape[-1]!=512:
            inval+=1
        else:
            val+=1
    print("Valid:",val,"Invalid:",inval)