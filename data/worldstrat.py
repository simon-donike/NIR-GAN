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
import geopandas as gpd

# Step 1: Define a custom dataset
class worldstrat(Dataset):
    def __init__(self, config,phase="train",metadata_csv="/data1/simon/GitHub/worldstrat/pretrained_model/metadata_ablation.geojson"):
        self.data = gpd.read_file(metadata_csv)
        self.image_source = config.Data.worldstrat_settings.image_type
        self.hr_base_path = "/data2/simon/worldstrat/hr_dataset/"
        self.lr_base_path = "/data2/simon/worldstrat/lr_dataset/"

        self.config = config
        self.image_size = self.config.Data.worldstrat_settings.image_size
        self.return_coords = self.config.Data.worldstrat_settings.return_coords

        # set padding settings
        if self.config.Data.padding:
            self.pad = torch.nn.ReflectionPad2d(self.config.Data.padding_amount)
        
        if phase == "train":
            # select 80 percent of the data for training
            self.data = self.data.sample(frac=0.8,random_state=42)
        elif phase == "val":
            # select 20 percent of the data for validation
            self.data = self.data.sample(frac=0.2,random_state=42)
        else:
            raise ValueError("phase must be either 'train' or 'val'")
        
    def get_hr(self,idx):
        row = self.data.iloc[idx]
        hr_path = self.hr_base_path + str(row["id"]) + "/" + str(row["id"]) + "_ps.tiff"
        with rasterio.open(hr_path) as src:
            hr = src.read()
        hr = torch.tensor(hr.astype(np.float32))
        return hr
    
    def get_lr(self,idx):
        row = self.data.iloc[idx]
        id_ = row["id"]
        lr_path = self.lr_base_path + str(id_) + "/" + "L2A" + "/" + str(id_) + "-XXX-L2A_data.tiff"
        im_array = []
        for band_no in [4,3,2,8]:
            lr_band_path = lr_path.replace("XXX",str(row["n"]))
            with rasterio.open(lr_band_path) as band:
                im_array.append(band.read(band_no))
        im_array = np.stack(im_array, axis=-1)
        lr = im_array.transpose(2,0,1)
        lr = torch.tensor(lr.astype(np.float32))
        return(lr)
    
    def change_resolution(self, im, factor=0.6):
        # Get the current resolution
        c, h, w = im.shape
        # Calculate the scaling factor
        # Resize the image
        im = torch.nn.functional.interpolate(im.unsqueeze(0), scale_factor=factor, mode='bilinear', align_corners=False)
        return im.squeeze(0)
    
    def crop_square(self,im):
        assert len(im.shape)==3, "Input must be a 3D array"
        c, h, w = im.shape  # Get the dimensions of the image
        assert c<h and c<w, "Input must be a 3D array with the first dimension being the number of channels"
        min_dim = min(h, w) # Find the smaller dimension
        # Calculate the starting point for the crop
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        # Perform the crop
        im = im[:, start_h:start_h + min_dim, start_w:start_w + min_dim]
        return im
    
    def crop_center(self, im, target_height):
        target_width = target_height
        assert len(im.shape) == 3, "Input must be a 3D array"
        c, h, w = im.shape  # Get the dimensions of the image
        assert c < h and c < w, "Input must be a 3D array with the first dimension being the number of channels"
        assert target_height <= h and target_width <= w, "Target dimensions must be smaller than image dimensions"
        
        # Calculate the starting points for the crop
        start_h = (h - target_height) // 2
        start_w = (w - target_width) // 2
        
        # Perform the crop
        cropped_im = im[:, start_h:start_h + target_height, start_w:start_w + target_width]
        return cropped_im
    
    def create_patch(self,im):
        if im.shape[-1] > self.image_size:
            im = self.crop_center(im,self.image_size)
        elif im.shape[-1] < self.image_size:
            im = self.pad_image(im)
            # pad image to desired size with reflective padding in all 4 directions
        elif im.shape[-1] == self.image_size:
            pass
        return im

    def pad_image(self,im):
        c, h, w = im.shape  # assuming im shape is (channels, height, width)
        # Compute total padding needed for height and width
        pad_h_total = self.image_size - h
        pad_w_total = self.image_size - w
        # Randomly split the total padding into two parts for each dimension
        pad_h_top = np.random.randint(0, pad_h_total + 1)
        pad_h_bottom = pad_h_total - pad_h_top
        pad_w_left = np.random.randint(0, pad_w_total + 1)
        pad_w_right = pad_w_total - pad_w_left
        # Apply reflective padding along the height and width axes
        im = np.pad(im, ((0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), mode='reflect')
        return im

    def __len__(self):
        return len(self.data)
    
    def augment(self, im):
        # Check if the input is a torch.Tensor, convert to numpy if so
        im_is_torch = False
        if isinstance(im, torch.Tensor):
            im = im.numpy()
            im_is_torch = True

        # Randomly flip the image horizontally
        if random.random() > 0.5:
            im = np.flip(im, axis=-1)
        # Randomly flip the image vertically
        if random.random() > 0.5:
            im = np.flip(im, axis=-2)

        # Create a contiguous array to remove negative strides
        im = np.ascontiguousarray(im)

        # Convert back to torch.Tensor if necessary
        if im_is_torch:
            im = torch.tensor(im)
        return im
    

    
    def __getitem__(self, index):
        # get ID
        id_ = self.data.iloc[index]["id"]
        # Fetch data at a specific index
        if self.image_source == "lr":
            im = self.get_lr(index)
            #im = im / 10000
        elif self.image_source == "hr":
            im = self.get_hr(index)
            im = im  /10000
            im = self.change_resolution(im, factor=0.6) # go from 1.5 to 2.5m resolution for HR image

        # crop to square, crop center to desired size of 128/512
        im = self.crop_square(im)
        im = self.augment(im)
        im = self.create_patch(im)
        
        # extract bands
        rgb = im[:3,:,:]
        nir = im[3:,:,:]

        # apply padding
        if self.config.Data.padding:
            rgb = self.pad(rgb)
            nir = self.pad(nir)

        return_dict = {"rgb":torch.Tensor(rgb),"nir":torch.Tensor(nir)}

        if self.return_coords:
            row = self.data.iloc[index]
            coords = [row["geometry"].x,row["geometry"].y]
            return_dict["coords"] = torch.Tensor(coords)

        return return_dict
    
    
class worldstrat_datamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Initialize the dataset
        self.dataset_train = worldstrat(self.config,phase="train")
        self.dataset_val = worldstrat(self.config,phase="val")
        

    def train_dataloader(self):        
        return DataLoader(self.dataset_train,batch_size=self.config.Data.train_batch_size,
                          shuffle=True, num_workers=self.config.Data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,batch_size=self.config.Data.val_batch_size,
                          shuffle=True, num_workers=self.config.Data.num_workers,drop_last=True)
 

if __name__=="__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
    ds = worldstrat(config)
    ds_dm = worldstrat_datamodule(config)
    _ = ds.__getitem__(10)
    rgb = _["rgb"]
    nir = _["nir"]
    coords = _["coords"]

    for i in range(100):
        _ = ds.__getitem__(i)
        rgb = _["rgb"]
        nir = _["nir"]
        coords = _["coords"]
        print(rgb.mean().item(),nir.mean().item())

