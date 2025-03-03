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
class worldstrat_ds(Dataset):
    def __init__(self, config,image_source="hr",metadata_csv="/data1/simon/GitHub/worldstrat/pretrained_model/metadata_ablation.geojson"):
        self.data = gpd.read_file(metadata_csv)
        self.image_source = image_source
        self.hr_base_path = "/data2/simon/worldstrat/hr_dataset/"
        self.lr_base_path = "/data2/simon/worldstrat/lr_dataset/"

        self.config = config
        self.image_size = self.config.Data.worldstrat_settings.image_size
        self.return_coords = self.config.Data.worldstrat_settings.return_coords
        
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get ID
        id_ = self.data.iloc[index]["id"]
        # Fetch data at a specific index
        if self.image_source == "lr":
            im = self.get_lr(index)
            im = im / 10000
        elif self.image_source == "hr":
            im = self.get_hr(index)
            im = im / 10000
            im = self.change_resolution(im, factor=0.6) # go from 1.5 to 2.5m resolution for HR image

        #print(self.image_source,":",im.shape,im.mean())

        # crop to square, crop center to desired size of 128/512
        im = self.crop_square(im)
        im = self.crop_center(im,512)
        
        # extract bands
        rgb = im[:3,:,:]
        nir = im[3:,:,:]

        return_dict = {"rgb":rgb,"nir":nir}

        if self.return_coords:
            row = self.data.iloc[index]
            coords = [row["geometry"].x,row["geometry"].y]
            coords = torch.Tensor(coords)
            return_dict["coords"] = coords

        return return_dict
 

if __name__=="__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
    ds = worldstrat_ds(config)
    _ = ds.__getitem__(10)