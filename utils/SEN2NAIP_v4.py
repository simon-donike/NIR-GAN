# import dataset utilities
import torch
import os
import pandas as pd
import torch
import random
from einops import rearrange
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# Step 2: Custom Dataset Class
class S2NAIP_v4(Dataset):
    def __init__(self, config, csv_path="/data3/landcover_s2naip/csvs/train_metadata_landcover.csv",
                 phase="train",chip_size=512):
        assert phase in ["train","test","val"]
        self.config = config
        self.phase = phase
        self.chip_size=chip_size

        # read and clean DF
        self.dataframe = pd.read_csv(csv_path)
        self.dataframe = self.dataframe.fillna(0)
        self.dataframe = self.dataframe[self.dataframe["SuperClass"]!=0]

        # set paths
        self.input_dir = "/data3/final_s2naip_simon/"
        self.hr_input_path = os.path.join(self.input_dir,self.phase,"HR")
        self.lr_input_path = os.path.join(self.input_dir,self.phase,"LR")
        self.degradations = ["none","gaussian","bell","sigmoid"]

        # if val or test, just load from folder
        if self.phase=="test" or self.phase=="val":
            # list all files in input directory
            self.val_files_list = os.listdir(os.path.join(self.lr_input_path,"none"))
            
        # set padding settings
        if self.config.Data.padding:
            self.pad = torch.nn.ReflectionPad2d(self.config.Data.padding_amount)


    def __len__(self):
        # if val or test, skip dataframe
        if self.phase=="test" or self.phase=="val":
            return len(self.val_files_list)
        # if train, do dataframe
        return len(self.dataframe)

    
    def __getitem__(self, idx):
        """
        # get random degradation, get input paths
        random_degradation = np.random.choice(self.degradations)
        lr_input_path = os.path.join(self.input_dir,self.phase,"LR",random_degradation)
        lr_input_path = os.path.join(lr_input_path,self.dataframe.iloc[idx]["name"] + ".pt")
        hr_input_path = os.path.join(self.hr_input_path,self.dataframe.iloc[idx]["name"] + ".pt")
        """

        # get random degradation, get input paths
        random_degradation = np.random.choice(self.degradations)
        lr_input_path = os.path.join(self.input_dir,self.phase,"LR",random_degradation)

        if self.phase=="train":
            lr_input_path = os.path.join(lr_input_path,self.dataframe.iloc[idx]["name"] + ".pt")
            hr_input_path = os.path.join(self.hr_input_path,self.dataframe.iloc[idx]["name"] + ".pt")
        if self.phase=="val":
            lr_input_path = os.path.join(self.lr_input_path,random_degradation,  self.val_files_list[idx])
            hr_input_path = os.path.join(self.hr_input_path,  self.val_files_list[idx] )
        if self.phase=="test":
            lr_input_path = os.path.join(self.lr_input_path,random_degradation,  self.val_files_list[idx])
            hr_input_path = os.path.join(self.hr_input_path,  self.val_files_list[idx] )

        # check if files exist
        if not os.path.exists(lr_input_path) or not os.path.exists(hr_input_path):
            print("WARINING: file not available.")
            print(lr_input_path,hr_input_path)

       # Load iamges from disk
        hr_image = torch.load(hr_input_path)

        # bring to value range. Check since they are mixed while writing to disk is ongoing
        if hr_image.max()>10:
            hr_image = hr_image/10000
            
            
        rgb = hr_image[:3,:self.chip_size,:self.chip_size]
        nir = hr_image[3:,:self.chip_size,:self.chip_size]
        
        # apply a
        if self.phase in ["train","validation"]:
            nir = nir.unsqueeze(0)
            rgb,nir = self.augment_images(rgb.numpy(),nir.numpy())
            rgb = torch.tensor(rgb).float()
            nir = torch.tensor(nir).float().squeeze(0)
            
        # apply padding
        if self.config.Data.padding:
            rgb = self.pad(rgb)
            nir = self.pad(nir.unsqueeze(0)).squeeze(0)
            
        if len(nir.shape)==2:
            nir = nir.unsqueeze(0)
            
        # return images
        return {"rgb":hr_image[:3,:self.chip_size,:self.chip_size],
                "nir":hr_image[3:,:self.chip_size,:self.chip_size]}

    def augment_images(self,rgb,nir):
        # 1. value stretch
        if np.random.randint(0,100) > 50:
            m_value = np.random.uniform(0.85,1.15)
            rgb,nir = rgb*m_value,nir*m_value

        # 2. flip
        if np.random.randint(0,100) > 0:
            rgb,nir = np.flip(rgb,axis=(1,2)),np.flip(nir,axis=(1,2))

        # 3. rotate - 3 times 90 degrees for all possible rotations
        rotation_count = np.random.choice([0, 1, 2, 3])  # Choose how many 90-degree rotations to apply
        if rotation_count > 0:
            rgb = np.rot90(rgb, k=rotation_count,axes=(1,2))
            nir = np.rot90(nir, k=rotation_count,axes=(1,2))
            
        # 4. Add noise
        if np.random.randint(0,100) > 101:
            noise_level = np.random.uniform(0.0,0.001)
            noise = np.random.normal(0, noise_level, rgb.shape)
            rgb = np.clip(rgb + noise, 0, 1)

        # final clean up
        rgb,nir = rgb.clip(0,1),nir.clip(0,1)        
        return rgb,nir






class SEN2NAIP_datamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.root_dir = config.Data.data_dir
        self.num_workers = config.Data.num_workers
        self.config = config
        # Initialize the dataset
        self.dataset_train = S2NAIP_v4(config=config,
                                       phase="test",
                                       csv_path="/data3/landcover_s2naip/csvs/train_metadata_landcover.csv",
                                       chip_size=512)
        self.dataset_val = S2NAIP_v4(config=config,
                                     phase="val",
                                     csv_path="/data3/landcover_s2naip/csvs/train_metadata_landcover.csv",
                                     chip_size=512)
        

    def train_dataloader(self):        
        return DataLoader(self.dataset_train,batch_size=self.config.Data.train_batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,batch_size=self.config.Data.val_batch_size,
                          shuffle=True, num_workers=self.num_workers,drop_last=True)
        


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/config_px2px.yaml")
    ds = S2NAIP_v4(config)
    pl_datamodule = SEN2NAIP_datamodule(config)

    b = pl_datamodule.dataset_train[1]
    rgb,nir = b["rgb"],b["nir"]
    