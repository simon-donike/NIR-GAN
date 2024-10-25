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
    def __init__(self, csv_path="/data3/landcover_s2naip/csvs/train_metadata_landcover.csv",
                 phase="train",chip_size=256):
        assert phase in ["train","test","val"]
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


    def __len__(self):
        # if val or test, skip dataframe
        if self.phase=="test" or self.phase=="val":
            return len(self.val_files_list)
        # if train, do dataframe
        return len(self.dataframe)

    
    def apply_augmentations(self,lr,hr):
        lr,hr = lr.float(),hr.float()

        # set probabilities
        smoothen_liklelihood   = 0.0 # 0.75
        jitter_liklelihood     = 0.0 # 0.75
        black_spots_likelihood = 0.0

        # get random numbers
        smoothen_rand = random.uniform(0, 1)
        jitter_rand = random.uniform(0, 1)
        black_spots_rand = random.uniform(0, 1)

        # perform smoothen
        if smoothen_rand<smoothen_liklelihood:
            # Define Kernel
            sigma_rand = random.uniform(0.65, 1.2)
            gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=sigma_rand)
            # Apply the blur to the image tensor
            band_ls = []
            for band in lr:
                band = torch.unsqueeze(band,0)
                band = gaussian_blur(band)
                band = torch.squeeze(band,0)
                band_ls.append(band)
            lr = torch.stack(band_ls)
        
        if black_spots_rand<black_spots_likelihood:
            lr = self.add_black_spots(lr)
            
                
        return(lr,hr)
    
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
            
        # return images
        return {"rgb":hr_image[:3,:self.chip_size,:self.chip_size],
                "nir":hr_image[3:,:self.chip_size,:self.chip_size]}






class SEN2NAIP_datamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.root_dir = config.Data.data_dir
        self.num_workers = config.Data.num_workers
        self.config = config
        # Initialize the dataset
        self.dataset_train = S2NAIP_v4(phase="test",
                                       csv_path="/data3/landcover_s2naip/csvs/train_metadata_landcover.csv",
                                       chip_size=256)
        self.dataset_val = S2NAIP_v4(phase="val",
                                     csv_path="/data3/landcover_s2naip/csvs/train_metadata_landcover.csv",
                                     chip_size=256)
        

    def train_dataloader(self):        
        return DataLoader(self.dataset_train,batch_size=self.config.Data.train_batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,batch_size=self.config.Data.val_batch_size,
                          shuffle=True, num_workers=self.num_workers,drop_last=True)