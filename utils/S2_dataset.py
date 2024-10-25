import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import rasterio
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import OmegaConf

class S2_dataset(Dataset):
    def __init__(self, root_dir, patch_size=512, no=10000, phase="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
            patch_size (int): Size of the patch to be extracted (default: 512).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.no = no
        self.phase=phase
        
        # Collect all the paths of the stacked_10m.tif files in the directory and subdirectories
        self.image_paths = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename == "stacked_10m.tif":
                    self.image_paths.append(os.path.join(dirpath, filename))
        assert len(self.image_paths) > 0, "No images found in the directory"
        # order file names alphabetically
        self.image_paths.sort()

        if phase=="train":
            self.image_paths = self.image_paths[:-1]
        elif phase=="val":
            self.image_paths = [self.image_paths[-1]]
            self.no = 32
        else:
            raise ValueError("phase must be either 'train' or 'val'")

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
               
        # turn to 0..1
        patch = patch / 10000.0
        patch = torch.from_numpy(patch).float()
        patch = torch.clamp(patch, 0, 1)

        # extract RGB and NIR bands
        batch = {"rgb": patch[:3,:,:], "nir": patch[3:,:,:]}

        return batch



class S2_datamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.root_dir = config.Data.data_dir
        self.patch_size = 256
        self.num_workers = config.Data.num_workers
        self.config = config
        # Initialize the dataset
        self.dataset_train = S2_dataset(root_dir=self.root_dir,
                                        patch_size=self.patch_size,phase="train")
        self.dataset_val = S2_dataset(root_dir=self.root_dir,
                                      patch_size=self.patch_size,phase="val")
        

    def train_dataloader(self):        
        return DataLoader(self.dataset_train,batch_size=self.config.Data.train_batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,batch_size=self.config.Data.val_batch_size,
                          shuffle=True, num_workers=self.num_workers,drop_last=True)


if __name__ == "__main__":
    # Example usage
    config = OmegaConf.load("configs/config_NIR.yaml")
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