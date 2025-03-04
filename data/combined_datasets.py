from torch.utils.data import ConcatDataset, Subset, ChainDataset
from data.S2_dataset import S2_datamodule, S2_rand_dataset
from data.S2NAIP_final import S2NAIP_dm, SEN2NAIPv2
from data.s2_75k_dataset import S2_75k,S2_75k_datamodule
from data.s100k_dataset import S2_100k,S2_100k_datamodule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import numpy as np


class CombinedDataModule(LightningDataModule):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.train_batch_size = config.Data.train_batch_size
            self.val_batch_size = config.Data.val_batch_size
            self.num_workers = config.Data.num_workers

            # Here, we want to get our datasets, not PL datamodules
            # Get and Concat train Datasets
            train_dataset1 = SEN2NAIPv2(self.config,phase="train")
            train_dataset2 = S2_rand_dataset(self.config,phase="train")
            train_dataset3 = S2_75k(self.config,phase="train")
            train_dataset4 = S2_100k(self.config,phase="train")
            self.train_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3, train_dataset4])
            print("Combined Train Dataset Length:",len(self.train_dataset))
            
            val_dataset1 = SEN2NAIPv2(self.config,phase="val")
            val_dataset2 = S2_rand_dataset(self.config,phase="val")
            val_dataset3 = S2_75k(self.config,phase="val")
            val_dataset4 = S2_100k(self.config,phase="val")
            self.val_dataset = ConcatDataset([val_dataset1, val_dataset2, val_dataset3, val_dataset4])
            print("Combined Val Dataset Length:",len(self.val_dataset))
            
        
        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                              num_workers=self.num_workers, shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.val_dataset, num_workers=4,
                              batch_size=self.val_batch_size, shuffle=True)
        
        
if __name__=="__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
    dm = CombinedDataModule(config)
      
