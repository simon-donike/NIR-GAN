from torch.utils.data import ConcatDataset, Subset
from utils.S2_dataset import S2_datamodule
from utils.S2NAIP_final import S2NAIP_dm
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import numpy as np

def create_combined_dataset(config):
    dm1 = S2_datamodule(config)
    dm2 = S2NAIP_dm(config)    
    combined_train_dataset = ConcatDataset([dm1.train_dataloader().dataset, dm2.train_dataloader().dataset])
    combined_val_dataset = ConcatDataset([dm1.val_dataloader().dataset, dm2.val_dataloader().dataset])

        # Create a new LightningDataModule for the combined dataset
    class CombinedDataModule(LightningDataModule):
        def __init__(self, train_dataset, val_dataset,config):
            super().__init__()
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.train_batch_size = config.Data.train_batch_size
            self.val_batch_size = config.Data.val_batch_size
            self.num_workers = config.Data.num_workers
            
            print("----------------------")
            print("Instanciated Combined S2/S2NAIP dataset with",len(self.train_dataset),"datapoints for phase: train")
            print("Instanciated Combined S2/S2NAIP dataset with",len(self.val_dataset),"datapoints for phase: val")


        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                              num_workers=self.num_workers, shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.val_dataset, num_workers=4, batch_size=self.val_batch_size)
        
    # instanciate object
    datamodule = CombinedDataModule(combined_train_dataset, combined_val_dataset,config)
    return datamodule

