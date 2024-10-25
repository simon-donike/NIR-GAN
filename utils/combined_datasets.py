from torch.utils.data import ConcatDataset
from utils.S2_dataset import S2_datamodule
from utils.SEN2NAIP_v4 import SEN2NAIP_datamodule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


def create_combined_dataset(config):
    datamodule1 = SEN2NAIP_datamodule(config)
    datamodule2 = S2_datamodule(config)
    combined_train_dataset = ConcatDataset([datamodule1.train_dataloader().dataset, datamodule2.train_dataloader().dataset])
    combined_val_dataset = ConcatDataset([datamodule1.val_dataloader().dataset, datamodule2.val_dataloader().dataset])
    

    # Create a new LightningDataModule for the combined dataset
    class CombinedDataModule(LightningDataModule):
        def __init__(self, train_dataset, val_dataset,config):
            super().__init__()
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.train_batch_size = config.Data.train_batch_size
            self.val_batch_size = config.Data.val_batch_size
            self.num_workers = config.Data.num_workers

        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                              num_workers=self.num_workers, shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.val_dataset, num_workers=4, batch_size=self.val_batch_size)
        
    datamodule = CombinedDataModule(combined_train_dataset, combined_val_dataset,config)
    return datamodule
