from torch.utils.data import ConcatDataset, Subset, ChainDataset
from utils.S2_dataset import S2_datamodule, S2_rand_dataset
from utils.S2NAIP_final import S2NAIP_dm, SEN2NAIPv2
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

            train_dataset1 = SEN2NAIPv2(self.config,phase="train")
            train_dataset2 = S2_rand_dataset(self.config,phase="train")
            self.train_dataset = ConcatDataset([train_dataset1, train_dataset2])
            
            val_dataset1 = SEN2NAIPv2(self.config,phase="val")
            val_dataset2 = S2_rand_dataset(self.config,phase="val")
            self.val_dataset = ConcatDataset([val_dataset1, val_dataset2])
        
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
        
"""
  File "/usr/lib/python3.10/selectors.py", line 416, in select
    fd_event_list = self._selector.poll(timeout)
  File "/data1/simon/envs/bare2/lib/python3.10/site-packages/torch/utils/data/_utils/signal_handling.py", line 66, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 3983472) is killed by signal: Aborted. 
"""


"""
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
            return DataLoader(self.val_dataset, num_workers=4, batch_size=self.val_batch_size, shuffle=True)
        
    # instanciate object
    datamodule = CombinedDataModule(combined_train_dataset, combined_val_dataset,config)
    return datamodule
"""