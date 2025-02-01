import torch
from torch.utils.data import Dataset, DataLoader
import mlstac
from rasterio import CRS, Affine
import os
from io import BytesIO
import numpy as np
import h5py
import pytorch_lightning as pl


# Define your dataset class
class SEN2NAIPv2(Dataset):
    def __init__(self, config, phase="train"):        
        # extract infos from config
        self.config = config
        base_path = config.Data.sen2naip_settings.base_path
        dataset_type = config.Data.sen2naip_settings.dataset_type

        assert dataset_type in ["real","synthetic1","synthetic2"],"Dataset type not found. Choose from ['real','synthetic-v1','synthetic-v2']"
        self.path = os.path.join(base_path,'SEN2NAIPv2-'+dataset_type,"main.json")
        assert os.path.exists(self.path), "Dataset not found. Please check the path."
        self.dataset = mlstac.load(self.path,force=True)

        # train-test-val split
        if phase=="val":
            phase="validation"
        if "split" in self.dataset.metadata.columns:
            self.dataset.metadata = self.dataset.metadata[self.dataset.metadata["split"]==phase]
        else:
            from sklearn.model_selection import train_test_split
            # Assuming df is your pandas dataframe
            train_val, test = train_test_split(self.dataset.metadata, test_size=0.1,random_state=42)  # 80% train+val, 20% test
            train, val = train_test_split(train_val, test_size=0.10,random_state=42)  # 20% val from 80% -> 60% train, 20% val
            if phase=="train":
                self.dataset.metadata = train
            elif phase=="validation":
                self.dataset.metadata = val
            elif phase=="test":
                self.dataset.metadata = test
        self.phase=phase

        print("Instanciated SEN2NAIPv2 dataset with",len(self.dataset.metadata),"datapoints for",phase)

    def __len__(self):
        return len(self.dataset.metadata)

    def get_b4(self,t):
        if t.shape[0]==3:
            average = torch.mean(t, dim=0)
            result = torch.cat((t, average.unsqueeze(0)), dim=0)
            return(result)
        else:
            return(t)

    def __getitem__(self, idx):
        datapoint = self.dataset.metadata.iloc[idx]
        lr,hr = self.get_data(datapoint)
        lr = lr.transpose(2,0,1)
        hr = hr.transpose(2,0,1)
        lr = lr[:,:128,:128]
        hr = hr[:,:512,:512]
        lr,hr = torch.tensor(lr).float(),torch.tensor(hr).float()
        lr = self.get_b4(lr)
        hr = self.get_b4(hr)
        lr = lr/10000.
        hr = hr/10000.
        
        # get random number between 0 and 1
        #random_number = np.random.rand()
        #if random_number>0.85:
        #    pass     
        return {"rgb": hr[:3,:,:], "nir": hr[3:,:,:]}

    def get_data(self,datapoint):
        data_bytes = mlstac.get_data(dataset=datapoint,
            backend="bytes",
            save_metadata_datapoint=True,
            quiet=True)

        with BytesIO(data_bytes[0][0]) as f:
            with h5py.File(f, "r") as g:
                #metadata = eval(g.attrs["metadata"])
                lr1 = np.moveaxis(g["input"][0:4], 0, -1)
                hr1 = np.moveaxis(g["target"][0:4], 0, -1)
        lr1 = lr1.astype(np.float32)
        hr1 = hr1.astype(np.float32)

        return(lr1,hr1)
    


class S2NAIP_dm(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        # Initialize the dataset
        self.config=config
        self.num_workers = config.Data.num_workers
        self.dataset_train = SEN2NAIPv2(config,phase="train")
        self.dataset_val = SEN2NAIPv2(config,phase="val")

    def train_dataloader(self):        
        return DataLoader(self.dataset_train,batch_size=self.config.Data.train_batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,batch_size=self.config.Data.val_batch_size,
                          shuffle=True, num_workers=self.num_workers,drop_last=True)





if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/config_NIR.yaml")
    pl_datamodule = S2NAIP_dm(config)

