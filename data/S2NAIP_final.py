import torch
from torch.utils.data import Dataset, DataLoader
import mlstac
from rasterio import CRS, Affine
import os
from io import BytesIO
import numpy as np
import h5py
import pytorch_lightning as pl
#from albumentations import Compose, HorizontalFlip, RandomRotate90, ShiftScaleRotate
#from albumentations.pytorch import ToTensorV2


# Define your dataset class
class SEN2NAIPv2(Dataset):
    def __init__(self, config, phase="train"):        
        # extract infos from config
        self.config = config
        self.image_size = self.config.Data.sen2naip_settings.image_size
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
        
        # set padding settings
        if self.config.Data.padding:
            self.pad = torch.nn.ReflectionPad2d(self.config.Data.padding_amount)
            
        # set return coords condition
        if "return_coords" in config.Data.sen2naip_settings:
            self.return_coords = config.Data.sen2naip_settings.return_coords
        else:
            self.return_coords = False

        print("Instanciated SEN2NAIPv2 dataset with",len(self.dataset.metadata),"datapoints for phase: ",phase)

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
        lr,hr,metadata = self.get_data(datapoint)
        hr = hr.transpose(2,0,1)
        hr = hr[:,:self.image_size,:self.image_size]
        hr = torch.tensor(hr).float()
        hr = self.get_b4(hr)
        hr = hr/10000.
        
        rgb = hr[:3,:,:]
        nir = hr[3,:,:]
        
        # apply augmentation
        if False:
            if self.phase in ["train","validation"]:
                nir = nir.unsqueeze(0)
                rgb,nir = self.augment_images(rgb.numpy(),nir.numpy())
                rgb = torch.tensor(rgb).float()
                nir = torch.tensor(nir).float().squeeze(0)
                
        # apply padding
        if self.config.Data.padding:
            rgb = self.pad(rgb)
            nir = self.pad(nir.unsqueeze(0)).squeeze(0)
            
        # assure shapes match
        if len(nir.shape)==2:
            nir = nir.unsqueeze(0)
            
        # if wanted, return coordinates as well
        return_dict = {"rgb":rgb, "nir": nir}
        
        # add location to dictionary if wanted
        if self.return_coords == True:
            coords = self.get_centroid_from_metadata(metadata)
            return_dict["coords"] = coords

        return return_dict
        
    def augment_images(self,rgb,nir):
        # 1. value stretch
        if np.random.randint(0,100) > 101:
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
        
    def get_data(self,datapoint):
        data_bytes = mlstac.get_data(dataset=datapoint,
            backend="bytes",
            save_metadata_datapoint=True,
            quiet=True)

        with BytesIO(data_bytes[0][0]) as f:
            with h5py.File(f, "r") as g:
                metadata = eval(g.attrs["metadata"])
                lr1 = np.moveaxis(g["input"][0:4], 0, -1)
                hr1 = np.moveaxis(g["target"][0:4], 0, -1)
        lr1 = lr1.astype(np.float32)
        hr1 = hr1.astype(np.float32)

        return(lr1,hr1,metadata)
    
    def get_centroid_from_metadata(self,metadata):
        from pyproj import Transformer
        # Extract Information
        width = metadata["hr_profile"]["width"]
        height = metadata["hr_profile"]["height"]
        crs = metadata["hr_profile"]["crs"]
        trans = metadata["hr_profile"]["transform"]
        # Get middle of image
        centroid_x = width / 2
        centroid_y = height / 2
        
        # Get geo coordinates
        geo_x, geo_y = trans * (centroid_x, centroid_y)
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(geo_x, geo_y)
        return(torch.Tensor([lon,lat]))
        


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
                          shuffle=True, num_workers=self.num_workers,
                          prefetch_factor=self.config.Data.prefetch_factor,drop_last=True,
                          persistent_workers=self.config.Data.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,batch_size=self.config.Data.val_batch_size,
                          shuffle=True, num_workers=self.num_workers,drop_last=True,)



if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/config_px2px.yaml")
    pl_datamodule = S2NAIP_dm(config)
    b = pl_datamodule.dataset_train[1]
    rgb,nir = b["rgb"],b["nir"]
    
    
    """
    Ablation over DataLoader Settings
    """    
    import time
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    def benchmark_dataloader(config, num_workers_list=[2, 4, 8, 16, 24, 32], prefetch_list=[2, 4, 6, 8]):
        """ Benchmark DataLoader speed with different num_workers and prefetch settings """
        
        pl_datamodule = S2NAIP_dm(config)
        dataset = pl_datamodule.dataset_train  # Assuming dataset_train is the dataset

        batch_size = 24  # Load batch size from config

        print(f"\n🚀 Benchmarking DataLoader for batch_size={batch_size}")
        results = []

        for num_workers in num_workers_list:
            for prefetch_factor in prefetch_list:
                # Initialize DataLoader with different settings
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor if num_workers > 0 else 0,
                    persistent_workers=True if num_workers > 0 else False,
                    pin_memory=True if torch.cuda.is_available() else False,
                )

                # Measure data loading speed
                start_time = time.time()
                num_batches = 250  # Measure over n batches
                
                for i, batch in enumerate(dataloader):
                    if i >= num_batches:
                        break  # Stop after measuring 10 batches
                
                end_time = time.time()
                time_per_batch = (end_time - start_time) / num_batches
                print(f"🟢 num_workers={num_workers}, prefetch_factor={prefetch_factor} → {time_per_batch:.4f} sec/batch")
                results.append((num_workers, prefetch_factor, time_per_batch))

        # Find best setting
        best_config = min(results, key=lambda x: x[2])  # Min time per batch
        print(f"\n🚀 Best setting: num_workers={best_config[0]}, prefetch_factor={best_config[1]} → {best_config[2]:.4f} sec/batch")
        return best_config

    # Run benchmark
    best_setting = benchmark_dataloader(config)