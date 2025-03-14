from torch.utils.data import ConcatDataset, DataLoader
from pytorch_lightning import LightningDataModule
import importlib

# Mapping dataset names to their respective classes
DATASET_CLASSES = {
    "SEN2NAIP": "data.SEN2NAIP_dataset",
    "S2_rand_dataset": "data.S2_dataset",
    "S2_75k": "data.s2_75k_dataset",
    "S2_100k": "data.s100k_dataset",
    "worldstrat": "data.worldstrat"
}

class CombinedDataModule(LightningDataModule):
    def __init__(self, config, datasets):
        """
        Args:
            config: Configuration object
            train_datasets: List of dataset names for training
            val_datasets: List of dataset names for validation
        """
        super().__init__()
        self.config = config
        self.train_batch_size = config.Data.train_batch_size
        self.val_batch_size = config.Data.val_batch_size
        self.num_workers = config.Data.num_workers

        # Load datasets dynamically based on provided names
        self.train_dataset = self._load_datasets(datasets, phase="train")
        print("Combined Train Dataset Length:", len(self.train_dataset))

        self.val_dataset = self._load_datasets(datasets, phase="val")
        print("Combined Val Dataset Length:", len(self.val_dataset))

    def _load_datasets(self, dataset_names, phase):
        """Load and concatenate datasets based on names."""
        datasets = []
        for name in dataset_names:
            if name in DATASET_CLASSES:
                module_name = DATASET_CLASSES[name]
                print("importing ", module_name)
                dataset_class = getattr(importlib.import_module(module_name), name)
                datasets.append(dataset_class(self.config, phase=phase))
            else:
                raise ValueError(f"Unknown dataset: {name}")
        
        return ConcatDataset(datasets) if datasets else None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size,
                          num_workers=self.num_workers, shuffle=False)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
    
    # Example: Specify which datasets to load
    datasets = ["worldstrat"]

    dm = CombinedDataModule(config, datasets)