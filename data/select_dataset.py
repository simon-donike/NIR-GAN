

def dataset_selector(config):
    dataset_type = config.Data.dataset_type
    
    if dataset_type == "S2NAIP":
        from data.S2NAIP_final import S2NAIP_dm
        return S2NAIP_dm(config)
    elif dataset_type == "S2_rand":
        from data.S2_dataset import S2_datamodule
        return S2_datamodule(config)
    elif dataset_type == "worldstrat":
        from data.worldstrat import worldstrat_datamodule
        return worldstrat_datamodule(config)
    elif dataset_type == "S2_75k":
        from data.s2_75k_dataset import S2_75k_datamodule
        return S2_75k_datamodule(config)
    elif dataset_type == "S2_100k":
        from data.s100k_dataset import S2_100k_datamodule
        return S2_100k_datamodule(config)
    
    elif dataset_type == "mixed":
        from data.combined_datasets import CombinedDataModule
        return CombinedDataModule(config)
    
    else:
        raise NotImplementedError(f"Dataset Type {dataset_type} not implemented")
    
if __name__=="__main__":
    from omegaconf import OmegaConf
    import torch

    config = OmegaConf.load("configs/config_px2px.yaml")
    dm = dataset_selector(config)
    
    batch = next(iter(dm.train_dataloader()))
    coords = batch["coords"]    



