

def dataset_selector(config):
    dataset_type = config.Data.dataset_type
    
    if dataset_type == "S2NAIP":
        from utils.S2NAIP_final import S2NAIP_dm
        return S2NAIP_dm(config)
    elif dataset_type == "S2_rand":
        from utils.S2_dataset import S2_datamodule
        return S2_datamodule(config)
    elif dataset_type == "mixed":
        from utils.combined_datasets import CombinedDataModule
        return CombinedDataModule(config)
    else:
        raise NotImplementedError(f"Dataset Type {dataset_type} not implemented")
    
if __name__=="__main__":
    from omegaconf import OmegaConf
    import torch

    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
    dm = dataset_selector(config)
    
    batch = next(iter(dm.train_dataloader()))
    coords = batch["coords"]    

    # Write out coordinates to verify lon/lats
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    for v,batch in tqdm(enumerate(dm.train_dataloader()),total=len(dm.train_dataloader())):
        coords = torch.cat((coords,batch["coords"]),dim=0)
    coords = coords.cpu().numpy()
    df = pd.DataFrame(coords)
    df.columns = ["lon","lat"]
    df.to_csv("coords_mixed.csv",index=True)


