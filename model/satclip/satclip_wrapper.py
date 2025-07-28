import torch
import pytorch_lightning as pl
from model.satclip.load import get_satclip
from model.satclip.main import *


class SatClIP_wrapper(pl.LightningModule):
    """
    Wrapper around the SatClIP model to use it in Pytorch Lightning.
    """
    def __init__(self, satclip_path=None,device="cpu"):
        super().__init__()
        if satclip_path is None:
            satclip_path = "/data1/simon/GitHub/NIR-GAN/model/satclip/satclip-resnet50-l10.ckpt"
        self.encoder_model =  get_satclip(satclip_path,device=device)
        
        # set all params to freeze
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        

    def predict(self, x):
        with torch.no_grad():
            x = x.double()
            embeds = self.encoder_model(x)
            embeds = embeds.float().detach()
            return embeds
        
    def forward(self, x):
        with torch.no_grad():
            print("Don't use fwd, use 'predict' step instead")
            return self.encoder_model(x.double())
    
    
    
# Test 
if __name__=="__main__":
    satclip_path = "/data1/simon/GitHub/NIR-GAN/model/satclip/satclip-resnet50-l10.ckpt"
    model = SatClIP_wrapper(satclip_path)
    print("Amount of Parameters:", sum(p.numel() for p in model.parameters()))
    
    
# Run inf on an unrelated dataset
if __name__=="__main__" and False:
    # load data
    import pandas as pd
    import os
    from shapely import wkt
    import numpy as np
    from tqdm import tqdm
    df = pd.read_pickle("/data1/simon/GitHub/latent-diffusion/md_satclip.pkl")
    for coor in tqdm(df["stac:centroid"]):
        pt = wkt.loads(coor)
        lon = pt.x
        lat = pt.y  
        embedding = model.predict(torch.tensor([[lon, lat]]))
        embedding_np = embedding.detach().cpu().numpy()
        out_path = "/data3/SEN2NAIP_global/satclip_embs"
        filename = f"satclip_{lon:.5f}_{lat:.5f}"
        filename = os.path.join(out_path, filename)
        # now save compressed embedding
        np.savez_compressed(filename, embedding=embedding_np)             
        
        
    def load_embedding(lon, lat):
        filename = f"satclip_{lon:.5f}_{lat:.5f}.npz"
        arr = np.load(filename)
        embedding = torch.from_numpy(arr["embedding"])
        return embedding   
    
        
