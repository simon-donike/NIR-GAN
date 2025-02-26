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
            satclip_path = "model/satclip/satclip-resnet50-l10.ckpt"
        self.encoder_model =  get_satclip(satclip_path,device=device)
        
        """
        ckpt = torch.load(satclip_path)
        ckpt['hyper_parameters'].pop('eval_downstream')
        ckpt['hyper_parameters'].pop('air_temp_data_path')
        ckpt['hyper_parameters'].pop('election_data_path')
        _ = SatCLIPLightningModule(**ckpt['hyper_parameters'])
        _.load_state_dict(ckpt['state_dict'])
        _.eval()
        self.encoder_model = _.model
        """
        

    def predict(self, x):
        with torch.no_grad():
            x = x.double()
            embeds = self.encoder_model(x)
            embeds = embeds.float().detach()
            return embeds
        
    def forward(self, x):
        print("Don't use fwd, use 'predict' step instead")
        return self.encoder_model(x.double())
    
    
    
# Test 
if __name__=="__main__":
    satclip_path = "/data1/simon/GitHub/NIR_GAN/model/satclip/satclip-resnet50-l10.ckpt"
    model = SatClIP_wrapper(satclip_path)
    print("Amount of Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    from tqdm import tqdm
    for i in tqdm(range(1000)):
        # create random coordinates
        amount = 24
        lon = torch.rand(amount) * 180 - 90  # Scale to [-90, 90]
        lat = torch.rand(amount) * 360 - 180 # Scale to [-180, 180]
        coords = torch.stack((lon,lat),dim=-1)
        embeds = model.predict(coords)
    
