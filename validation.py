from tqdm import tqdm
from data.worldstrat import worldstrat_ds
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import os
import torch
import pandas as pd
from utils.logging_helpers import plot_tensors
from PIL import Image
# import losses
from utils.losses import ssim_loss



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Get Data
config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
ds = worldstrat_ds(config)
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

# Load Model
from model.pix2pix import Px2Px_PL
model = Px2Px_PL(config)
ckpt = torch.load(config.custom_configs.Model.weights_path)
model.load_state_dict(ckpt['state_dict'])
model = model.eval()
print("Loaded (only) Weights from:",config.custom_configs.Model.weights_path)



# ITERATE OVER DATASET

# set empty metrics dict
metrics_dict = {"id":[],
                "x":[],
                "y":[],
                "ssim":[]}
# --- iterate and get metrics ----
for v,batch in tqdm(enumerate(dl),total=len(dl)):

    rgb,nir,coords = batch["rgb"],batch["nir"],batch["coords"],
    pred = model.predict_step(rgb,coords)

    # get metrics:
    ssim = ssim_loss(nir,pred)

    # get info
    x = coords[0][0].item()
    y = coords[0][1].item()
    id_ = v

    # append to list
    metrics_dict["id"].append(id_)
    metrics_dict["x"].append(x)
    metrics_dict["y"].append(y)
    metrics_dict["ssim"].append(ssim.item())

    # plot image
    img = plot_tensors(rgb, nir, pred,title="Worldstrat Validation")
    img.save(f'validation_utils/images/example_image_{id_}.png', 'PNG')

    # save metrics
    df = pd.DataFrame(metrics_dict)
    df.to_csv("validation_utils/worldstrat_metrics.csv")

    if v==10:
        break

