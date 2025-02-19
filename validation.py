from tqdm import tqdm
from data.worldstrat import worldstrat_ds
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import os
import torch
import pandas as pd
from utils.logging_helpers import plot_tensors
from PIL import Image
import kornia
from validation_utils.val_utils import crop_center



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
                "ssim":[],
                "psnr":[],
                "l1":[],
                "l2":[],
                }
# --- iterate and get metrics ----
for v,batch in tqdm(enumerate(dl),total=len(dl)):

    rgb,nir,coords = batch["rgb"],batch["nir"],batch["coords"],
    pred = model.predict_step(rgb,coords)

    # crop center
    rgb = crop_center(rgb.squeeze(0),450).unsqueeze(0)
    nir = crop_center(nir.squeeze(0),450).unsqueeze(0)
    pred = crop_center(pred.squeeze(0),450).unsqueeze(0)

    # get info
    x = coords[0][0].item()
    y = coords[0][1].item()
    id_ = v

    # get metrics:
    ssim = kornia.metrics.ssim(nir, pred, window_size=11).mean()
    psnr = kornia.metrics.psnr(nir, pred, max_val=1.0)
    l1 = torch.mean(torch.abs(nir - pred))
    l2 = torch.mean((nir - pred) ** 2)

    # append to list
    metrics_dict["id"].append(id_)
    metrics_dict["x"].append(x)
    metrics_dict["y"].append(y)
    metrics_dict["ssim"].append(ssim.item())
    metrics_dict["psnr"].append(psnr.item())
    metrics_dict["l1"].append(l1.item())
    metrics_dict["l2"].append(l2.item())

    # plot image
    if id_%10==0:
        img = plot_tensors(rgb, nir, pred,title="Worldstrat Validation")
        img.save(f'validation_utils/images/example_image_{id_}.png', 'PNG')

    # save metrics
    df = pd.DataFrame(metrics_dict)
    if id_%25==0:
        df.to_csv("validation_utils/worldstrat_metrics.csv")

    if v==-1:
        break


# Get Context info for dataset
from validation_utils.geo_ablation import append_info_to_df
df = append_info_to_df(df)

