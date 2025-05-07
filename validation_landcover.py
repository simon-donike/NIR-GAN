from tqdm import tqdm
from data.worldstrat import worldstrat_ds
from data.s100k_dataset import S2_100k
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import os
import torch
import pandas as pd
from utils.logging_helpers import plot_tensors,plot_index
from PIL import Image
import kornia
from validation_utils.val_utils import crop_center
from utils.remote_sensing_indices import RemoteSensingIndices

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Get Data
satclip=False
if satclip:
    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
else:
    config = OmegaConf.load("configs/config_px2px.yaml")

#ds = worldstrat_ds(config)
ds = S2_100k(config,phase="train")
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

# Load Model
from model.pix2pix import Px2Px_PL
model = Px2Px_PL(config)
ckpt = torch.load(config.custom_configs.Model.weights_path,map_location=device)
model.load_state_dict(ckpt['state_dict'])
model = model.eval()
print("Loaded (only) Weights from:",config.custom_configs.Model.weights_path)


# ITERATE OVER DATASET
# set empty metrics dict
metrics_dict = {"id":[],
                "region_id":[],
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
    
        # Assume you also have: batch["clc_mask"] with shape [B, H, W]
    clc_mask = batch["clc_mask"]

    for i in range(rgb.shape[0]):  # iterate over batch
        true_nir = nir[i, 0].cpu().numpy()
        pred_nir = pred[i, 0].cpu().numpy()
        region_mask = clc_mask[i].cpu().numpy()

        region_ids = np.unique(region_mask)
        for region_id in region_ids:
            mask = (region_mask == region_id)

            if np.count_nonzero(mask) < 10:
                continue  # Skip small regions

            # Extract region patches
            t_patch = true_nir[mask]
            p_patch = pred_nir[mask]

            # Create 2D masked images for SSIM and PSNR
            t_img = np.zeros_like(true_nir); t_img[mask] = t_patch
            p_img = np.zeros_like(pred_nir); p_img[mask] = p_patch

            # SSIM requires 2D arrays
            ssim_val = ssim(t_img, p_img, data_range=1.0)
            psnr_val = psnr(t_img, p_img, data_range=1.0)
            
            l1 = torch.mean(torch.abs(nir - pred)).item()
            l2 = torch.mean((nir - pred) ** 2).item()
            
            # append metrics and info to dict
            metrics_dict["id"].append(v)
            metrics_dict["region_id"].append(region_id)
            metrics_dict["x"].append(coords[i][0].item())
            metrics_dict["y"].append(coords[i][1].item())
            metrics_dict["ssim"].append(ssim_val)
            metrics_dict["psnr"].append(psnr_val)
            metrics_dict["l1"].append(l1)
            metrics_dict["l2"].append(l2)

            print(f"Region {region_id}: SSIM={ssim_val:.3f}, PSNR={psnr_val:.2f}")
            
# dict to pd
df = pd.DataFrame(metrics_dict)
df.to_csv("validation_utils/validation_utils/CLC_metrics.csv",index=False)