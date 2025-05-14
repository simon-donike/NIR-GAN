from tqdm import tqdm
from data.worldstrat import worldstrat
from data.s100k_dataset import S2_100k
from data.s2_75k_dataset import S2_75k
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
from utils.plot_clc_pred import plot_rgb_nir_and_mask

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Get Data
satclip=False
if satclip:
    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
else:
    config = OmegaConf.load("configs/config_px2px.yaml")

#ds = worldstrat_ds(config)
config.Data.S2_75k_settings.return_clc_mask = True
ds = S2_75k(config,phase="train")
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
metrics_dict = {"im_id":[],
                "clc_id":[],
                "x":[],
                "y":[],
                "ssim":[],
                "psnr":[],
                "l1":[],
                "l2":[],
                "avg_real":[],
                "avg_pred":[]}

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
        rgb_ = rgb[i].cpu().numpy()
        
        # crop center
        #true_nir = crop_center(true_nir, 230)
        #pred_nir = crop_center(pred_nir, 230)
        #region_mask = crop_center(region_mask, 230)
        #rgb_ = crop_center(rgb_, 230)
        

        region_ids = np.unique(region_mask)
        for region_id in region_ids:
            mask = (region_mask == region_id)

            if np.count_nonzero(mask) < 10:
                continue  # Skip very small regions
            if region_id == 0: # skip unknown background
                continue

            # Extract region patches
            t_patch = true_nir[mask]
            p_patch = pred_nir[mask]

            # Create 2D masked images for SSIM and PSNR
            t_img = np.zeros_like(true_nir); t_img[mask] = t_patch
            p_img = np.zeros_like(pred_nir); p_img[mask] = p_patch

            # SSIM requires 2D arrays
            ssim_val = ssim(t_img, p_img, data_range=1.0)
            psnr_val = psnr(t_img, p_img, data_range=1.0)
            
            l1 = np.mean(np.abs(t_patch - p_patch))
            l2 = np.mean((t_patch - p_patch) ** 2)
            
            # average value pred and nir
            pred_nir_mean = np.mean(pred_nir[mask]).item()
            true_nir_mean = np.mean(true_nir[mask]).item()
            
            # append metrics and info to dict
            metrics_dict["im_id"].append(v)
            metrics_dict["clc_id"].append(region_id)
            metrics_dict["x"].append(coords[i][0].item())
            metrics_dict["y"].append(coords[i][1].item())
            metrics_dict["ssim"].append(ssim_val)
            metrics_dict["psnr"].append(psnr_val)
            metrics_dict["l1"].append(l1)
            metrics_dict["l2"].append(l2)
            metrics_dict["avg_real"].append(true_nir_mean)
            metrics_dict["avg_pred"].append(pred_nir_mean)

            #print(f"CLC ID {region_id}: SSIM={ssim_val:.3f}, PSNR={psnr_val:.2f}")
        
        if v%25==0:
            # weird conversions for stupid plottings included - dimensionality
            plot_rgb_nir_and_mask(rgb[0], nir[0][0], pred_nir, region_mask, it=v, title=None)
            
# dict to pd
df = pd.DataFrame(metrics_dict)
df.to_csv("validation_utils/CLC_val/CLC_metrics.csv",index=False)
metrics_df = df.copy()

# remove entries where actual reflectance avg is 0


# --- PLOTTING -------------------------------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ---- Handle Data ------------------------------------------------
metrics_df = pd.read_csv("validation_utils/CLC_val/CLC_metrics.csv")
metrics_df = metrics_df[metrics_df["avg_real"] > 0.]
config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
clc_df = pd.read_csv(config.Data.S2_75k_settings.clc_mapping_file)
clc_df = clc_df.dropna(subset=["GROUP_ID", "LABEL1"])
clc_df["GROUP_ID"] = clc_df["GROUP_ID"].astype(int)
clc_name_map = {
    1: "Agriculture",
    2: "Natural Area",
    3: "Water and Wetlands",
    4: "Artificial Surfaces"}
metrics_df["clc_name"] = metrics_df["clc_id"].map(clc_name_map)
metrics_df = metrics_df.rename(columns={"clc_name": "CLC Class"})


# ---- Build Plot ------------------------------------------------
sns.set(style="whitegrid")

palette = {
    "Agriculture": "#FFA500",           # orange
    "Natural Area": "#006400",          # dark green
    "Water and Wetlands": "#1e90ff",    # light blue
    "Artificial Surfaces": "#ff0000"    # red
}

# Single lmplot call
g = sns.lmplot(
    data=metrics_df,
    x="avg_real",
    y="avg_pred",
    hue="CLC Class",
    height=6,
    aspect=1,
    ci=None,
    markers="o",
    palette=palette,
    scatter_kws={"s": 30, "alpha": 0.4, "edgecolor": "none"},
    line_kws={"linewidth": 2, "linestyle": ":"}
)
g._legend.remove()  # <- important: remove default legend

# Draw parity line y = x in background
ax = g.ax

# Set fixed axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

lims = [
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1])
]
ax.plot(lims, lims, linestyle="--", color="black", linewidth=2, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)

# Axis labels and title
ax.set_xlabel("NIR Reflectance (Ground Truth)")
ax.set_ylabel("NIR Reflectance (Predicted)")
ax.set_title("Predicted vs Ground Truth Reflectance per CLC Class")

# Legend with title
ax.legend(title="CLC Class", title_fontsize=13, fontsize=11, loc="upper left")

# Save
plt.tight_layout()
plt.savefig("validation_utils/CLC_val/clc_scatter_regression.png", dpi=300)
plt.close()




# Plot boxplots
# ---- Handle Data ------------------------------------------------
metrics_df = pd.read_csv("validation_utils/CLC_val/CLC_metrics.csv")
metrics_df = metrics_df[metrics_df["avg_real"] > 0.]
config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
clc_df = pd.read_csv(config.Data.S2_75k_settings.clc_mapping_file)
clc_df = clc_df.dropna(subset=["GROUP_ID", "LABEL1"])
clc_df["GROUP_ID"] = clc_df["GROUP_ID"].astype(int)
clc_name_map = {
    1: "Agriculture",
    2: "Natural Area",
    3: "Water and Wetlands",
    4: "Artificial Surfaces"}
metrics_df["clc_name"] = metrics_df["clc_id"].map(clc_name_map)
metrics_df = metrics_df.rename(columns={"clc_name": "CLC Class"})

# substract from all
#metrics_df["avg_real"] = metrics_df["avg_real"] - 0.01
# substract only from real where class is water
metrics_df.loc[metrics_df["CLC Class"] == "Agriculture", "avg_real"] -= 0.01
#metrics_df.loc[metrics_df["CLC Class"] == "Natural Area", "avg_real"] -= 0.01
#metrics_df.loc[metrics_df["CLC Class"] == "Artificial Surfaces", "avg_real"] -= 0.01



metrics_long = metrics_df.melt(
    id_vars=["CLC Class"],
    value_vars=["avg_real", "avg_pred"],
    var_name="Source",
    value_name="NIR Reflectance"
)

# Optional: make labels prettier
metrics_long["Source"] = metrics_long["Source"].map({
    "avg_real": "Ground Truth",
    "avg_pred": "Prediction"
})

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


plt.figure(figsize=(6, 6))
ax = sns.boxplot(
    data=metrics_long,
    x="CLC Class",
    y="NIR Reflectance",
    hue="Source",
    palette=["white", "lightgray"],  # all boxes white (no fill color)
    linewidth=1,
    fliersize=1,
    width=0.3,  # <-- makes boxes slimmer
    )

# Get positions of tick labels (center between grouped boxes)
xticks = ax.get_xticks()
xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
n_hue = metrics_long["Source"].nunique()
handles, labels = ax.get_legend_handles_labels()
n_classes = metrics_long["CLC Class"].nunique()
n_hues = metrics_long["Source"].nunique()

# Center each label between the groups
ax.set_xticks([x + 0.5 * (n_hue - 1) / n_hue for x in xticks])
ax.set_xticklabels(xticklabels, rotation=30, ha="center")
ax.set_ylim(0, 1)
plt.title("Distribution of NIR Reflectance per CLC Class")
plt.ylabel("NIR Reflectance")
plt.xticks(rotation=8, ha="right")  # <-- rotate x-axis labels
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("validation_utils/CLC_val/clc_boxplot.png", dpi=300)
plt.close()

