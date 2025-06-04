from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset, DataLoader
import os
from skimage.exposure import match_histograms
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# GPU Settings --------
# set visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


# 2. --------
# Get Model and weights
from model.pix2pix import Px2Px_PL
config = OmegaConf.load("configs/config_px2px.yaml") # Get config file
# GET CKPT FROM https://huggingface.co/simon-donike/NIR-GAN
ckpt_path = "logs/best/S2.ckpt" # Path to checkpoint file
model = Px2Px_PL(config) # Instantiate the model
model.load_state_dict(torch.load(ckpt_path)['state_dict'],strict=False) # load into model
model = model.eval() # set to eval mode
model = model.to(device) # send to device



# 3. --------
# Define Predict and save Functions
def histogram_match(image,reference):
    
    # interpolate S2 to HR size
    reference = F.interpolate(reference, size=image.shape[-2:], mode='bilinear', align_corners=False)
    
    # Align Batches
    matched = []
    for img, ref in zip(image, reference):  # iterate over batch
        img_np = img.squeeze().cpu().numpy()
        ref_np = ref.squeeze().cpu().numpy()
        matched_np = match_histograms(img_np, ref_np, channel_axis=None)
        matched.append(torch.from_numpy(matched_np).unsqueeze(0))  # [1, H, W]
    image = torch.stack(matched, dim=0) # stack batches
    return image # return matched tensor

def save_image(pred_nir, out_path,name):
    # save image as tif to path
    out_filename = os.path.join(out_path,f"{name}")
    np.savez_compressed(out_filename, nir=pred_nir.numpy())
    
def plot_example(lr,hr,pred_nir,real_nir,idx):
    # Take first image in batch
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        hr = (hr*3).clamp(0, 1)  # [3, H, W]
        lr = (lr*3).clamp(0, 1)
        
        # HR RGB
        hr_rgb = hr[0][:3]  # [3, H, W]
        axs[0].imshow(hr_rgb.permute(1, 2, 0).clamp(0, 1).numpy())
        axs[0].set_title("HR RGB")

        # LR RGB (upsampled)
        lr_rgb = F.interpolate(lr[0][:3].unsqueeze(0), size=hr_rgb.shape[1:], mode="nearest")[0]
        axs[1].imshow(lr_rgb.permute(1, 2, 0).clamp(0, 1).numpy())
        axs[1].set_title("LR RGB")

        # Synth NIR
        axs[2].imshow(pred_nir[0][0].numpy(), cmap="gray")
        axs[2].set_title("Synth NIR")

        # Real NIR (upsampled)
        real_nir = F.interpolate(s2_nir[0].unsqueeze(0), size=hr_rgb.shape[1:], mode="nearest")[0]
        axs[3].imshow(real_nir[0].numpy(), cmap="gray")
        axs[3].set_title("Real NIR")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"images/synth_NIR_2/example_{idx}.png")
        plt.close()


# 4. --------
# Get Data
from data.SR_dataset_RGB import SR_dataset
dataset  = SR_dataset(root_dir = "data/synthDataset")
dl = DataLoader(dataset, batch_size=2, shuffle=False)
nir_out_path = "data/synthDataset/synth_nirs" # Path to save synthetic NIR images
if not os.path.exists(nir_out_path):
    os.makedirs(nir_out_path) # Create directory if it does not exist
    
# 5. --------
# Predict and save NIR
for v,batch in tqdm(enumerate(dl),total=len(dl), desc="Predicting NIR..."):
    # Get Batch Data
    lr,hr,s2_nir,name,coor = batch["lr"],batch["hr"],batch["s2_nir"],batch["id"],batch["coords"]
    lr,hr, = lr.to(device),hr.to(device)
    
    # Get Prediction
    with torch.no_grad():
        pred = model(hr)
    lr,hr,s2_nir,pred = lr.cpu(),hr.cpu(),s2_nir.cpu(),pred.cpu()
    
    # Histogram Match prediction to S2 NIR - LR
    s2_nir_int = torch.nn.functional.interpolate(s2_nir,scale_factor=4)  # interpolate to HR size for int.
    pred_nir = histogram_match(image=pred, reference=s2_nir_int)
    
    # Save Image
    for im,t_id in zip(pred_nir,name):
        # change to float 16
        im = im.to(torch.float16)
        save_image(im, out_path=nir_out_path, name=t_id)
        
    # Save Example
    if v % 10 == 0:
        # Plot example
        plot_example(lr=lr,hr=hr,pred_nir=pred_nir,real_nir=s2_nir,idx=v)
        print(f"Step {v} - Example saved.")