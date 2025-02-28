import glob
import rasterio
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import io
from PIL import Image
from rasterio.warp import transform

# surpress plotting warnings
import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)



def get_pred_nirs_and_info(model=None,device=None):
    
    reset_model_mode = False
    if model!=None and model.training:
        model = model.eval()
        reset_model_mode = True
    
    tif_files = glob.glob("validation_utils/time_series_images/*.tif")
    # sort
    tif_files = sorted(tif_files)
    
    rgbs = []
    nirs = []
    nir_preds = []
    timestamps = []
    
    for file in tif_files:
        
        # extract date
        _ = file.split("/")[-1].split(".")[0]
        _ = _.split("_")[1]
        date  = _.split("T")[0]
        timestamps.append(date)
        
        
        with rasterio.open(file) as src:
            img = src.read()  # Read as (bands, H, W)
            h, w = img.shape[1], img.shape[2]
            
            # Get centroid lon/lat
            cx, cy = w // 2, h // 2  # Center pixel coordinates
            lon, lat = src.transform * (cx, cy)  # Convert to geographic coordinates
            # Convert to EPSG:4326 if necessary
            if src.crs and src.crs.to_epsg() != 4326:
                lon, lat = transform(src.crs, "EPSG:4326", [lon], [lat])
                lon, lat = lon[0], lat[0]  # Extract first values from lists
            coords = torch.Tensor([lon, lat])        
                
            # Crop center 512x512
            cx, cy = w // 2, h // 2
            x1, y1 = max(cx - 256, 0), max(cy - 256, 0)
            x2, y2 = min(cx + 256, w), min(cy + 256, h)
            cropped_img = img[:, y1:y2, x1:x2]  # Keep bands dimension
            
            # turn to float16 and replace nan with 0
            cropped_img = cropped_img.astype(np.float32)
            cropped_img = np.nan_to_num(cropped_img, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert to tensor
            img_tensor = torch.tensor(cropped_img, dtype=torch.float32)
            img_tensor = img_tensor / 10000.0
            # extract channels
            rgb = img_tensor[:3]
            nir = img_tensor[3:]
            
            # Predict
            if model is not None:
                if device!=None:
                    rgb = rgb.to(device)
                    coords = coords.to(device)
                    model =  model.to(device)
                
                with torch.no_grad():
                    output = model.predict_step(rgb.unsqueeze(0),coords.unsqueeze(0))
                    output = output.squeeze(0)
                    rgb = rgb.cpu()
                    output = output.cpu()
            else:
                output = nir*1.15 # fake data for testing
            
            # append to lists
            rgbs.append(rgb)
            nirs.append(nir)
            nir_preds.append(output)
            
    rgbs = torch.stack(rgbs)
    nirs = torch.stack(nirs)
    nir_preds = torch.stack(nir_preds)
    
    # reset model training if necessary
    if reset_model_mode:
        model = model.train()
    
    return(rgbs, nirs, nir_preds,timestamps)



def plot_timeline(rgbs, nirs, nir_preds, timestamps,mean_patch_size=32):
    """
    Plots a timeline of centroid pixel values for NIR and NIR predictions,
    and a row of 6 sample RGB images without axes.
    """
    # Extract centroid pixel values
    num_samples = len(timestamps)
    
    # Get image height and width dynamically
    _, h, w = nirs.shape[1:]  # Assuming shape (batch, channels, height, width)
    cx, cy = w // 2, h // 2  # Dynamic centroid calculation
    patch_size = mean_patch_size // 2  # Half of the input patch size
    # Compute the mean NIR value for the centroid patch dynamically
    centroid_nirs = [
        nirs[i, 0, cy - patch_size : cy + patch_size, cx - patch_size : cx + patch_size].mean().item()
        for i in range(num_samples)]
    centroid_preds = [
        nir_preds[i, 0, cy - patch_size : cy + patch_size, cx - patch_size : cx + patch_size].mean().item()
        for i in range(num_samples)]
    
    # Create figure with two subplots (one for the line graph, one for images)
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [2, 1]})

    # --- Line plot for NIR values ---
    axs[0].plot(timestamps, centroid_nirs, marker="o", label="NIR (centroid)", linestyle="-",color="blue")
    axs[0].plot(timestamps, centroid_preds, marker="s", label="Predicted NIR (centroid)", linestyle="--",color="red")
    axs[0].set_ylabel("NIR Value")
    
    # Explicitly set x-ticks to avoid warning
    axs[0].set_xticks(range(len(timestamps)))
    
    timestamps_labels_graph = timestamps.copy()
    timestamps_labels_graph = [timestamp[:4] + "-" + timestamp[4:6] + "-" + timestamp[6:] for timestamp in timestamps_labels_graph]    
    axs[0].set_xticklabels(timestamps_labels_graph, rotation=25)
    
    axs[0].legend()
    axs[0].set_title("Centroid NIR vs. Predicted NIR over Time")
    axs[1].axis("off")

    # --- Row of 6 RGB images (without axes) ---
    num_images = min(6, num_samples)
    selected_indices = np.linspace(0, num_samples - 1, num_images, dtype=int)

    # Create a subplot row for the images
    img_axs = fig.add_axes([0.1, 0.05, 0.8, 0.3])  # Custom position for the image row
    img_axs.axis("off")  # Turn off the entire subplot

    # Create 6 subplots in a row inside axs[1] (for RGB images)
    for i, idx in enumerate(selected_indices):
        ax = fig.add_subplot(2, num_images, num_images + i + 1)  # Place images in second row            
            
        # Crop center 128
        rgb_img = rgbs[idx]
        h, w = rgb_img.shape[1], rgb_img.shape[2]
        cx, cy = w // 2, h // 2
        plot_patch_size=64
        plot_patch_size_half = plot_patch_size // 2
        x1, y1 = max(cx - plot_patch_size_half, 0), max(cy - plot_patch_size_half, 0)
        x2, y2 = min(cx + plot_patch_size_half, w), min(cy + plot_patch_size_half, h)
        rgb_img = rgb_img[:, y1:y2, x1:x2]  # Keep bands dimension
        
        rgb_img = rgb_img.permute(1, 2, 0).numpy()  # Convert to HWC format
        rgb_img = rgb_img*3.5
        rgb_img = np.clip(rgb_img, 0, 1)  # Normalize for display

        ax.imshow(rgb_img)
        
        # Define red box (centroid mean patch)
        mean_patch_half = mean_patch_size // 2
        box_x1 = (plot_patch_size // 2) - mean_patch_half
        box_y1 = (plot_patch_size // 2) - mean_patch_half

        # Create red rectangle
        rect = patches.Rectangle((box_x1, box_y1), mean_patch_size, mean_patch_size, 
                                linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)  # Add red box
        
        
        #ax.set_title(timestamps[idx], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        timestamp_label = timestamps[idx]
        timestamp_label = timestamp_label[:4] + "-" + timestamp_label[4:6] + "-" + timestamp_label[6:]
        ax.set_xlabel(timestamp_label, fontsize=10, labelpad=5)
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjusts vertical spacing
    #plt.savefig("validation_utils/timeline_plot.png")
    #plt.close()   
    
    # Save and return image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close()
    return pil_image
    

def calculate_and_plot_timeline(model=None,device=None,mean_patch_size=16):
    r,n,p,t  = get_pred_nirs_and_info(model,device)
    im = plot_timeline(r,n,p,t,mean_patch_size)
    return im


if __name__ == "__main__":
    r,n,p,t  = get_pred_nirs_and_info(None)
    im = plot_timeline(r,n,p,t,mean_patch_size=16)
    im.save("validation_utils/timeline_plot.png")
