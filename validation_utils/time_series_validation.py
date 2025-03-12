import glob
import rasterio
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image
from rasterio.warp import transform
from matplotlib.gridspec import GridSpec
import gc


# surpress plotting warnings
import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)



def get_pred_nirs_and_info(model=None,device=None,root_dir="validation_utils/time_series_bavaria/*.tif",size_input=256):
    
    reset_model_mode = False
    if model!=None and model.training:
        model = model.eval()
        reset_model_mode = True
    
    tif_files = glob.glob(root_dir)
    # sort
    tif_files = sorted(tif_files)
    
    rgbs = []
    nirs = []
    nir_preds = []
    timestamps = []
    
    for file in tif_files:
        
        # extract date
        _ = file.split("/")[-1].split(".")[0]
        if "SKIP" in _:
            continue
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
            size_input_half = size_input//2
            cx, cy = w // 2, h // 2
            x1, y1 = max(cx - size_input_half, 0), max(cy - size_input_half, 0)
            x2, y2 = min(cx + size_input_half, w), min(cy + size_input_half, h)
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
                    output = output.detach().cpu()
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

    gc.collect()
    torch.cuda.empty_cache()
    del model
    
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
    axs[0].set_title("NIR vs. Predicted NIR")
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
    pil_image = Image.open(buf).copy()
    plt.close()
    buf.close()
    return pil_image
    


def plot_ndvi_timeline(rgbs, nirs, nir_preds, timestamps, mean_patch_size=32):
    """
    Plots a timeline of centroid NDVI values (true and predicted) and rows of images: 
    RGB, NDVI (True), and NDVI (Predicted).
    """
    # Crop center for plotting
    h, w = rgbs.shape[-1], rgbs.shape[-2]
    cx, cy = w // 2, h // 2
    plot_patch_size = 64
    plot_patch_size_half = plot_patch_size // 2
    x1, y1 = max(cx - plot_patch_size_half, 0), max(cy - plot_patch_size_half, 0)
    x2, y2 = min(cx + plot_patch_size_half, w), min(cy + plot_patch_size_half, h)

    rgbs = rgbs[:, :, y1:y2, x1:x2]
    nirs = nirs[:, :, y1:y2, x1:x2]
    nir_preds = nir_preds[:, :, y1:y2, x1:x2]

    num_samples = len(timestamps)
    _, h, w = nirs.shape[1:]
    cx, cy = w // 2, h // 2
    patch_size = mean_patch_size // 2

    # Extract Red channel (assuming Red is at index 0)
    reds = rgbs[:, 0, :, :]

    # Compute NDVI and NDVI predictions
    def compute_ndvi(nir, red):
        return (nir - red) / (nir + red + 1e-6)  # Avoid division by zero

    ndvi_true = compute_ndvi(nirs[:, 0, :, :], reds)
    ndvi_pred = compute_ndvi(nir_preds[:, 0, :, :], reds)
    
    # Define the window size for NDVI computation
    mean_patch_half = mean_patch_size // 2
    shift_x = 3  # Number of pixels to shift
    shift_y = 10  # Number of pixels to shift
    x1, y1 = max(cx - mean_patch_half - shift_x, 0), max(cy - mean_patch_half - shift_y, 0)
    x2, y2 = min(cx + mean_patch_half - shift_x, w), min(cy + mean_patch_half - shift_y, h)
    """
    # Compute centroid mean NDVI values
    centroid_ndvi_true = [
        ndvi_true[i, cy - patch_size : cy + patch_size, cx - patch_size : cx + patch_size].mean().item()
        for i in range(num_samples)]
    centroid_ndvi_pred = [
        ndvi_pred[i, cy - patch_size : cy + patch_size, cx - patch_size : cx + patch_size].mean().item()
        for i in range(num_samples)]
    """
    # Compute centroid mean NDVI values
    centroid_ndvi_true = [ndvi_true[i, y1:y2, x1:x2].median().item() for i in range(num_samples)]
    centroid_ndvi_pred = [ndvi_pred[i, y1:y2, x1:x2].median().item() for i in range(num_samples)]
    
    # Define figure and GridSpec layout
    num_images = min(6, num_samples)
    selected_indices = np.linspace(0, num_samples - 1, num_images, dtype=int)
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(4, num_images, height_ratios=[1.8, 1, 1, 1])  # 4 rows: Graph, RGB, NDVI True, NDVI Pred

    # --- Line plot for NDVI values ---
    ax_graph = fig.add_subplot(gs[0, :])  # Use the full width for the graph
    ax_graph.plot(timestamps, centroid_ndvi_true, marker="o", label="NDVI (true)", linestyle="-", color="blue")
    ax_graph.plot(timestamps, centroid_ndvi_pred, marker="s", label="NDVI (pred.)", linestyle="--", color="red")
    ax_graph.set_ylabel("NDVI",fontweight="bold",fontsize=12)
    ax_graph.set_ylim(-1, 1)
    ax_graph.set_xticks(range(len(timestamps)),)
    ax_graph.set_xticklabels([f"{t[:4]}-{t[4:6]}-{t[6:]}" for t in timestamps], rotation=25)
    ax_graph.legend()
    ax_graph.set_title("NDVI vs. Predicted NDVI over Time")
    # **Move x-ticks to the top**
    ax_graph.tick_params(axis="x", pad=-36,direction="in")  # Moves x-ticks up slightly


    # Define datasets, colormaps, and row labels
    # stretchings for plotting
    ndvi_true = (np.array(ndvi_true)+1)/2
    ndvi_pred = (np.array(ndvi_pred)+1)/2
    ndvi_true = np.clip(ndvi_true, 0, 1)
    rgbs = rgbs*5
    rgbs = np.clip(rgbs, 0, 1)
    centroid_ndvi_pred = np.clip(centroid_ndvi_pred, 0, 1)
    datasets = [rgbs, ndvi_true, ndvi_pred]
    cmaps = [None, "viridis", "viridis"]
    titles = ["RGB", "NDVI (True)", "NDVI (Pred.)"]

    # Loop through image types and create plots
    for row, (dataset, cmap, title) in enumerate(zip(datasets, cmaps, titles)):
        for col, idx in enumerate(selected_indices):
            ax = fig.add_subplot(gs[row + 1, col])  # Place in the correct row

            img = dataset[idx]

            if row == 0:  # RGB
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
            else:  # NDVI images
                img = (img - img.min()) / (img.max() - img.min())  # Normalize for visualization

            ax.imshow(img, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if row == 0 or row == 1:
                ax.set_xlabel("")  # No labels for RGB row
            else:
                ax.set_xlabel(
                    timestamps[idx][:4] + "-" + timestamps[idx][4:6] + "-" + timestamps[idx][6:], 
                    fontsize=10
                )

            """
            # Define red box (centroid mean patch)
            mean_patch_half = mean_patch_size // 2
            box_x1 = (plot_patch_size // 2) - mean_patch_half
            box_y1 = (plot_patch_size // 2) - mean_patch_half

            # Create red rectangle
            rect = patches.Rectangle((box_x1, box_y1), mean_patch_size, mean_patch_size, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            """
            box_x1 = (plot_patch_size // 2) - mean_patch_half - shift_x
            box_y1 = (plot_patch_size // 2) - mean_patch_half - shift_y

            # Create red rectangle
            rect = patches.Rectangle((box_x1, box_y1), mean_patch_size, mean_patch_size, 
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)


            # Add row title on the leftmost column
            if col == 0:
                ax.set_ylabel(title, fontsize=12, rotation=90, labelpad=15,fontweight="bold")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)  # Decrease space between image rows

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    pil_image = Image.open(buf).copy()
    plt.close()
    buf.close()
    return pil_image

def calculate_and_plot_timeline(model=None,device=None,root_dir="validation_utils/time_series_bavaria/*.tif",size_input=256,mean_patch_size=4):
    r,n,p,t  = get_pred_nirs_and_info(model,device,root_dir,size_input=size_input)
    im = plot_ndvi_timeline(r,n,p,t,mean_patch_size=mean_patch_size)
    del r,n,p,t
    gc.collect()
    return im


if __name__ == "__main__":
    pass
    #r,n,p,t  = get_pred_nirs_and_info(None)
    #im = plot_ndvi_timeline(r,n,p,t,mean_patch_size=4)
    #im.save("validation_utils/timeline_ndvi_plot.png")
    
if __name__ == "__main__":
    # Get Model
    import torch
    from omegaconf import OmegaConf
    from model.pix2pix import Px2Px_PL
    yaml_config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
    model = Px2Px_PL(yaml_config)
    ckpt_path = "logs/exp_NIR_GAN_SatCLIP/2025-03-07_14-52-31_SatCLIP_best/epoch=196-step=621600.ckpt"
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.eval()
    
    # Prepare Data
    tx_crop_path = "validation_utils/time_series_texas_cropcircles/*.tif"
    pil_im = calculate_and_plot_timeline(model,device=None,root_dir=tx_crop_path,size_input=256)
    # save image
    pil_im.save("validation_utils/timeline_plot_cherry.png")
    
    