import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import torchvision.transforms as transforms
from utils.sen2_stretch import sen2_stretch
from utils.normalise_s2 import normalise_s2
from utils.normalise_s2 import minmax_percentile
import skimage
import numpy as np


def plot_tensors(rgb, nir, pred_nir,title="Train"):

    rgb = rgb.clamp(0,1)
    nir = nir.clamp(0,1)
    pred_nir = pred_nir.clamp(0,1)
    rgb = minmax_percentile(rgb,perc=2)

    num_images_to_plot = min(pred_nir.shape[0], 5)
    fig, axes = plt.subplots(num_images_to_plot, 3, figsize=(15, 5 * num_images_to_plot))

    # plot images
    for i in range(num_images_to_plot):
        # Extract the i-th RGB and NIR images
        rgb_image = rgb[i].permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
        nir_image = nir[i][0]  # Change from (1, H, W) to (H, W)
        pred_nir_image = pred_nir[i][0]  # Change from (1, H, W) to (H, W)

        rgb_image = rgb_image.cpu().numpy()
        nir_image = nir_image.cpu().numpy()
        pred_nir_image = pred_nir_image.cpu().numpy()

        # histogram match images
        #if config != None:
        #    if config.Data.spectral_matching == "histogram":
        #        pred_nir_image =  skimage.exposure.match_histograms(pred_nir_image, nir_image)

        # Plot the RGB image
        axes[i, 0].imshow(rgb_image)
        #axes[i, 0].axis('off')

        # Plot the NIR image
        axes[i, 1].imshow(nir_image, cmap='RdYlGn') # inverted to Rg->Low Gn->High
        #axes[i, 1].axis('off')

        # Plot the NIR image
        axes[i, 2].imshow(pred_nir_image, cmap='RdYlGn')
        #axes[i, 2].axis('off')

        if i == 0:
            axes[i, 0].set_title('RGB Image')
            axes[i, 1].set_title('NIR Image')
            axes[i, 2].set_title('Predicted NIR Image')
    plt.tight_layout()

    # Create a PIL image from the BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png',dpi=100)
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close()
    return(pil_image)


def plot_ndvi(rgb, nir, pred_nir, title="Train"):
    rgb = rgb.clamp(0, 1)
    nir = nir.clamp(0, 1)
    pred_nir = pred_nir.clamp(0, 1)

    num_images_to_plot = min(pred_nir.shape[0], 5)
    fig, axes = plt.subplots(num_images_to_plot, 3, figsize=(15, 5 * num_images_to_plot))
    
    fig.suptitle("R channel from RGB, structural information invalid.", fontsize=12, y=1.02)
    for i in range(num_images_to_plot):
        # Prepare RGB and NIR images
        rgb_image = rgb[i].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
        nir_image = nir[i][0].cpu().numpy()  # Convert to (H, W)
        pred_nir_image = pred_nir[i][0].cpu().numpy()  # Convert to (H, W)

        # Calculate NDVI for actual and predicted NIR
        ndvi_actual = (nir_image - rgb_image[..., 0]) / (nir_image + rgb_image[..., 0] + 1e-6)
        ndvi_pred = (pred_nir_image - rgb_image[..., 0]) / (pred_nir_image + rgb_image[..., 0] + 1e-6)

        # Clip NDVI for display
        ndvi_actual = np.clip(ndvi_actual, -1, 1)
        ndvi_pred = np.clip(ndvi_pred, -1, 1)
        
        # stretch to 0..1 for plotting
        ndvi_actual = (ndvi_actual+1)/2
        ndvi_pred = (ndvi_pred+1)/2

        # Plot RGB image
        axes[i, 0].imshow(minmax_percentile(torch.Tensor(rgb_image), perc=2).numpy())
        axes[i, 0].set_title('RGB Image')

        # Plot NDVI of the actual NIR
        axes[i, 1].imshow(ndvi_actual, cmap='RdYlGn')
        axes[i, 1].set_title('NDVI (Actual)')

        # Plot NDVI of the predicted NIR
        axes[i, 2].imshow(ndvi_pred, cmap='RdYlGn')
        axes[i, 2].set_title('NDVI (Predicted)')

    plt.tight_layout()

    # Save and return image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close()
    return pil_image
