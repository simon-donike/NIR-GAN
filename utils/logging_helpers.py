import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
from data.normalise_s2 import minmax_percentile
import numpy as np


def plot_tensors(rgb, nir, pred_nir,title="Train"):

    rgb = rgb.clamp(0,1)
    nir = nir.clamp(0,1)
    pred_nir = pred_nir.clamp(0,1)
    rgb = minmax_percentile(rgb,perc=2)

    num_images_to_plot = min(pred_nir.shape[0], 5)
    fig, axes = plt.subplots(num_images_to_plot, 3, figsize=(15, 5 * num_images_to_plot))

    # If only one image to plot, convert axes to a 2D array format
    if num_images_to_plot == 1:
        axes = np.expand_dims(axes, 0)  # Make it 2D

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
    pil_image = Image.open(buf).copy()
    plt.close()
    buf.close()
    return(pil_image)



def plot_tensors_hist(rgb, nir, pred_nir, title="Train"):
    # stretch images
    if True:
        nir = nir*1.5
        pred_nir = pred_nir*1.5
    
    rgb = rgb.clamp(0, 1)
    nir = nir.clamp(0, 1)
    pred_nir = pred_nir.clamp(0, 1)
    rgb = minmax_percentile(rgb,perc=2)
    
    # Crop Middle
    if rgb.shape[-1]<350: # assume input images are 256x256
        crop_size = 240
    else:
        crop_size = 500

    B, C, H, W = rgb.shape
    crop_height, crop_width = crop_size, crop_size
    start_x = (W - crop_width) // 2
    start_y = (H - crop_height) // 2
    rgb = rgb[:,:, start_y:start_y+crop_height, start_x:start_x+crop_width]
    nir = nir[:,:, start_y:start_y+crop_height, start_x:start_x+crop_width]
    pred_nir = pred_nir[:,:, start_y:start_y+crop_height, start_x:start_x+crop_width]

    num_images_to_plot = min(pred_nir.shape[0], 5)
    fig, axes = plt.subplots(num_images_to_plot, 4, figsize=(20, 5 * num_images_to_plot))  # Changed to 4 columns

    if num_images_to_plot == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_images_to_plot):
        rgb_image = rgb[i].permute(1, 2, 0).cpu().numpy()
        nir_image = nir[i][0].cpu().numpy()
        pred_nir_image = pred_nir[i][0].cpu().numpy()

        axes[i, 0].imshow(rgb_image)
        axes[i, 1].imshow(nir_image, cmap='viridis')
        axes[i, 2].imshow(pred_nir_image, cmap='viridis')

        # Histogram for real NIR
        bins_num = 100
        total_pixels = nir_image.size  # Total number of pixels in the image
        vals_nir = np.histogram(nir_image.ravel(), bins=bins_num, range=(0, 1))[0]
        vals_pred = np.histogram(pred_nir_image.ravel(), bins=bins_num, range=(0, 1))[0]
        vals_nir_percentage = (vals_nir / total_pixels) 
        vals_pred_percentage = (vals_pred / total_pixels) 
        bins = np.linspace(0, 1, bins_num + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers
        axes[i, 3].plot(bin_centers, vals_nir_percentage, color='blue')
        axes[i, 3].plot(bin_centers, vals_pred_percentage, color='red')
        #axes[i, 3].set_xlim([0, 1])
        axes[i, 3].legend(['Real NIR', 'Predicted NIR'])
        axes[i, 3].set_xlabel('Pixel Intensity')
        axes[i, 3].set_ylabel('Value Frequency')
        if i == 0:
            axes[i, 0].set_title('RGB Image')
            axes[i, 1].set_title('NIR Image')
            axes[i, 2].set_title('Predicted NIR Image')
            axes[i, 3].set_title('NIR/ predNIR Histogram')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    pil_image = Image.open(buf).copy()
    plt.close()
    buf.close()
    return pil_image


def plot_index(rgb, nir, pred_nir, title="Train",index_name="NDVI"):
    #rgb = rgb.clamp(0, 1)
    # bring to 0..1 and clamp
    #nir = (nir+1)/2
    #nir = nir.clamp(0, 1)
    # bring to 0..1 and clamp
    #pred_nir = (pred_nir+1)/2
    #pred_nir = pred_nir.clamp(0, 1)

    num_images_to_plot = min(pred_nir.shape[0], 5)
    fig, axes = plt.subplots(num_images_to_plot, 3, figsize=(15, 5 * num_images_to_plot))
    if num_images_to_plot==1: # expand if only 1 batch
        axes = np.expand_dims(axes, 0)
    
    #fig.suptitle("R channel from RGB, structural information invalid.", fontsize=12, y=1.02)
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
        axes[i, 1].set_title(f'{index_name} (Actual)')

        # Plot NDVI of the predicted NIR
        axes[i, 2].imshow(ndvi_pred, cmap='RdYlGn')
        axes[i, 2].set_title(f'{index_name} (Predicted)')

    plt.tight_layout()

    # Save and return image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=60)
    buf.seek(0)
    pil_image = Image.open(buf).copy()
    plt.close()
    buf.close()
    return pil_image
