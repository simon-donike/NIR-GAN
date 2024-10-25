import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import torchvision.transforms as transforms
from utils.sen2_stretch import sen2_stretch
from utils.normalise_s2 import normalise_s2
from utils.normalise_s2 import minmax_percentile
import skimage


def plot_tensors(rgb, nir, pred_nir,title="Train",stretch=None,config=None):
    # prepare tensors
    if stretch=="SEN2":
        nir = normalise_s2(nir,stage="denorm")
        pred_nir = normalise_s2(pred_nir,stage="denorm")
        pred_nir = sen2_stretch(pred_nir)
    elif stretch==None:
        pass
    else:
        raise NotImplementedError("Stretch not implemented:",stretch)
    
    zoom = 256
    rgb = rgb * (10.0 / 3.0) # visualization stretch
    rgb = rgb[:, :, :zoom, :zoom]
    nir = nir[:, :, :zoom, :zoom]
    pred_nir = pred_nir[:, :, :zoom, :zoom]

    rgb = rgb.clamp(0,1)
    nir = nir.clamp(0,1)
    #pred_nir = pred_nir.clamp(0,1)

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
        if config != None:
            if config.Data.spectral_matching == "histogram":
                pred_nir_image =  skimage.exposure.match_histograms(pred_nir_image, nir_image)

        # Plot the RGB image
        axes[i, 0].imshow(rgb_image)
        #axes[i, 0].axis('off')

        # Plot the NIR image
        axes[i, 1].imshow(nir_image, cmap='YlGn')
        #axes[i, 1].axis('off')

        # Plot the NIR image
        axes[i, 2].imshow(pred_nir_image, cmap='YlGn')
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

