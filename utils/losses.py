# SSIM Loss
import kornia
import torch
from scipy.stats import wasserstein_distance
import numpy as np


def ssim_loss(img1, img2, window_size=11):
    """
    Compute the SSIM loss between two images using Kornia.
    
    Parameters:
    - img1, img2: Tensor of shape [batch, channels, height, width]
    - window_size: Size of the Gaussian window, default is 11
    - reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    
    Returns:
    - SSIM loss between img1 and img2
    """
    # Kornia expects all tensors to be float, so ensure your images are floats
    img1 = img1.float()
    img2 = img2.float()

    # Compute SSIM
    loss = kornia.metrics.ssim(img1, img2, window_size).mean()
    loss = 1 - loss  # SSIM returns similarity, so subtract from 1 to get loss
    return loss

def hist_loss(image1, image2, bins=256):
    """
    Calculate the Wasserstein distance between two single-channel images.

    Parameters:
    image1 (numpy.ndarray): The first image (single-channel).
    image2 (numpy.ndarray): The second image (single-channel).
    bins (int): Number of histogram bins.

    Returns:
    float: The Wasserstein distance between the histograms of the two images.
    """
    if type(image1) == torch.Tensor:
        image1 = image1.cpu().detach().numpy()
    if type(image2) == torch.Tensor:
        image2 = image2.cpu().detach().numpy()
        
    # Flatten the images to 1D arrays
    flat_image1 = image1.ravel()
    flat_image2 = image2.ravel()

    # Compute histograms
    hist1, bin_edges = np.histogram(flat_image1, bins=bins, range=[0, 256], density=True)
    hist2, _ = np.histogram(flat_image2, bins=bins, range=[0, 256], density=True)

    # Compute the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate the Wasserstein distance
    distance = wasserstein_distance(bin_centers, bin_centers, u_weights=hist1, v_weights=hist2)

    return distance



if __name__ == "__main__":
    a,b = torch.rand(1,1,512,512), torch.rand(1,1,512,512)+0.1
    l = ssim_loss(a,b)
    print("SSIM Loss:", l)
    
    l2 = hist_loss(a,b)
    print("Hist Loss:", l2)


