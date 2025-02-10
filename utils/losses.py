# SSIM Loss
import kornia
import torch

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
    return loss

if __name__ == "__main__":
    a,b = torch.rand(1,1,512,512), torch.rand(1,1,512,512)
    l = ssim_loss(a,b)
    print("SSIM Loss:", l)
