
import torch
import torch.nn.functional as F
import kornia

def calculate_metrics(pred, target, phase="train"):
    """
    Calculate PSNR, SSIM, L1 loss, and L2 loss between sr and hr.

    Args:
        sr (torch.Tensor): Super-resolved image tensor [B, C, H, W].
        hr (torch.Tensor): High-resolution image tensor [B, C, H, W].

    Returns:
        dict: Dictionary containing the calculated metrics.
    """

    # Calculate L1 loss
    l1_loss = F.l1_loss(pred, target).item()

    # Calculate L2 loss (MSE)
    l2_loss = F.mse_loss(pred, target).item()

    # Calculate PSNR
    psnr = kornia.metrics.psnr(pred, target, 1.0).item()
    
    # Calculate SSIM
    ssim = kornia.metrics.ssim(pred,target,window_size=5,max_val=1.).mean().item()

    metrics = {
        phase+'/L1': l1_loss,
        phase+'/L2': l2_loss,
        phase+'/PSNR': psnr,
        phase+'/SSIM': ssim
    }

    return metrics

