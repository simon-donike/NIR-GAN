
import torch
import torch.nn.functional as F
import kornia
from utils.losses import rrmse


def calculate_metrics(pred, target, phase="train"):
    """
    Calculate PSNR, SSIM, L1 loss, and L2 loss between nir and prednir.

    Args:
        nir (torch.Tensor): NIR image tensor [B, C, H, W].
        prednir (torch.Tensor): Predicted NIR image tensor [B, C, H, W].

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
    
    # Calculate RRSME
    rrmse_val = rrmse(pred, target)

    metrics = {
        phase+'/L1': l1_loss,
        phase+'/L2': l2_loss,
        phase+'/PSNR': psnr,
        phase+'/SSIM': ssim,
        phase+'/RRSME': rrmse_val
    }

    return metrics

