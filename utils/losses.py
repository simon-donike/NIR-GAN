# SSIM Loss
import kornia
import torch
from scipy.stats import wasserstein_distance
import numpy as np
import torch.nn.functional as F



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

def hist_loss_old(image1, image2, bins=256):
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

def emd_loss(pred, target):
    # assert neither NaN or Inf
    assert not torch.isnan(pred).any()
    assert not torch.isnan(target).any()
    assert not torch.isinf(pred).any()
    assert not torch.isinf(target).any()
    
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    pred_cdf = torch.cumsum(F.softmax(pred, dim=1), dim=1)
    target_cdf = torch.cumsum(F.softmax(target, dim=1), dim=1)

    emd = torch.mean(torch.abs(pred_cdf - target_cdf))
    return emd

def rrmse_old(pred, ref, reduction=True, eps=1e-12):
    # ---- NumPy ----
    if isinstance(pred, np.ndarray) and isinstance(ref, np.ndarray):
        assert pred.shape == ref.shape, "Shapes must match"
        x = pred.astype(np.float64)
        y = ref.astype(np.float64)

        if x.ndim == 3:   # [C,H,W] or [H,W,C]
            B = 1
            p = x.reshape(1, -1)
            r = y.reshape(1, -1)
        elif x.ndim == 4: # [N,C,H,W] or [N,H,W,C]
            B = x.shape[0]
            p = x.reshape(B, -1)
            r = y.reshape(B, -1)
        else:
            raise ValueError("Expected 3D or 4D input for NumPy arrays")

        mse = np.mean((r - p) ** 2, axis=1)
        ref_energy = np.mean(r ** 2, axis=1)

        denom = np.sqrt(ref_energy)
        out = np.sqrt(mse) / np.maximum(denom, eps)  # avoid /0
        out = np.where(ref_energy > 0, out, np.inf)  # explicit inf if zero-energy

        if reduction is False:
            return out[0] if B == 1 else out
        else:
            return out[0] if B == 1 else float(out.mean())

    # ---- PyTorch ----
    if isinstance(pred, torch.Tensor) and isinstance(ref, torch.Tensor):
        assert pred.shape == ref.shape, "Shapes must match"
        x, y = pred.float(), ref.float()

        if x.ndim == 3:
            B = 1
            p = x.reshape(1, -1)
            r = y.reshape(1, -1)
        elif x.ndim == 4:
            B = x.shape[0]
            p = x.reshape(B, -1)
            r = y.reshape(B, -1)
        else:
            raise ValueError("Expected 3D or 4D input for torch tensors")

        mse = torch.mean((r - p) ** 2, dim=1)
        ref_energy = torch.mean(r ** 2, dim=1)

        denom = torch.sqrt(ref_energy + eps)  # epsilon
        out = torch.sqrt(mse) / denom
        out = torch.where(ref_energy > 0, out, torch.full_like(out, float('inf')))

        if reduction is False:
            return out.squeeze(0) if B == 1 else out
        else:
            return out.mean()

    raise TypeError("Inputs must both be np.ndarray or both torch.Tensor")


def rrmse(pred, ref, reduction=True, eps=1e-12, mask=None):
    assert isinstance(pred, torch.Tensor) and isinstance(ref, torch.Tensor)
    assert pred.shape == ref.shape, "Shapes must match"
    x, y = pred.float(), ref.float()

    # Flatten by batch
    if x.ndim == 3:
        B = 1; P = x.reshape(1, -1); R = y.reshape(1, -1)
    elif x.ndim == 4:
        B = x.shape[0]; P = x.reshape(B, -1); R = y.reshape(B, -1)
    else:
        raise ValueError("Expected 3D or 4D tensors")

    if mask is None:
        M = (R != 0)
    else:
        M = mask.reshape(B, -1).bool()

    # per-sample masked means
    valid = M.sum(dim=1)
    mse = torch.zeros(B, dtype=x.dtype, device=x.device)
    ref_energy = torch.zeros_like(mse)

    has_valid = valid > 0
    if has_valid.any():
        diff2 = (R[has_valid] - P[has_valid])**2
        ref2  = (R[has_valid])**2
        mse[has_valid] = (diff2 * M[has_valid]).sum(dim=1) / valid[has_valid].clamp_min(1)
        ref_energy[has_valid] = (ref2 * M[has_valid]).sum(dim=1) / valid[has_valid].clamp_min(1)

    # cases:
    # 1) ref_energy>0 → normal rrmse
    # 2) ref_energy==0 & mse==0 → define 0
    # 3) ref_energy==0 & mse>0 → inf
    out = torch.empty_like(mse)
    normal = ref_energy > 0
    zero_ref = ~normal
    out[normal] = torch.sqrt(mse[normal]) / torch.sqrt(ref_energy[normal] + eps)
    out[zero_ref] = torch.where(mse[zero_ref] == 0, torch.zeros_like(mse[zero_ref]), torch.full_like(mse[zero_ref], float('inf')))

    if reduction is False:
        return out if B > 1 else out.squeeze(0)
    else:
        return out.mean()


if __name__ == "__main__":
    a,b = torch.rand(3,1,512,512), torch.rand(3,1,512,512)+0.1
    ssim = ssim_loss(a,b)
    print("SSIM Loss:", ssim)
    
    emd = hist_loss_old(a,b)
    print("Hist Loss:", emd)
    
    emd_loss = emd_loss(a,b)
    print("EMD Loss:", emd_loss)
    
    rrmse_value = rrmse(a, b)
    print("RRMSE:", rrmse_value)
    



