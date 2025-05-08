import torch
import matplotlib.pyplot as plt


def plot_rgb_nir_and_mask(rgb_tensor, nir, pred_nir, mask_tensor, it=0, title=None):
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    rgb_tensor = torch.Tensor(rgb_tensor)
    nir = torch.Tensor(nir)
    mask_tensor = torch.Tensor(mask_tensor)
    pred_nir = torch.Tensor(pred_nir)
    # Convert tensors to NumPy arrays
    rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    nir_np = nir.squeeze().cpu().numpy()       # [H, W]
    pred_np = pred_nir.squeeze().cpu().numpy() # [H, W]
    mask_np = mask_tensor.cpu().numpy()

    # RGB scale adjustment (optional)
    rgb_np = (rgb_np * 5).clip(0, 1)

    # Define colormap for mask
    cmap = ListedColormap([
        "#ffffff",  # 0: white
        "#90ee90",  # 1: light green
        "#006400",  # 2: dark green
        "#1e90ff",  # 3: blue
        "#ff0000"   # 4: red
    ])

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(rgb_np)
    plt.title("RGB")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(nir_np, cmap="viridis", vmin=0, vmax=1)
    plt.title("Ground Truth NIR")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred_np, cmap="viridis", vmin=0, vmax=1)
    plt.title("Predicted NIR")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(mask_np, cmap=cmap, vmin=0, vmax=4)
    plt.title("CLC Mask")
    plt.axis("off")

    if title:
        plt.suptitle(title)

    #plt.tight_layout()
    plt.savefig(f"/data1/simon/GitHub/NIR_GAN/images/clc_validation/clc_mask_{it}.png", dpi=300)
    plt.close()
