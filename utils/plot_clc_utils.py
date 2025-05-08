

# plot rgb and clc
import matplotlib.pyplot as plt

def plot_rgb_and_mask(rgb_tensor, mask_tensor,it=0, title=None):
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    """
    rgb_tensor: [3, H, W] torch tensor, assumed in [0, 1]
    mask_tensor: [H, W] torch tensor, integer class mask
    """
    rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    mask_np = mask_tensor.cpu().numpy()
    
    cmap = ListedColormap([
        "#ffffff",  # 0: white (background / no class)
        "#90ee90",  # 1: light green (Agricultural)
        "#006400",  # 2: dark green (Natural Vegetation)
        "#1e90ff",  # 3: blue (Water)
        "#ff0000"   # 4: red (Artificial surfaces)
    ])

    plt.figure(figsize=(10, 5))

    # prepare arrays for plotting
    rgb_np = rgb_np * 5
    rgb_np = rgb_np.clip(0, 1)    

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_np)
    plt.title("RGB Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap=cmap, vmin=0, vmax=4)
    plt.title("CLC Mask")
    plt.axis("off")

    plt.savefig(f"/data1/simon/GitHub/NIR_GAN/images/clc_masks/clc_mask_{it}.png", dpi=300)
    plt.close()