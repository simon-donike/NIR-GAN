import matplotlib.pyplot as plt

def save_ds_image(pl_datamodule,model):
    b = next(iter(pl_datamodule.train_dataloader()))
    model = model.eval()
    if "coords" in b.keys():
        fake_nir = model.predict_step(b["rgb"],b["coords"])
    else:
        fake_nir = model.predict_step(b["rgb"])
    model = model.train()
    fake_nir = fake_nir.detach().cpu().numpy()[0]  # Assuming this is a batch with shape [batch, channels, height, width]

    # Prepare images
    real_nir_image = b["nir"].detach().cpu().numpy()[0, 0, :, :]
    rgb_image = b["rgb"].detach().cpu().numpy()[0, :3, :, :].transpose(1, 2, 0)
    rgb_image = rgb_image * 3  # Adjusting brightness if necessary
    fake_nir_image = fake_nir[0, :, :]  # Assuming fake_nir is [channels, height, width]
    rgb_image,fake_nir_image,real_nir_image = rgb_image.clip(0, 1),fake_nir_image.clip(0, 1),real_nir_image.clip(0, 1)

    # Create a subplot of 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust the size as needed

    # Display images
    axes[0].imshow(rgb_image)
    axes[0].axis('off')  # Hide axes for clean look
    axes[0].set_title('RGB Image')

    axes[1].imshow(fake_nir_image, cmap='viridis')  # Assuming fake_nir is a single channel image
    axes[1].axis('off')
    axes[1].set_title('Fake NIR Image')

    axes[2].imshow(real_nir_image, cmap='viridis')
    axes[2].axis('off')
    axes[2].set_title('Real NIR Image')

    # Save the entire figure
    plt.savefig("images/dataset_model_sanity_check.png")
    plt.close(fig)  # Close the figure to free up memory