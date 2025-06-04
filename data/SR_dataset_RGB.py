import os
import torch
import numpy as np
import rasterio
from rasterio.warp import transform
from torch.utils.data import Dataset, DataLoader

class SR_dataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Path to dataset root with LR/ and HR/ subfolders.
        """
        self.lr_dir = os.path.join(root_dir, "LR")
        self.hr_dir = os.path.join(root_dir, "HR")
        self.filenames = sorted([
            f for f in os.listdir(self.lr_dir)
            if f.endswith(".tif") and os.path.isfile(os.path.join(self.hr_dir, f))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr_path = os.path.join(self.lr_dir, fname)
        hr_path = os.path.join(self.hr_dir, fname)

        with rasterio.open(lr_path) as src_lr:
            lr = src_lr.read().astype(np.float32)/10000
            # Save centroid in native CRS
            centroid_x, centroid_y = src_lr.xy(src_lr.height // 2, src_lr.width // 2)

            # Transform to WGS84 (lon, lat)
            lon, lat = transform(
                src_lr.crs, "EPSG:4326", [centroid_x], [centroid_y]
            )
            centroid_coords = torch.tensor([lon[0], lat[0]], dtype=torch.float32)
                

        with rasterio.open(hr_path) as src_hr:
            hr = src_hr.read().astype(np.float32)/10000

        lr_tensor = torch.from_numpy(lr)
        hr_tensor = torch.from_numpy(hr)
        
        lr_rgb = lr_tensor[:3, :, :]
        lr_nir = lr_tensor[3, :, :]
        hr_rgb = hr_tensor[:3, :, :]
        
        return {"lr":lr_rgb,
                "hr":hr_rgb,
                "s2_nir":lr_nir.unsqueeze(0),  # Assuming NIR is the 4th band in LR
                "coords":centroid_coords,
                "id": fname.split('.')[0]
            }


if __name__ == "__main__":
    # Example usage
    dataset = SR_dataset(root_dir = "data/synthDataset")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = dataset.__getitem__(0)