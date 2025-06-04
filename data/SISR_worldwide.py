import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
import pathlib
from tqdm import tqdm
import os
import rasterio

class SISRWorldWide(Dataset):
    def __init__(self, path, split="train"):
        self.path = pathlib.Path(path)
        self.files = list(self.path.glob("*.tortilla"))
        self.metadata = tacoreader.load(self.files)
        
        if split!=None:
            # train/test split
            df = self.metadata.reset_index(drop=True)
            if split == "test":
                df = df.iloc[-100:].reset_index(drop=True)
            elif split == "val":
                df = df.iloc[-600:-100].reset_index(drop=True)
            elif split == "train":
                df = df.iloc[:-600].reset_index(drop=True)
            elif split == "all":
                df = df.reset_index(drop=True)
            else:
                raise ValueError(f"Unknown split: {split}")
        else:
            df = self.metadata.reset_index(drop=True)
        self.metadata = df
        
        print(f"Instanciated TortillaDataset for phase {split} with {len(self.metadata)} image pairs.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        s2_path = self.metadata.read(idx).read(0)
        lr_path = self.metadata.read(idx).read(2)
        hr_path = self.metadata.read(idx).read(3)

        with rio.open(lr_path) as src1, rio.open(hr_path) as src2, rio.open(s2_path) as src3:
            lr = src1.read().astype(np.float32) / 10000
            hr = src2.read().astype(np.float32) / 10000
            s2 = src3.read().astype(np.float32) / 10000

        lr = torch.from_numpy(lr)  # shape: [C, H, W]
        hr = torch.from_numpy(hr)
        s2 = torch.from_numpy(s2[7:8,:,:]) # select NIR band
        return {"lr":lr,"hr": hr,"s2_nir": s2,"tortilla:id": self.metadata["tortilla:id"][idx]}

    def save_geotiff(self, idx, out_dir="data/synthDataset"):
        """
        Save LR and HR images at given index to GeoTIFFs in <out_dir>/LR and <out_dir>/HR.

        Args:
            idx (int): Index of the sample to save.
            out_dir (str or Path): Root output directory.
        """
        out_dir = pathlib.Path(out_dir)
        lr_dir = out_dir / "LR"
        hr_dir = out_dir / "HR"
        lr_dir.mkdir(parents=True, exist_ok=True)
        hr_dir.mkdir(parents=True, exist_ok=True)

        s2_path = self.metadata.read(idx).read(0)
        lr_path = self.metadata.read(idx).read(2)
        hr_path = self.metadata.read(idx).read(3)
        tortilla_id = self.metadata["tortilla:id"][idx]

        # Read LR and HR
        with rio.open(lr_path) as src1:
            lr = src1.read().astype(np.float32)
            lr_meta = src1.meta.copy()
            lr_transform = src1.transform
            lr_crs = src1.crs

        with rio.open(hr_path) as src2:
            hr = src2.read().astype(np.float32)
            hr_meta = src2.meta.copy()

        # Read S2 NIR band (8th band = index 7)
        with rio.open(s2_path) as src3:
            s2_nir = src3.read(8).astype(np.float32)[np.newaxis, :, :]  # shape: (1, H, W)

        # Append S2 NIR to LR
        lr_aug = np.concatenate([lr, s2_nir], axis=0)  # shape: (C+1, H, W)
        
            # Update LR metadata
        lr_meta.update({
            "count": lr_aug.shape[0],
            "dtype": "float32",
            "transform": lr_transform,
            "crs": lr_crs
        })


        # Output paths
        lr_out = lr_dir / f"{tortilla_id}.tif"
        hr_out = hr_dir / f"{tortilla_id}.tif"

        with rio.open(lr_out, "w", **lr_meta) as dst:
            dst.write(lr_aug.astype(np.float32))
        with rio.open(hr_out, "w", **hr_meta) as dst:
            dst.write(hr.astype(np.float32))

        print(f"Saved:\n  {lr_out}\n  {hr_out}")


if __name__ == "__main__":
    # Usage
    tortilla_path = "/data3/SEN2NAIP_global"
    dataset = SISRWorldWide(tortilla_path,None)

    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dl))
    
    """
    for i in range(250):
        dataset.save_geotiff(i)
    """