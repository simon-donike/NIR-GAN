import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
import pathlib
from tqdm import tqdm

class SISRWorldWide(Dataset):
    def __init__(self, path, split="train"):
        self.path = pathlib.Path(path)
        self.files = list(self.path.glob("*.tortilla"))
        self.metadata = tacoreader.load(self.files)
        
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
        self.metadata = df
        
        print(f"Instanciated TortillaDataset for phase {split} with {len(self.metadata)} iamge pairs.")

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


if __name__ == "__main__":
    # Usage
    tortilla_path = "/data3/SEN2NAIP_global"
    dataset = SISRWorldWide(tortilla_path,"train")

    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dl))