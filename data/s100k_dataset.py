import os
import random
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pyproj import Transformer
from tqdm import tqdm
import pandas as pd
from rasterio.mask import mask


class S2_100k(Dataset):
    def __init__(self, config, phase="train",overwrite_metadata=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            patch_size (int): Size of the patch to be extracted (default: 512).
            transform (callable, optional): Optional transform to be applied on a sample.
        """        
        self.config = config
        self.root_dir = self.config.Data.S2_100k_settings.base_path
        self.patch_size = self.config.Data.S2_100k_settings.image_size
        self.phase=phase
        
        # do assertions before starting to build dataset
        assert phase in ["train","val","test"], "Phase must be one of 'train','val','test'"
        assert self.patch_size == 256, "Only 512x512 patches are supported for Sentinel-2 100k dataset."

        # chech if dataframe exists
        if  os.path.exists(os.path.join(self.root_dir,"metadata.pkl")):
            if overwrite_metadata: # overwrite metadata is selected
                self.metadata = self.create_metadata() # get metadata
                self.metadata.to_pickle(os.path.join(self.root_dir,"metadata.pkl")) # save file
            else:
                print("Metadata file found, reading...")
                self.metadata = pd.read_pickle(os.path.join(self.root_dir,"metadata.pkl")) #read file if it exists
        
        # if file does not exist, create it
        else:
            print("Metadata file not found, creating...")
            self.metadata = self.create_metadata() # get metadata
            self.metadata.to_pickle(os.path.join(self.root_dir,"metadata.pkl")) # save file

        # Filter Dataset
        self.metadata = self.metadata[self.metadata.valid==True] # filter for valid images
        self.metadata = self.metadata[self.metadata.null_percentage<=5] # filter for null value percentage
        # Set train/test/split
        self.metadata = self.metadata[self.metadata.split==self.phase]
        
        # keep only Europe if CLC mask is enabled
        if self.config.Data.S2_100k_settings.return_clc_mask:
            print("Filtering to only include Europe...")
            self.metadata = self.metadata[self.metadata["continent"]=="Europe"]
            
        # set padding settings
        if self.config.Data.padding:
            self.pad = torch.nn.ReflectionPad2d(self.config.Data.padding_amount)

        print(f"Instanciated S2_100k dataset with {len(self.metadata)} datapoints for phase: {self.phase}")


    def create_metadata(self):
        image_paths = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                # assert file is tiff
                if filename.endswith(".tif"):
                    image_paths.append(os.path.join(dirpath, filename))
        print(f"Found {len(image_paths)} images in {self.root_dir}")
        assert len(image_paths) > 0, "No images found in directory."
        # open every image, if it fails remove from list
        image_paths_new = []
        image_validity = []
        image_coords_x = []
        image_coords_y = []
        null_percentage = []
        for idx, image_path in tqdm(enumerate(image_paths),desc="Checking images...",total=len(image_paths)):
            #if idx==100:
            #    break
            try:
                with rasterio.open(image_path) as src:
                    # Get image dimensions
                    rgb_nir_bands = [2, 3, 4, 8]
                    patch = src.read(rgb_nir_bands)
                    patch = np.nan_to_num(patch, nan=0.0)
                    patch = patch.astype(np.int64)
                    patch = torch.Tensor(patch)
                    patch = patch / 10000.0
                    patch = torch.clamp(patch, 0, 1)
                    # get centroid lon/lat
                    crs = src.crs
                    trans = src.transform
                    centroid_x = self.patch_size / 2
                    centroid_y = self.patch_size / 2
                    geo_x, geo_y = trans * (centroid_x, centroid_y)
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    lon, lat = transformer.transform(geo_x, geo_y)
                    coords = [lon,lat]
                    # get 0 percentage
                    num_zeros = torch.sum(patch == 0.00).item()
                    total_elements = patch.numel()
                    percentage_zeros = (num_zeros / total_elements) * 100

                # append data to list
                null_percentage.append(percentage_zeros)                
                image_paths_new.append(image_path)
                image_validity.append(True)
                image_coords_x.append(coords[0])
                image_coords_y.append(coords[1])
            except KeyboardInterrupt: # catch keyboard interrupt
                raise
            except: # catch all other exceptions and treat file as corrupt
                null_percentage.append(percentage_zeros)                
                image_paths_new.append(image_path)
                image_validity.append(False)
                image_coords_x.append(0.)
                image_coords_y.append(0.)
        metadata = pd.DataFrame({"image_path":image_paths_new,"valid":image_validity,"coords_x":image_coords_x,"coords_y":image_coords_y,"null_percentage":null_percentage})

        # create train/test split
        metadata['split'] = 'train'

        # Define percentages
        val_percentage = 0.01  # 10% for validation
        test_percentage = 0.05  # 50% for testing

        # Calculate sample sizes based on percentages
        num_val = int(len(metadata) * val_percentage)
        num_test = int(len(metadata) * test_percentage)

        val_indices = metadata.sample(n=num_val, random_state=1).index
        metadata.loc[val_indices, 'split'] = 'val'

        # Remove already selected validation indices and sample 5000 rows for testing
        test_indices = metadata.drop(val_indices).sample(n=num_test, random_state=1).index
        metadata.loc[test_indices, 'split'] = 'test'
        
        # Add continent column from spatial join with geojson
        import geopandas as gpd
        metadata = gpd.GeoDataFrame(metadata, geometry=gpd.points_from_xy(metadata.coords_x, metadata.coords_y), crs="EPSG:4326")
        continents = gpd.read_file(self.config.Data.S2_100k_settings.continent_geojson, driver="GeoJSON")
        print("Performing Spatial Join with continents...")
        metadata = gpd.sjoin(metadata, continents, how="left")
        # remove index_right column
        metadata = metadata.drop(columns=["index_right"])
        # rename continent column
        metadata = metadata.rename(columns={"CONTINENT": "continent"})

        return(metadata)
                
    def __len__(self):
        return len(self.metadata)
    
    def get_clc_mask(self, clc_path):
        # get CLC mask for image
        with rasterio.open(clc_path) as src:
            clc_mask = src.read(1)
            clc_mask = torch.from_numpy(clc_mask).float()
            
        # unify mapping to classes
        df = pd.read_csv(self.config.Data.S2_100k_settings.clc_mapping_file)
        clc_mask = clc_mask.numpy()
        # for GRID_CODE in csv, get GROUP_ID and assign new value in numpy array
        mapping = dict(zip(df["GRID_CODE"], df["GROUP_ID"]))
        # Remap using numpy vectorized function
        clc_remapped = np.vectorize(mapping.get)(clc_mask)
        clc_remapped = np.nan_to_num(clc_remapped, nan=0).astype(np.uint8)  # Optional: fill unmapped with 0
        clc_mask = torch.from_numpy(clc_remapped)
        
        return clc_mask
    
    def __getitem__(self, idx):

        image_path = self.metadata.iloc[idx]["image_path"]

        # try reading image, if it fails, add to error list and return random image
        try:
            with rasterio.open(image_path) as src:
                # Get image dimensions
                rgb_nir_bands = [2, 3, 4, 8]
                patch = src.read(rgb_nir_bands)

                if self.config.Data.S2_100k_settings.return_coords:
                    # get centroid lon/lat
                    crs = src.crs
                    trans = src.transform
                    centroid_x = self.patch_size / 2
                    centroid_y = self.patch_size / 2
                    geo_x, geo_y = trans * (centroid_x, centroid_y)
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    lon, lat = transformer.transform(geo_x, geo_y)
                    coords = torch.Tensor([lon,lat])
        except Exception as e:
            print("Error reading file:",self.image_paths[idx], "Error:",e)
            self.error_idx.append(idx)
            rand_idx = random.randint(0,len(self.image_paths)-1)
            return self.__getitem__(rand_idx)
        

        # After successful reading, do preprocessing and return
        # turn to 0..1
        patch = patch / 10000.0
        patch = torch.from_numpy(patch).float()
        patch = torch.clamp(patch, 0, 1)
        rgb = patch[:3,:,:]
        nir = patch[3:,:,:]
        if self.config.Data.padding:
            rgb = self.pad(rgb)
            nir = self.pad(nir)
            
        # extract RGB and NIR bands
        batch = {"rgb": rgb, "nir": nir}
        if self.config.Data.S2_100k_settings.return_coords:
            batch["coords"] = coords
        if self.config.Data.S2_100k_settings.return_clc_mask:
            clc_mask = get_clc_mask(self.metadata.iloc[idx]["clc_path"])
            batch["clc_mask"] = clc_mask
        
        return(batch)
                
                
                
class S2_100k_datamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Initialize the dataset
        self.dataset_train = S2_100k(self.config,phase="train")
        self.dataset_val = S2_100k(self.config,phase="val")
        

    def train_dataloader(self):        
        return DataLoader(self.dataset_train,batch_size=self.config.Data.train_batch_size,
                          shuffle=True, num_workers=self.config.Data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,batch_size=self.config.Data.val_batch_size,
                          shuffle=True, num_workers=self.config.Data.num_workers,drop_last=True)



if __name__=="__main__":
    config = OmegaConf.load("configs/config_baselines.yaml")
    ds = S2_100k(config,overwrite_metadata=True)
    #dm = S2_100k_datamodule(config)
    #batch = next(iter(dm.train_dataloader()))
    #coords = batch["coords"]


    #for batch in tqdm(dm.train_dataloader()):
    #    pass



# This is only used once to write the CLC masks to disk. After that, it shouldnt be used again.
def get_clc_mask(metadata):
    
    import rasterio
    from rasterio.mask import mask
    from shapely.geometry import box, mapping
    import geopandas as gpd

    def extract_clc_patch(clc_path, image_patch_path, output_path):
        with rasterio.open(image_patch_path) as src_img:
            bounds = src_img.bounds
            img_crs = src_img.crs
            geom = box(*bounds)
            gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs=img_crs)

        with rasterio.open(clc_path) as src_clc:
            clc_crs = src_clc.crs
            if clc_crs != gdf.crs:
                gdf = gdf.to_crs(clc_crs)

            try:
                out_image, out_transform = mask(
                    dataset=src_clc,
                    shapes=gdf.geometry.map(mapping),
                    crop=True
                )
            except ValueError as e:
                if "do not overlap" in str(e):
                    print(f"Skipped {image_patch_path} — no overlap with CLC.")
                    return None
                else:
                    raise

            out_meta = src_clc.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
        
        return output_path
            
    #metadata = pd.read_pickle(os.path.join(config.Data.S2_100k_settings.base_path,"metadata.pkl"))
    clc_path = "/data2/simon/s100k/clc_4326.tif"
    metadata["clc_path"] = None  # Initialize the column

    for idx, row in tqdm(metadata.iterrows(),total=len(metadata)):
        image_path = row["image_path"]
        image_continent = row["continent"]
        if image_continent != "Europe":
            continue
        
        # construct out_path
        out_base = "/data2/simon/s100k/clc_patches/"
        out_path = os.path.join(out_base,os.path.basename(image_path))

        extract_clc_patch(clc_path, image_path, out_path)
        
        metadata.at[idx, "clc_path"] = out_path
        
    metadata.to_pickle(os.path.join(config.Data.S2_100k_settings.base_path,"metadata.pkl"))
    return metadata





md = ds.metadata.copy()
# Add continent column from spatial join with geojson
import geopandas as gpd
md = gpd.GeoDataFrame(md, geometry=gpd.points_from_xy(md.coords_x, md.coords_y), crs="EPSG:4326")
continents = gpd.read_file(ds.config.Data.S2_100k_settings.continent_geojson, driver="GeoJSON")
print("Performing Spatial Join with continents...")
md = gpd.sjoin(md, continents, how="left")
# remove index_right column
md = md.drop(columns=["index_right"])
# rename continent column
md = md.rename(columns={"CONTINENT": "continent"})


md = get_clc_mask(md)

