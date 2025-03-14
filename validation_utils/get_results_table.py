import geopandas as gpd
import pandas as pd


from tqdm import tqdm
from data.worldstrat import worldstrat_ds
from data.s100k_dataset import S2_100k
from data.worldstrat import worldstrat_ds
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import os
from datetime import datetime
import torch
import pandas as pd
from utils.logging_helpers import plot_tensors,plot_index
from PIL import Image
import kornia
from validation_utils.val_utils import crop_center
from utils.remote_sensing_indices import RemoteSensingIndices


def create_val_metrics(config_path,folder):

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # get config
    config = OmegaConf.load(config_path)
    satclip = config.satclip.use_satclip

    ds = worldstrat_ds(config)
    
    #ds = S2_100k(config,phase="val")
    ds = worldstrat_ds(config)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    # Load Model
    from model.pix2pix import Px2Px_PL
    model = Px2Px_PL(config)
    ckpt = torch.load(config.custom_configs.Model.weights_path,map_location=device)
    model.load_state_dict(ckpt['state_dict'],strict=False)
    model = model.eval()
    print("Loaded (only) Weights from:",config.custom_configs.Model.weights_path)

    # ITERATE OVER DATASET
    # set empty metrics dict
    metrics_dict = {"id":[],
                    "x":[],
                    "y":[],
                    "ssim":[],
                    "psnr":[],
                    "l1":[],
                    "l2":[],
                    "l1_ndvi":[],
                    "l1_ndwi":[],
                    "l1_evi":[],
                    }
    # --- iterate and get metrics ----
    for v,batch in tqdm(enumerate(dl),total=len(dl)):

        rgb,nir,coords = batch["rgb"],batch["nir"],batch["coords"],
        pred = model.predict_step(rgb,coords)

        # crop center
        rgb = crop_center(rgb.squeeze(0),240).unsqueeze(0)
        nir = crop_center(nir.squeeze(0),240).unsqueeze(0)
        pred = crop_center(pred.squeeze(0),240).unsqueeze(0)

        # get info
        x = coords[0][0].item()
        y = coords[0][1].item()
        id_ = v

        # get standard metrics:
        ssim = kornia.metrics.ssim(nir, pred, window_size=11).mean()
        psnr = kornia.metrics.psnr(nir, pred, max_val=1.0)
        l1 = torch.mean(torch.abs(nir - pred))
        l2 = torch.mean((nir - pred) ** 2)
            
        # get RS indices metrics
        val_obj = RemoteSensingIndices(mode="loss",criterion="l1")
        result_dict = RemoteSensingIndices.get_and_weight_losses(val_obj,rgb,nir,pred,loss_config=None,mode="logging_dict")

        # append to list
        metrics_dict["id"].append(id_)
        metrics_dict["x"].append(x)
        metrics_dict["y"].append(y)
        metrics_dict["ssim"].append(ssim.item())
        metrics_dict["psnr"].append(psnr.item())
        metrics_dict["l1"].append(l1.item())
        metrics_dict["l2"].append(l2.item())
        metrics_dict["l1_ndvi"].append(result_dict["indices_loss/ndvi_error"].item())
        metrics_dict["l1_ndwi"].append(result_dict["indices_loss/ndwi_error"].item())
        metrics_dict["l1_evi"].append(result_dict["indices_loss/evi_error"].item())

        # plot standard image
        if id_%50==0:
            # Create Image folder and name
            image_folder = os.path.join(folder,"images")
            os.makedirs(image_folder,exist_ok=True)
            type_name = "SatCLIP" if satclip else "NoSatCLIP"
            id_clean = str(id_).zfill(4)
            im_path = os.path.join(image_folder,f"example_image_{id_clean}_{type_name}.png")
            # get and save image
            img = plot_tensors(rgb, nir, pred,title="Worldstrat Validation")
            img.save(im_path, 'PNG')
            
            if False:
                # get and plot Indices Image
                val_obj.mode = "index"
                # get indices
                ndvi,ndvi_pred = val_obj.ndvi_calculation(rgb,nir,pred)
                ndwi,ndwi_pred = val_obj.ndwi_calculation(rgb,nir,pred)
                evi,evi_pred = val_obj.evi_calculation(rgb,nir,pred)
                # create plots
                im_ndvi = plot_index(rgb,ndvi,ndvi_pred,title="NDVI",index_name="NDVI")
                im_ndwi = plot_index(rgb,ndwi,ndwi_pred,title="NDWI",index_name="NDWI")
                im_evi = plot_index(rgb,evi,evi_pred,title="EVI",index_name="EVI")
                # save plots
                # TODO: Find out why all the indices look the same
                im_ndvi.save(f'images/indices/{id_}_ndvi.png', 'PNG')
                im_ndwi.save(f'images/indices/{id_}_ndwi.png', 'PNG')  
                im_evi.save(f'images/indices/{id_}_evi.png', 'PNG')
        

        # save metrics
        df = pd.DataFrame(metrics_dict)
        if id_%25==0:
            df.to_csv(os.path.join(folder,"validation_metrics.csv"))
            
        # break logic
        if v==-1:
            break


    # Get Context info for dataset
    from validation_utils.geo_ablation import append_info_to_df,clean_economy
    gdf = append_info_to_df(df)
    gdf = clean_economy(gdf)
    gdf = gdf.loc[:, ~gdf.columns.duplicated()] # remove double geo column
        
    # get current data and time string

    #create folder
    # save final version with context info
    gdf.to_file(os.path.join(folder,"validation_metrics_ablation_satclip_"+str(satclip)+".geojson"),driver='GeoJSON')
    

def filter_for_countries(gdf):
    world_countries = gpd.read_file("validation_utils/datasets/countries_noAntGreen.geojson")
    gdf = gpd.sjoin(gdf,world_countries,how="inner")
    return gdf

# Create Out Folder for this experiment
now = datetime.now()
folder = now.strftime("%d_%m_%Y_%H_%M_%S")
folder = os.path.join("validation_utils/metrics_folder/",folder)
os.makedirs(folder,exist_ok=True)

# get Results
create_val_metrics("configs/val_confs/val_config_px2px.yaml",folder=folder)
create_val_metrics("configs/val_confs/val_config_px2px_SatCLIP.yaml",folder=folder)


# PLOT RESULTS
import geopandas as gpd
sc_path = os.path.join(folder,"validation_metrics_ablation_satclip_True.geojson")
nosc_path = os.path.join(folder,"validation_metrics_ablation_satclip_False.geojson")

gdf_noSatCLIP = gpd.read_file(nosc_path)
gdf_SatCLIP = gpd.read_file(sc_path)
gdf_noSatCLIP_f = filter_for_countries(gdf_noSatCLIP)
gdf_SatCLIP_f = filter_for_countries(gdf_SatCLIP)

# start plotting
from validation_utils.plot_val_spiders import plot_radar_comparison
plot_radar_comparison(gdf_SatCLIP_f,gdf_noSatCLIP_f,"Continent",folder=folder)
plot_radar_comparison(gdf_SatCLIP_f,gdf_noSatCLIP_f,"Koppen_Class",folder=folder)
plot_radar_comparison(gdf_SatCLIP_f,gdf_noSatCLIP_f,"economy",folder=folder)




