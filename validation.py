from tqdm import tqdm
from data.worldstrat import worldstrat_ds
from data.s100k_dataset import S2_100k
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import os
import torch
import pandas as pd
from utils.logging_helpers import plot_tensors,plot_index
from PIL import Image
import kornia
from validation_utils.val_utils import crop_center
from utils.remote_sensing_indices import RemoteSensingIndices


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Get Data
satclip=False
if satclip:
    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
else:
    config = OmegaConf.load("configs/config_px2px.yaml")

#ds = worldstrat_ds(config)
ds = S2_100k(config,phase="val")
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

# Load Model
from model.pix2pix import Px2Px_PL
model = Px2Px_PL(config)
ckpt = torch.load(config.custom_configs.Model.weights_path,map_location=device)
model.load_state_dict(ckpt['state_dict'])
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
    if id_%10==0:
        img = plot_tensors(rgb, nir, pred,title="Worldstrat Validation")
        img.save(f'validation_utils/images/example_image_{id_}.png', 'PNG')
        
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
        df.to_csv("validation_utils/validation_metrics.csv")
        
    # break logic
    if v==-1:
        break


# Get Context info for dataset
from validation_utils.geo_ablation import append_info_to_df,clean_economy
gdf = append_info_to_df(df)
gdf = clean_economy(gdf)
gdf = gdf.loc[:, ~gdf.columns.duplicated()] # remove double geo column
    
# save final version with context info
gdf.to_file("validation_utils/metrics_folder/validation_metrics_ablation_satclip_"+str(satclip)+".geojson",driver='GeoJSON')







