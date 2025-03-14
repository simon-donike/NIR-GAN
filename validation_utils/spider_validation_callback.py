import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.remote_sensing_indices import RemoteSensingIndices
from validation_utils.val_utils import crop_center
import kornia
import os
import pandas as pd
import geopandas as gpd
from utils.logging_helpers import plot_tensors,plot_index


def spider_validation_callback(model,ds,satclip,folder="validation_utils/automated_spiders/",epoch_no=0):
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
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    for v,batch in tqdm(enumerate(dl),total=len(dl),desc="Creating Validation Geodataframe---"):

        rgb,nir,coords = batch["rgb"],batch["nir"],batch["coords"],
        pred = model.predict_step(rgb.to(model.device),coords.to(model.device))

        # crop center
        rgb = crop_center(rgb.squeeze(0),240).unsqueeze(0).to("cpu")
        nir = crop_center(nir.squeeze(0),240).unsqueeze(0).to("cpu")
        pred = crop_center(pred.squeeze(0),240).unsqueeze(0).to("cpu")

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
    gdf.to_file(os.path.join(folder,"validation_metrics_ablation_satclip_"+str(satclip)+"_e"+str(epoch_no)+".geojson"),driver='GeoJSON')
    
    