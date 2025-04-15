
# Do some plotting

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import re


def plot_radar_comparison(sc, no_sc, data_type,out_name="",folder="validation_utils/metrics_folder/"):
    df1 = sc
    df2 = no_sc
    # Aggregate data by the specified type
    stats_df1 = df1.groupby(data_type).agg({
        'psnr': 'mean',
        'ssim': 'mean'
    }).reset_index()

    stats_df2 = df2.groupby(data_type).agg({
        'psnr': 'mean',
        'ssim': 'mean'
    }).reset_index()

    # Prepare data
    categories = stats_df1[data_type].tolist()  # Assumes both dataframes have the same categories
    N = len(categories)

    # Angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Set up plot
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), subplot_kw=dict(polar=True))

    # Function to plot each radar chart
    def plot_radar(ax, values1, values2, title):
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='grey', size=13)
        ax.set_title(title, size=15, position=(0.5, 1.1))

        # First dataset
        values1 += values1[:1]
        ax.plot(angles, values1, linewidth=2, linestyle='solid', label='SatCLIP')

        # Second dataset
        values2 += values2[:1]
        ax.plot(angles, values2, linewidth=2, linestyle='dashed', label='No SatCLIP')

        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # PSNR
    plot_radar(ax1, stats_df1['psnr'].tolist(), stats_df2['psnr'].tolist(), 'PSNR')

    # SSIM
    plot_radar(ax2, stats_df1['ssim'].tolist(), stats_df2['ssim'].tolist(), 'SSIM')

    if out_name != "":
        out_name = "_"+out_name
    # Save the plot and close
    out_file_name = f"metrics_radar_satclip{out_name}_{data_type}.png"
    out_file_name = out_file_name.replace(" ", "_")
    out_file_name = os.path.join(folder, out_file_name)
    plt.tight_layout()
    plt.savefig(out_file_name)
    plt.close()
    
    

# read all geojson files in the folder
folder = 'logs/exp_NIR_GAN_SatCLIP/2025-03-14_17-29-03_metrics'
geojsons_paths = []

for file in os.listdir(folder):
    if file.endswith('.geojson') or file.endswith('.json'):
        path = os.path.join(folder, file)
        geojsons_paths.append(path)
geojsons_paths = list(geojsons_paths)
# order list
geojsons_paths.sort()


# iterate over all epoch files
out_folder = "validation_utils/paper_graphs/"
for f in geojsons_paths:
    e = int(re.search(r'_e(\d+)\.geojson$', f).group(1))
    e_str = f"E{e:03d}"
    print(f"Doing {e_str}...")
    
    # read files
    gdf_SatCLIP = gpd.read_file(f)
    gdf_noSatCLIP = gpd.read_file("validation_utils/metrics_folder/13_03_2025_14_47_01/validation_metrics_ablation_satclip_False.geojson")    
    
    # start plotting
    out_folder = "validation_utils/paper_graphs/"
    plot_radar_comparison(gdf_SatCLIP,gdf_noSatCLIP,data_type="Continent",out_name=e_str,folder=out_folder)
    plot_radar_comparison(gdf_SatCLIP,gdf_noSatCLIP,data_type="Koppen_Class",out_name=e_str,folder=out_folder)
    plot_radar_comparison(gdf_SatCLIP,gdf_noSatCLIP,data_type="economy",out_name=e_str,folder=out_folder)
    
    
    
    
    
    
    
# SINGLE EXAMPLE    
# Do plotting for the validation metrics and PAPER
gdf_SatCLIP = gpd.read_file("logs/exp_NIR_GAN_SatCLIP/2025-03-14_17-29-03_metrics/validation_metrics_ablation_satclip_True_e86.geojson")
gdf_noSatCLIP = gpd.read_file("validation_utils/metrics_folder/13_03_2025_14_47_01/validation_metrics_ablation_satclip_False.geojson")

# plot stuff
out_folder = "validation_utils/paper_graphs/"
plot_radar_comparison(gdf_SatCLIP,gdf_noSatCLIP,data_type="Continent",out_name="",folder=out_folder)
plot_radar_comparison(gdf_SatCLIP,gdf_noSatCLIP,data_type="Koppen_Class",out_name="",folder=out_folder)
plot_radar_comparison(gdf_SatCLIP,gdf_noSatCLIP,data_type="economy",out_name="",folder=out_folder)

    


if __name__ == "__main__":
    
    # read files
    gdf_noSatCLIP = gpd.read_file("validation_utils/metrics_folder/validation_metrics_ablation_satclip_False.geojson")
    gdf_SatCLIP = gpd.read_file("validation_utils/metrics_folder/validation_metrics_ablation_satclip_True.geojson")

    # start plotting
    plot_radar_comparison(gdf_SatCLIP,gdf_noSatCLIP,data_type="Continent",out_name="")
    plot_radar_comparison(gdf_SatCLIP,gdf_noSatCLIP,data_type="Koppen_Class",out_name="")
    plot_radar_comparison(gdf_SatCLIP,gdf_noSatCLIP,data_type="economy",out_name="")
    
    



