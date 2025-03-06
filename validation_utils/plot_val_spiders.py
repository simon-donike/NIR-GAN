
# Do some plotting

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_radar_comparison(sc, no_sc, data_type):
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

    # Save the plot and close
    out_file_name = f"validation_utils/plots/metrics_radar_satclip_{data_type}.png"
    out_file_name = out_file_name.replace(" ", "_")
    plt.tight_layout()
    plt.savefig(out_file_name)
    plt.close()



# read files
gdf_noSatCLIP = gpd.read_file("validation_utils/metrics_folder/validation_metrics_ablation_satclip_False.geojson")
gdf_SatCLIP = gpd.read_file("validation_utils/metrics_folder/validation_metrics_ablation_satclip_True.geojson")

# start plotting
plot_radar_comparison(gdf_SatCLIP,gdf_noSatCLIP,"Continent")
plot_radar_comparison(gdf_SatCLIP,gdf_noSatCLIP,"Koppen_Class")
plot_radar_comparison(gdf_SatCLIP,gdf_noSatCLIP,"economy")


