
# Do some plotting

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_radar(stats_df, data_type):

    # prepare dataset
    stats_df = stats_df.groupby(data_type).agg({
    'psnr': 'mean',
    'ssim': 'mean'
        }).reset_index()

    # Number of variables we're plotting.
    categories = stats_df[data_type].tolist()
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variables)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # ensure the polygon is closed by connecting it back to the starting point

    # Initialise the radar plot
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))

    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], categories, color='grey', size=15, verticalalignment='top')
    #plt.title('Average PSNR and SSIM by ' + data_type)

    # Set y-axis labels for PSNR
    ax.set_rlabel_position(0)
    plt.yticks([10, 20, 30], ["10\n0.3", "20\n0.6", "30\n1."], color="grey", size=13)
    plt.ylim(0, 30)

    # Set y-axis labels for SSIM (0 to 1)
    #ax.set_rlabel_position(1) 
    #plt.yticks([-10,-20,-30], ["0.3","0.6","1,0"], color="grey", size=7)
    #plt.ylim(0, 1)

    # PSNR Data
    values = stats_df['psnr'].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label='PSNR')

    # SSIM Data (consider scaling SSIM by a factor to fit the same scale as PSNR if needed)
    values2 = stats_df['ssim'].tolist()
    values2 = [v * 30 for v in values2]  # scale SSIM to match PSNR for visual purposes
    values2 += values2[:1]
    ax.plot(angles, values2, linewidth=2, linestyle='solid', label='SSIM')

    # Add a legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1),fontsize=13)

    # Save the plot and close
    out_file_name = f"validation_utils/plots/metrics_radar_{data_type}.png"
    out_file_name = out_file_name.replace(" ", "_")
    plt.tight_layout()
    plt.savefig(out_file_name)
    plt.close()


if False:
    gdf = gpd.read_file("validation_utils/metrics_folder/worldstrat_metrics.geojson")
    plot_radar(gdf,"Continent")
    plot_radar(gdf,"economy")
    plot_radar(gdf,"Koppen_Class")



def plot_radar_comparison(df1, df2, data_type):
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
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), subplot_kw=dict(polar=True))

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
gdf_noSatCLIP = gpd.read_file("validation_utils/metrics_folder/validation_metrics_noSatCLIP.geojson")
#gdf_SatCLIP = gpd.read_file("validation_utils/worldstrat_metrics_SatCLIP.geojson")

# start plotting
plot_radar_comparison(gdf_noSatCLIP,gdf_noSatCLIP,"Continent")
plot_radar_comparison(gdf_noSatCLIP,gdf_noSatCLIP,"Koppen_Class")
plot_radar_comparison(gdf_noSatCLIP,gdf_noSatCLIP,"economy")


