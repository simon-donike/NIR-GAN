import geopandas as gpd
import pandas as pd


gdf_noSatCLIP = gpd.read_file("validation_utils/metrics_folder/validation_metrics_noSatCLIP.geojson")
df = pd.DataFrame(gdf_noSatCLIP)
