import torch
import pandas as pd
import os
import rasterio
from rasterstats import point_query


# GET COUNTRIES
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import ast
from tqdm import tqdm




def get_countries(df,world="/data1/simon/GitHub/worldstrat/pretrained_model/countries.shp"):
    
    df = df.set_crs('EPSG:4326')

    world = gpd.read_file(world)
    world['Country'] = world['SOV_A3']
    world["Continent"] = world['CONTINENT']
    world["Economy"] = world['ECONOMY']
    columns_to_keep = ["Country","Continent","ECONOMY","geometry"]
    world = world[columns_to_keep]
    world = world.set_crs('EPSG:4326')
    result = gpd.sjoin(df, world, how="left")
    result = result.set_crs('EPSG:4326')
    return result

def get_climate_zones(df,koppen_path="/data1/simon/GitHub/worldstrat/pretrained_model/koppen.tif",
                      legend_csv="/data1/simon/GitHub/worldstrat/pretrained_model/koppen_zones.csv"):
    points = df.geometry  # Ensure this is in the same CRS as the raster
    # Iterate over points with tqdm for progress tracking
    koppen_classes = []
    for point in tqdm(df.geometry, desc="Processing points"):
        # Perform point query for each point individually
        result = point_query([point], koppen_path)
        koppen_classes.append(result[0] if result else None)  # Append the result or None if no data
    # prepare list, read legend list
    indices_list = [int(f) if f is not None else 0 for f in koppen_classes]
    legend = pd.read_csv(legend_csv)
    # add row for 0s
    unknown_row = pd.DataFrame({'id': [0], 'Code': ['Unknown'], 'Description': ['No valid data'], 'Color': ['[0, 0, 0]']})
    legend = pd.concat([unknown_row, legend], ignore_index=True)
    # Extract the climate codes based on the indices    
    climate_codes = [legend.loc[legend['id'] == idx, 'Code'].values[0] if idx <= len(legend) else 'Unknown' for idx in indices_list]
    climate_codes2 = [word[0].upper() for word in climate_codes if word] # get first letter, capitalize
    # Add the extracted values to your GeoDataFrame
    df['Koppen_Class'] = climate_codes2
    return df


def final_touch(df,cols_to_keep=[]):
    df = df.set_crs('EPSG:4326')
    # remove comumns we dont need
    columns_to_keep = cols_to_keep + ["id","x","y","ssim","geometry","Country","Continent","ECONOMY","Koppen_Class"]
    df = df[columns_to_keep]
    return(df)


def append_info_to_df(df):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    gdf = gdf.set_crs('EPSG:4326')
    gdf_original_cols = list(gdf.columns) # keep original columns
    gdf = get_countries(gdf)
    gdf = get_climate_zones(gdf)
    gdf = final_touch(gdf,cols_to_keep=gdf_original_cols)
    return(gdf)

def clean_economy(df):
    # economy class extraction
    ls = df["ECONOMY"]
    ls_ = []
    k = {1:"Developed: G7",
        2:"Developed: Non G7",
        3:"Emerging: BRIC",
        4:"Emerging: MIKT",
        5:"Emerging: G20",
        6:"Developing",
        7:"Least Developed"}

    for i in ls:
        if type(i)==str:
            num = int(i[0])
        else:
            num = 999

        if num in k.keys():
            ls_.append(k[num])
        else:
            ls_.append("Unknown")
            
    # remove column ECONOMY
    df["economy"] = ls_
    df = df.drop(columns=["ECONOMY"])
    return(df)


if __name__ == "__name__":
    # get results
    df = pd.read_csv("validation_utils/worldstrat_metrics.csv")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))

    # get countries, continents and economies
    gdf = get_countries(gdf)
    gdf = get_climate_zones(gdf)
    df = final_touch(df)

