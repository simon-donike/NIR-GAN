import ee
import cubexpress
import json
import random
from tqdm import tqdm

import pathlib
import matplotlib.pyplot as plt
import rasterio
import numpy as np


# Authenticate for script-based execution
ee.Authenticate()

# Initialize Earth Engine
ee.Initialize()
samples_file = "/data1/simon/GitHub/NIR_GAN/playground/world_50k_L8.geojson"
with open(samples_file) as f:
    data = json.load(f)

#  Randomly select 5 features
random_features = random.sample(data['features'],5000) #len(data['features']))#, 10)
print("Randomly sampled features:", len(random_features))

# Extract coordinates
points = [
   (
      feature["geometry"]["coordinates"][0],
      feature["geometry"]["coordinates"][1]
    ) for feature in random_features
]

# Start with a broad L8 collection
collection = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
    .filterDate("2020-05-15", "2024-09-15")
    .filter(ee.Filter.lt("CLOUD_COVER", 5))
)

# Build a list of Request objects
requestset = []

for i, (lon, lat) in tqdm(enumerate(points),total=len(points),desc="Sampling Points..."):

    # Create a point geometry for the current coordinates
    point_geom = ee.Geometry.Point([lon, lat])
    collection_filtered = collection.filterBounds(point_geom)

    # Convert the filtered collection into a list of asset IDs
    #image_ids = collection_filtered.aggregate_array("system:id").getInfo()

     # Convert the filtered collection into a list of asset IDs
    try:
      image_ids = collection_filtered.first().get("system:id").getInfo()
    except:
      continue
    if not image_ids:
        continue

    # Define a geotransform for this point
    geotransform = cubexpress.lonlat2rt(
        lon=lon,
        lat=lat,
        edge_size=512,
        scale=30
    )

    if type(image_ids)==str:
      image_ids = [image_ids]

    # Create one Request per image found for this point
    requestset.extend([
        cubexpress.Request(
            id=f"l8_{i}_{idx}",
            raster_transform=geotransform,
            bands=["B4", "B3", "B2", "B5"], # You can add more bands here
            image=image_id
        )
        for idx, image_id in enumerate(image_ids)
    ])

# save query result to file
import pickle

# Save the list to a file
with open("requestset_l8.pkl", "wb") as f:
    pickle.dump(requestset, f)

print("Starting Download...")


# Combine into a RequestSet
cube_requests = cubexpress.RequestSet(requestset=requestset)
#print(cube_requests._dataframe)

# Download everything in parallel
results = cubexpress.getcube(
    request=cube_requests,
    nworkers=4,
    output_path="/data2/simon/nirgan_l8",
    max_deep_level=5
)
print("Downloaded files:", results)
