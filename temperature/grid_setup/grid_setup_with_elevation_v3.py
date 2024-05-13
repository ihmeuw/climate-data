import numpy as np
import pandas as pd
import math
from multiprocessing import Pool
import rtree
import rasterio as reo
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from rasterio.features import shapes
import shapely
from shapely.geometry import Point, Polygon, shape, box
from shapely import wkt
from osgeo import gdal, osr
import sklearn
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from joblib import load
from scipy import stats
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray
import re
import os
import xarray as xr
import calendar
import datetime as dt
from datetime import datetime
import argparse
import dask.dataframe as dd
import dask.array as da
import pygeohash as pgh
import getpass
import subprocess

import pyproj

pyproj.datadir.set_data_dir(
    "/ihme/homes/nhashmeh/miniconda3/envs/earth-analytics-python/share/proj"
)
os.environ["PROJ_LIB"] = (
    "/ihme/homes/nhashmeh/miniconda3/envs/earth-analytics-python/share/proj"
)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="filename")

args = parser.parse_args()

chunks_path = (
    "/mnt/share/erf/ERA5/merged_era5_with_stations/grid_setup/chunks_by_latlon_v3/"
)

chunk_file = args.file

grid_df = pd.read_feather(chunks_path + chunk_file)

grid_df.drop(
    columns=["latitude_right", "longitude_right"], inplace=True
)  # this should have been dropped earlier...

grid_df["geometry"] = grid_df["geometry"].apply(wkt.loads)
grid_gdf = gpd.GeoDataFrame(grid_df, geometry="geometry")


# Function to adjust longitude
def adjust_longitude(geometry):
    if geometry.centroid.x < 0:
        return shapely.affinity.translate(geometry, xoff=360)
    else:
        return geometry


gdf_merge = grid_gdf
gdf_merge["elevation"] = np.nan

merge_counter = 0
success_counter = 0

for file in os.listdir("/mnt/share/erf/SRTM_GL1_srtm/"):
    file_ext = file.split(".")[-1]

    if file_ext != "tif":
        continue

    NS = file[0:1]
    NS_num = float(file[1:3])
    EW = file[3:4]
    EW_num = float(file[4:7])

    if file_ext != "tif":
        continue

    if NS == "S":
        NS_num = float(NS_num) * -1

    if EW == "W":
        EW_num = 360 - float(EW_num)

    if (
        (EW_num < math.floor(gdf_merge["longitude_left"].min()))
        | (EW_num > math.ceil(gdf_merge["longitude_left"].max()))
    ) | (
        (NS_num < math.floor(gdf_merge["latitude_left"].min()))
        | (NS_num > math.ceil(gdf_merge["latitude_left"].max()))
    ):
        merge_counter += 1
        continue

    # rescaling 30 meters to 1 km
    downscale_factor = 30 / 1000

    # Load the raster
    with reo.open("/mnt/share/erf/SRTM_GL1_srtm/" + file) as src:
        # resample data to target shape
        data = src.read(
            out_shape=(
                src.count,
                int(
                    src.height * downscale_factor
                ),  # multiply height by downscale factor
                int(src.width * downscale_factor),  # multiply widrh by downscale factor
            ),
            resampling=Resampling.bilinear,  # resampling method
        )

        # scale image transform
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]), (src.height / data.shape[-2])
        )

        # convert the resampled raster to a geodataframe
        results = (
            {"properties": {"raster_val": v}, "geometry": s}
            for i, (s, v) in enumerate(
                reo.features.shapes(data[0], transform=transform)
            )
        )  # assumes a single band image

        geodf = gpd.GeoDataFrame.from_features(list(results))

        # Apply the function to the geometry column
        geodf["geometry"] = geodf["geometry"].apply(adjust_longitude)

        prev_len = len(gdf_merge[~gdf_merge["elevation"].isna()])  # used for debugging

        gdf_merge = gpd.sjoin(gdf_merge, geodf, how="left", predicate="intersects")
        # test_merge.rename(columns = {'raster_val' : 'elevation'}, inplace=True)
        gdf_merge["elevation"] = gdf_merge["elevation"].combine_first(
            gdf_merge["raster_val"]
        )
        gdf_merge.drop(columns="index_right", inplace=True)
        gdf_merge.drop(columns="raster_val", inplace=True)

        if len(gdf_merge[~gdf_merge["elevation"].isna()]) > prev_len:
            success_counter += 1

gdf_merge.to_feather(
    "/mnt/share/erf/ERA5/merged_era5_with_stations/grid_setup/chunks_with_elevation_v3/"
    + chunk_file,
    index=False,
)
