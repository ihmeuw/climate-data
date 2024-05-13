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
parser.add_argument("era5_file", type=str, help="filename")

args = parser.parse_args()

chunks_path = (
    "/mnt/share/erf/ERA5/merged_era5_with_stations/grid_setup/chunks_with_elevation_v3/"
)
era5_path = "/mnt/share/erf/ERA5/three_hourly_temp_9km/daily_temps/"
output_path = "/mnt/share/erf/ERA5/merged_era5_with_stations/grid_setup/chunks_with_elevation_and_era5_v3/"

chunk_file = args.file
era5_file = args.era5_file

chunk_file_name = chunk_file.split(".")[0]

print(chunk_file_name)

grid_df = pd.read_feather(chunks_path + chunk_file)

# grid_df['geometry'] = grid_df['geometry'].apply(wkt.loads)
# grid_gdf = gpd.GeoDataFrame(grid_df, geometry='geometry')

geometry = [
    Point(xy) for xy in zip(grid_df["longitude_left"], grid_df["latitude_left"])
]
grid_gdf = gpd.GeoDataFrame(grid_df, geometry=geometry)

grid_gdf = grid_gdf[grid_gdf["lcz_value"] != 0.0]
values_to_remove = [-999.0, -999.9]  # replace with your actual values
grid_gdf = grid_gdf[~grid_gdf["elevation"].isin(values_to_remove)]
grid_gdf = grid_gdf[~grid_gdf["elevation"].isnull()]

# if len(grid_gdf) < 1:
#     print("Reason for stop: The grid filtered out rows where LCZ = 0 and elevation = null. If no rows are left after this operation, there is nothing left to merge.")
#     break

print(era5_file)

year = era5_file.split("_")[0]

# Start with an empty GeoDataFrame
all_data = []

for i in range(1, 13):
    if (year == 2023) & (
        i > 8
    ):  # ERA5 data only available up to a certain date. No need to keep going.
        continue

    print("Month: " + str(i))

    era5_data = xr.open_dataset(era5_path + era5_file)

    now = datetime.now()
    current_time = now.strftime("%I:%M:%S %p")
    print("Prepping era5 data at", current_time)
    era5_data_month = era5_data.t2m.sel(
        time=era5_data["time.month"] == i
    )  # subset by month
    era5_data_month_df = (
        era5_data_month.to_dataframe().reset_index().dropna()
    )  # convert to dataframe, reset indices, drop nan
    era5_data_month_df["t2m"] = era5_data_month_df["t2m"] - 273.15  # convert to C
    # era5_data_month_df['longitude'] = era5_data_month_df['longitude'].apply(lambda x: x-360 if x > 180 else x) # convert 0:360 to -180:180

    now = datetime.now()
    current_time = now.strftime("%I:%M:%S %p")
    print("Converting era5 data to GeoDataFrame at", current_time)
    gdf_era5 = gpd.GeoDataFrame(
        era5_data_month_df,
        geometry=gpd.points_from_xy(
            era5_data_month_df.longitude, era5_data_month_df.latitude
        ),
    )

    now = datetime.now()
    merge1_time = now.strftime("%I:%M:%S %p")
    print("Spatial joining era5 data at time", merge1_time)
    gdf_merged_era5 = gpd.sjoin_nearest(
        grid_gdf,
        gdf_era5,
        how="left",
        distance_col="distances_era5",
        lsuffix="grid",
        rsuffix="era5",
    )
    gdf_merged_era5.drop(columns=["index_era5"], inplace=True)

    gdf_merged_era5.dropna(inplace=True)  # get rid of rows with NaN.

    # get rid of data that have not yet been "validated"
    now = datetime.now()
    export_time = now.strftime("%I:%M:%S %p")
    print("Checking if ERA5T data exists", export_time)
    if "expver" in gdf_merged_era5.columns:
        print("Removing ERA5T data")
        gdf_merged_era5 = gdf_merged_era5[gdf_merged_era5.expver != 5.0]
        gdf_merged_era5.drop(columns=["expver"], inplace=True)
    else:
        print("All included data is considered validated (not ERA5T)")

    # More cleaning/processing...
    now = datetime.now()
    export_time = now.strftime("%I:%M:%S %p")
    print("More cleaning/processing...", export_time)

    # Convert 'time' column to datetime format
    gdf_merged_era5["time"] = pd.to_datetime(gdf_merged_era5["time"])
    gdf_merged_era5["month"] = gdf_merged_era5["time"].dt.month
    gdf_merged_era5["year"] = gdf_merged_era5["time"].dt.year
    gdf_merged_era5["day_of_year"] = gdf_merged_era5["time"].dt.dayofyear

    # gdf_merged_era5 = gdf_merged_era5[gdf_merged_era5['lcz_value'] != 0.0]
    # values_to_remove = [-999.0, -999.9]  # replace with your actual values
    # gdf_merged_era5 = gdf_merged_era5[~gdf_merged_era5['elevation'].isin(values_to_remove)]
    # gdf_merged_era5 = gdf_merged_era5[~gdf_merged_era5['elevation'].isnull()]

    # drop columns not used
    gdf_merged_era5.drop(
        columns=["time", "distances_era5", "latitude", "longitude", "geometry"],
        inplace=True,
    )

    # rename columns to training data
    gdf_merged_era5.rename(
        columns={"latitude_left": "latitude_st", "longitude_left": "longitude_st"},
        inplace=True,
    )

    # Make sure these columns are integers...
    gdf_merged_era5["month"] = gdf_merged_era5["month"].astype(int)
    gdf_merged_era5["year"] = gdf_merged_era5["year"].astype(int)
    gdf_merged_era5["day_of_year"] = gdf_merged_era5["day_of_year"].astype(int)

    # add total_days_in_year to identify leap years
    gdf_merged_era5["total_days_in_year"] = (
        pd.to_datetime(gdf_merged_era5["year"], format="%Y").dt.is_leap_year + 365
    )

    print("Length of merged gdf: " + str(len(gdf_merged_era5)))

    now = datetime.now()
    export_time = now.strftime("%I:%M:%S %p")
    print("Appending gdf to list at time", export_time)
    # Append the new GeoDataFrame to the existing data
    all_data.append(gdf_merged_era5)

# Concatenate all GeoDataFrames in the list
now = datetime.now()
export_time = now.strftime("%I:%M:%S %p")
print("Concatenating dataframes at time", export_time)
all_data = gpd.pd.concat(all_data).reset_index(drop=True)

if not all_data.empty:
    now = datetime.now()
    export_time = now.strftime("%I:%M:%S %p")
    print("Outputting to file at time", export_time)
    all_data.to_feather(output_path + chunk_file_name + "_" + year + ".feather")

    now = datetime.now()
    export_time = now.strftime("%I:%M:%S %p")
    print("File saved!", export_time)
else:
    print("The DataFrame is empty, file not saved.")
