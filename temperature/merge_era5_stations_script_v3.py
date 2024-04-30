import numpy as np
import pandas as pd
import rasterio as reo
from rasterio.enums import Resampling
from rasterio.crs import CRS
import sklearn
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray
import re
import os
import xarray as xr
import datetime as dt
from datetime import datetime
import argparse
from shapely.geometry import Polygon, LineString, Point
import getpass
import subprocess

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='filename')

args = parser.parse_args()

file = args.file

era5_path = '/mnt/share/erf/ERA5/three_hourly_temp_9km/daily_temps/'
stations_path = '/ihme/homes/nhashmeh/downscale_temperature/global_summaries/'
lcz_filepath = "/ihme/homes/nhashmeh/downscale_temperature/climate_zones/lcz_filter_v2.tif"
output_path = '/mnt/share/erf/ERA5/merged_era5_with_stations/'

now = datetime.now()
current_time = now.strftime("%I:%M:%S %p")
print("Loop start time =", current_time)

with reo.open(lcz_filepath) as src:

    year = file.split('_')[0] # get year

    # load in era5 data
    now = datetime.now()
    current_time = now.strftime("%I:%M:%S %p")
    print("Loading in era5 data at", current_time)
    era5_data = xr.open_dataset(era5_path + year + '_era5_temp_daily.nc')

    # load in station data
    now = datetime.now()
    current_time = now.strftime("%I:%M:%S %p")
    print("Loading in and processing station data at", current_time)
    stations_filename = year + '_all_data.csv'
    station_data = pd.read_csv(stations_path + stations_filename) 

    # processing station data...
    station_data.columns = station_data.columns.str.lower()
    station_data.rename(columns={'date': 'time'}, inplace = True)
    station_data['time'] = pd.to_datetime(station_data['time'])
    station_data = station_data.dropna(how = 'any', subset = ['latitude', 'longitude', 'temp', 'elevation']) # drop rows where there are no coords (data isn't always clean...)
    station_data['temp'] = (station_data['temp'] - 32) * 5/9 # convert to C
    station_data.drop_duplicates(inplace=True) # these apparently exist...
    station_data.drop(columns="station", inplace=True) # don't need station numbers...

    grouped_stations = station_data.groupby(station_data['time'].dt.month)

    now = datetime.now()
    current_time = now.strftime("%I:%M:%S %p")
    print("Looping through months at", current_time)
    for i in range(1,13): # loop through each month
        print("Month: " + str(i))

        group_i = grouped_stations.get_group(i)

        now = datetime.now()
        current_time = now.strftime("%I:%M:%S %p")
        print("Converting climate stations to GeoDataFrame at time", current_time)
        gdf_station = gpd.GeoDataFrame(group_i, geometry=gpd.points_from_xy(group_i.longitude, group_i.latitude))

        # Spatial join
        now = datetime.now()
        merge1_time = now.strftime("%I:%M:%S %p")
        print("Spatial joining between climate stations and LCZs at time", merge1_time)

        coords = np.transpose((gdf_station.geometry.x, gdf_station.geometry.y))

        # Transpose your coordinate pairs, sample from the raster, and get raster coordinates
        raster_values_and_coords = [(value, src.xy(*src.index(*coord))) for coord, value in zip(coords, src.sample(coords))]

        # Separate values and coordinates
        raster_values = [value[0] for value in raster_values_and_coords]
        raster_coords = [coord for value, coord in raster_values_and_coords]

        # Add the raster values and coordinates to your dataframe
        gdf_station['band_1'] = raster_values
        gdf_station['band_1'] = gdf_station['band_1'].apply(lambda x: x[0])
        gdf_station['lcz_coords'] = raster_coords
        gdf_station[['lcz_longitude', 'lcz_latitude']] = gdf_station['lcz_coords'].apply(pd.Series)
        gdf_station.drop(columns='lcz_coords', inplace=True)

        # Calculate distance between station and LCZ points
        st_points = gpd.GeoSeries(gdf_station.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1))
        lcz_points = gpd.GeoSeries(gdf_station.apply(lambda row: Point(row['lcz_longitude'], row['lcz_latitude']), axis=1))

        gdf_station['distances_lcz'] = st_points.distance(lcz_points)

        # Prep era5 data
        now = datetime.now()
        current_time = now.strftime("%I:%M:%S %p")
        print("Prepping era5 data at", current_time)
        era5_data_month = era5_data.t2m.sel(time=era5_data['time.month'] == i) # subset by month
        era5_data_month_df = era5_data_month.to_dataframe().reset_index().dropna() # convert to dataframe, reset indices, drop nan
        era5_data_month_df['t2m'] = era5_data_month_df['t2m'] - 273.15 # convert to C
        era5_data_month_df['longitude'] = era5_data_month_df['longitude'].apply(lambda x: x-360 if x > 180 else x) # convert 0:360 to -180:180

        now = datetime.now()
        current_time = now.strftime("%I:%M:%S %p")
        print("Converting era5 data to GeoDataFrame at", current_time)
        gdf_era5 = gpd.GeoDataFrame(era5_data_month_df, geometry=gpd.points_from_xy(era5_data_month_df.longitude, era5_data_month_df.latitude))

        # merge era5 and stations dataframes
        now = datetime.now()
        merge1_time = now.strftime("%I:%M:%S %p")
        print("Spatial joining era5 data at time", merge1_time)
        # Currently setting max distance for join to 3 (distance in degrees between lat/lon points)
        gdf_merged_final = gpd.sjoin_nearest(gdf_station, gdf_era5, how="left", distance_col="distances_era5", lsuffix='st', rsuffix='era5', max_distance = 3)

        now = datetime.now()
        current_time = now.strftime("%I:%M:%S %p")
        print("Cleaning up final dataframe...", current_time)            
        # Drop unneeded columns that came from the era5 dataframe
        gdf_merged_final.drop(columns=['index_era5'], inplace=True)

        # get time matching rows because spatial join doesn't include the time aspect....
        gdf_merged_final = gdf_merged_final.loc[gdf_merged_final["time_st"]==gdf_merged_final["time_era5"]]

        # There are cases where multiple stations exist within a single era5 pixel. These often have very similar elevations and temperatures,
        # sometimes even the same elevation but different temps (or the other way around). Currently, I am keeping these in order to have the model
        # capture this behavior, though it probably doesn't matter all that much.

        # Because I used sjoin_nearest, every era5 pixel will try to find the nearest station it can join on, but we only want one per station.
        # This results in many rows where stations are repeated because era5 matched different pixels to a given station multiple times.
        # To deal with this, we take the row with the minimum distance between the pixel and the station to represent the era5 temp for
        # that station.

        gdf_merged_final = gdf_merged_final.loc[gdf_merged_final.groupby(['latitude_st', 'longitude_st', 'time_st'])['distances_era5'].idxmin()]

        # get rid of data that have not yet been "validated"
        now = datetime.now()
        export_time = now.strftime("%I:%M:%S %p")
        print("Checking if ERA5T data exists", export_time)
        if 'expver' in gdf_merged_final.columns:
            print("Removing ERA5T data")
            gdf_merged_final = gdf_merged_final[gdf_merged_final.expver != 5.0]
            gdf_merged_final.drop(columns = ['expver'], inplace=True)
        else:
            print("All included data is considered validated (not ERA5T)")

        # More cleaning/processing...
        now = datetime.now()
        export_time = now.strftime("%I:%M:%S %p")
        print("More cleaning/processing...", export_time)

        # Convert 'time' column to datetime format
        gdf_merged_final['time'] = pd.to_datetime(gdf_merged_final['time_st'])
        gdf_merged_final['month'] = gdf_merged_final['time'].dt.month
        gdf_merged_final['year'] = gdf_merged_final['time'].dt.year
        gdf_merged_final['day_of_year'] = gdf_merged_final['time'].dt.dayofyear

        # Create 'total_days_in_year' feature 
        gdf_merged_final['total_days_in_year'] = gdf_merged_final['time'].dt.is_leap_year + 365

        # drop columns not used
        gdf_merged_final.drop(columns=['time', 'time_st', 'geometry', 'lcz_longitude', 'lcz_latitude', 'distances_lcz',
                                      'time_era5', 'latitude_era5', 'longitude_era5', 'distances_era5',], inplace=True)

        now = datetime.now()
        export_time = now.strftime("%I:%M:%S %p")
        print("Exporting to .feather at time", export_time)
        gdf_merged_final.to_feather(output_path + year + '_' + str(i) + '_era5_stations_lcz_merged.feather', index=False)

        now = datetime.now()
        export_time = now.strftime("%I:%M:%S %p")
        print("File exported at time", export_time)