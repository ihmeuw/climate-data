import numpy as np
import pandas as pd
from joblib import load
import sys
from datetime import datetime, timedelta
from joblib import load
import argparse
import os
import xarray as xr


# task_id = int(sys.argv[1])
day = int(sys.argv[1])
year = int(sys.argv[2])
folder = sys.argv[3]
output_path = sys.argv[4]
print("day: ", day)
print("year: ", year)
print("folder: ", folder)
print("output_path: ", output_path)

final_data = pd.DataFrame()

for file in os.listdir(folder):
    if file.endswith('.feather'):
        print(file)
        # load the data
        print("Loading data")
        data = pd.read_feather(folder + file)
        # drop index column
        print("Dropping unnecessary columns")
        # data = data.drop(columns='index')
        data = data[['latitude_st', 'longitude_st', 'day_of_year', 'predictions']]
        # filter to get day
        print("Filtering to get day: ", day)
        data_day = data[data['day_of_year'] == day]
        # round lat/lon for storage
        print("Rounding lat/lon")
        data_day['latitude_st'] = data_day['latitude_st'].round(10)
        data_day['longitude_st'] = data_day['longitude_st'].round(10)
        data_day['predictions'] = data_day['predictions'].round(3)
        print("Appending data")
        # Append the data for the current day to the larger DataFrame
        final_data = pd.concat([final_data, data_day])

# Create a datetime object representing the first day of the year
date = datetime(year, 1, 1)

# Subtract one from the day of the year (since timedelta is zero-indexed)
day_of_year = day - 1

# Add the timedelta to the datetime object to get the date
date = date + timedelta(days=day_of_year)

# Format the datetime object into a string
date_string = date.strftime('%Y_%m_%d')
print("Here is the date string: ", date_string)

# replace old time columns with date
final_data.rename(columns={'latitude_st': 'lat', 'longitude_st': 'lon'}, inplace=True)
final_data = final_data[['lat', 'lon', 'predictions']]
final_data['date'] = date_string

filename = 'predictions_' + date_string

print('resetting index')
final_data.set_index(['date', 'lat', 'lon'], inplace=True)

# convert the dataframe to an xarray Dataset
print('converting to xarray Dataset')
final_data = xr.Dataset.from_dataframe(final_data)

# Calculate the scale factor and add offset
min_value = final_data['predictions'].min(dim=('date', 'lat', 'lon')).item()
max_value = final_data['predictions'].max(dim=('date', 'lat', 'lon')).item()
scale_factor = (max_value - min_value) / (2**16 - 1)
add_offset = min_value

# create the directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

print("saving file to: ", output_path + filename)

# save the Dataset to a netCDF file
final_data.to_netcdf(output_path + filename + '.nc',
            encoding={
                'predictions': {
                    'dtype': 'float32', 
                    'scale_factor': scale_factor, 
                    'add_offset': add_offset, 
                    '_FillValue': -9999,
                    'zlib': True
                    }
                })