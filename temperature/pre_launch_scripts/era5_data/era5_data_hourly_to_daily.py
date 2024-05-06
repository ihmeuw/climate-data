import numpy as np
import pandas as pd
import rasterio as reo
from rasterio.enums import Resampling
from rasterio.crs import CRS
import sklearn
import geopandas
import matplotlib.pyplot as plt
import rioxarray
import re
import os
import xarray as xr
import datetime as dt
from datetime import datetime
import argparse
import getpass
import subprocess

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='filename')

args = parser.parse_args()

file = args.file

path_to_files = '/mnt/share/erf/ERA5/three_hourly_temp_9km/downloaded_unzipped/'
output_path = '/mnt/share/erf/ERA5/three_hourly_temp_9km/daily_temps/'

get_name = file.split(".")[0]
new_filename = get_name + "_daily.nc"
data = xr.open_dataset(path_to_files + file) # loads in data
print("Calculating daily averages from hourly data for " + get_name)
daily_data = data.resample(time='D').mean() # calculates daily averages
print("Saving new daily averages as " + new_filename)
daily_data.to_netcdf(output_path + new_filename)