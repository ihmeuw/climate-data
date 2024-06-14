import pandas as pd
import xarray as xr
from pathlib import Path
import numpy as np
import tqdm

from climate_downscale.generate import utils

TARGET_LON = xr.DataArray(
    np.round(np.arange(-180.0, 180.0, 0.1, dtype="float32"), 1), dims="longitude"
)
TARGET_LAT = xr.DataArray(
    np.round(np.arange(90.0, -90.1, -0.1, dtype="float32"), 1), dims="latitude"
)

variable = 'tas'
scenario = 'ssp119'
year = '2024'

paths = sorted(list(Path("/mnt/share/erf/climate_downscale/extracted_data/cmip6").glob("tas_ssp119*.nc")))
p = paths[0]

def compute_anomaly(path, year):
    reference_period = slice("2015-01-01", "2024-12-31")
    ref = xr.open_dataset(p).sel(time=reference_period).compute().groupby("time.month").mean("time")
    
    time_slice = slice(f"{year}-01", f"{year}-12")
    time_range = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    target = xr.open_dataset(p).sel(time=time_slice).compute()
    target = target.assign_coords(time=pd.to_datetime(target.time.dt.date)).interp_calendar(time_range)
    
    anomaly = target.groupby('time.month') - ref
    anomaly = anomaly.rename({'lat': 'latitude', 'lon': 'longitude'})
    anomaly = anomaly.assign_coords(longitude=(anomaly.longitude + 180) % 360 - 180).sortby("longitude")
    anomaly = utils.interpolate_to_target_latlon(anomaly, target_lat=TARGET_LAT, target_lon=TARGET_LON)

    return anomaly

a = 1 / len(paths) * compute_anomaly(paths[0], year)

for p in tqdm.tqdm(paths[1:]):
    a += 1 / len(paths) * compute_anomaly(p, year)

