import pandas as pd
import xarray as xr
import rasterra as rt
from pathlib import Path

def get_era5_temperature(year: int | str, cs_df: pd.DataFrame):
    lat = xr.DataArray(cs_df["lat"].values, dims=["points"])
    lon = xr.DataArray(cs_df["lon"].values, dims=["points"])
    time = xr.DataArray(cs_df["date"].values, dims=["points"])
    
    era5 = xr.load_dataset(
        f"/mnt/share/erf/climate_downscale/extracted_data/era5_temperature_daily_mean/{year}_era5_temp_daily.nc"
    )

    era5 = era5.assign_coords(longitude=(((era5.longitude + 180) % 360) - 180)).sortby(['latitude', 'longitude'])
    arr = era5.sel(latitude=lat, longitude=lon, time=time, method="nearest")
    if "expver" in era5.coords:
        arr = arr.sel(expver=1).combine_first(arr.sel(expver=5))
    return arr['t2m'].to_numpy() - 273.15

year = 2023

# Load and cleanup
climate_stations = pd.read_parquet(
    f"/mnt/share/erf/climate_downscale/extracted_data/ncei_climate_stations/{year}.parquet"
)
column_map = {
    "DATE": "date",
    "LATITUDE": "lat",
    "LONGITUDE": "lon",
    "TEMP": "temperature",
    "ELEVATION": "ncei_elevation",
}
climate_stations = (
    climate_stations.rename(columns=column_map)
    .loc[:, list(column_map.values())]
    .dropna()
    .reset_index(drop=True)
)

# Do time things
climate_stations["date"] = pd.to_datetime(climate_stations["date"])
climate_stations["year"] = climate_stations["date"].dt.year
climate_stations["dayofyear"] = climate_stations["date"].dt.dayofyear

# Add temperature
climate_stations["temperature"] = 5 / 9 * (climate_stations["temperature"] - 32)
climate_stations['era5_temperature'] = get_era5_temperature(year, climate_stations)

# Elevation pieces
target_elevation = rt.load_mf_raster(list(Path("/mnt/share/erf/climate_downscale/model/predictors").glob("elevation_target_*.tif")))
climate_stations['target_elevation'] = srtm_elevation.select(climate_stations['lon'], climate_stations['lat'])
era5_elevation = rt.load_mf_raster(list(Path("/mnt/share/erf/climate_downscale/model/predictors").glob("elevation_era5_*.tif")))
climate_stations['era5_elevation'] = era5_elevation.select(climate_stations['lon'], climate_stations['lat'])

climate_stations['elevation'] = climate_stations['ncei_elevation']
missing_elevation = climate_stations['elevation'] < -999

climate_stations['elevation'] = climate_stations['ncei_elevation']
missing_elevation = climate_stations['elevation'] < -999
climate_stations.loc[missing_elevation, 'elevation'] = climate_stations.loc[missing_elevation, 'target_elevation']
still_missing_elevation = climate_stations['elevation'] < -999
climate_stations = climate_stations.loc[~still_missing_elevation]

# Local climate zone
target_lcz = rt.load_mf_raster(list(Path("/mnt/share/erf/climate_downscale/model/predictors").glob("lcz_target_*.tif")))
climate_stations['target_lcz'] = target_lcz.select(climate_stations['lon'], climate_stations['lat'])
era5_lcz = rt.load_mf_raster(list(Path("/mnt/share/erf/climate_downscale/model/predictors").glob("lcz_era5_*.tif")))
climate_stations['era5_lcz'] = era5_lcz.select(climate_stations['lon'], climate_stations['lat'])