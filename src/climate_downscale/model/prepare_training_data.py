year = 2023

climate_stations = pd.read_parquet(f'/mnt/share/erf/climate_downscale/extracted_data/ncei_climate_stations/{year}.parquet')
column_map = {
    "DATE": "date",
    "LATITUDE": "lat",
    "LONGITUDE": "lon",
    "TEMP": "temperature",
}
climate_stations = climate_stations.rename(columns=column_map).loc[:, list(column_map.values())].dropna()
climate_stations['date'] = pd.to_datetime(climate_stations['date'])
climate_stations['year'] = climate_stations['date'].dt.year
climate_stations['dayofyear'] = climate_stations['date'].dt.dayofyear
climate_stations['temperature'] = 5/9 * (climate_stations['temperature'] - 32)
climate_stations.loc[climate_stations.lon < 0, 'lon'] +=360

era5 = xr.load_dataset(f'/mnt/share/erf/climate_downscale/extracted_data/era5_temperature_daily_mean/{year}_era5_temp_daily.nc')
lat = xr.DataArray(climate_stations['lat'].values, dims=['points'])
lon = xr.DataArray(climate_stations['lon'].values, dims=['points'])
time = xr.DataArray(climate_stations['date'].values, dims=['points'])
arr = era5.sel(latitude=lat, longitude=lon, time=time, method='nearest')
if "expver" in arr.coords:
    arr = arr.sel(expver=1).combine_first(arr.sel(expver=5))
climate_stations['era5_temperature'] = arr['t2m'].to_numpy() + 273.15