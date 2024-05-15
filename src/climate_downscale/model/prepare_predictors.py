import geopandas as gpd
import pandas as pd
import xarray as xr
import tqdm
import matplotlib.pyplot as plt
import rasterra as rt
from affine import Affine
import numpy as np
from pathlib import Path

def to_raster(ds, nodata, lat_col='lat', lon_col='lon'):
    lat, lon = ds[lat_col].data, ds[lon_col].data

    dlat = (lat[1:] - lat[:-1]).mean()
    dlon = (lon[1:] - lon[:-1]).mean()

    transform = Affine(
        a=dlon,
        b=0.,
        c=lon[0],
        d=0.,
        e=-dlat,
        f=lat[-1],
    )
    raster = rt.RasterArray(
        data = ds.data,
        transform=transform,
        crs='EPSG:4326',
        no_data_value=nodata,
    )
    return raster
    
def make_template(x_min, y_min, stride, resolution):
    evenly_divides = (
        (stride % resolution < 1e-12)
        or (resolution - stride % resolution < 1e-12)
    )
    assert evenly_divides
    
    transform = Affine(
        a=resolution,
        b=0,
        c=x_min,
        d=0,
        e=-resolution,
        f=y_min + stride,
    )
    
    n_pix = int(stride / resolution)
    
    data = np.zeros((n_pix, n_pix), dtype=np.int8)
    template = rt.RasterArray(
        data,
        transform,
        crs='EPSG:4326',
        no_data_value=-1,
    )
    return template


STRIDE = 30  # degrees
PAD = 1
lat_start = 0
lon_start = 0

longitudes = range(lon_start - PAD, lon_start + STRIDE + PAD)
latitudes = range(lat_start - PAD, lat_start + STRIDE + PAD)

template_era5 = make_template(x_min=lon_start, y_min=lat_start, stride=STRIDE, resolution=0.1)
template_target = make_template(x_min=lon_start, y_min=lat_start, stride=STRIDE, resolution=0.01)

root = Path("/mnt/share/erf/climate_downscale/extracted_data/open_topography_elevation/SRTM_GL3_srtm")
paths = []
for lon in longitudes:
    lon_stub = f"E{lon:03}" if lon >= 0 else f"W{-lon:03}"
        
    for lat in range(lat_start, lat_start+STRIDE): 
        if lat >= 30:
            rel_path = f"North/North_30_60/N{lat:02}{lon_stub}.tif"
        elif lat >=0:
            rel_path = f"North/North_0_29/N{lat:02}{lon_stub}.tif"
        else:
            rel_path = f"South/S{-lat:02}{lon_stub}.tif"
        
        p = root / rel_path

        if p.exists():
            paths.append(p)

elevation = rt.load_mf_raster(paths)

elevation_target = elevation.resample_to(template_target, resampling='average')
elevation_era5 = elevation.resample_to(template_era5, resampling='average').resample_to(template_target, resampling='nearest')
elevation_anomaly = elevation_era5 - elevation_target