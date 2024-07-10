import numpy as np
import rasterra as rt
import xarray as xr
from affine import Affine


def to_raster(
    ds: xr.DataArray,
    nodata: float | int,
    lat_col: str = "lat",
    lon_col: str = "lon",
    crs: str = "EPSG:4326",
) -> rt.RasterArray:
    """Convert an xarray DataArray to a RasterArray."""
    lat, lon = ds[lat_col].data, ds[lon_col].data

    dlat = (lat[1:] - lat[:-1]).mean()
    dlon = (lon[1:] - lon[:-1]).mean()

    transform = Affine(
        a=dlon,
        b=0.0,
        c=lon[0],
        d=0.0,
        e=-dlat,
        f=lat[-1],
    )
    return rt.RasterArray(
        data=ds.data[::-1],
        transform=transform,
        crs=crs,
        no_data_value=nodata,
    )


def make_raster_template(
    x_min: int | float,
    y_min: int | float,
    stride: int | float,
    resolution: int | float,
    crs: str = "EPSG:4326",
) -> rt.RasterArray:
    """Create a raster template with the specified dimensions and resolution."""
    tolerance = 1e-12
    evenly_divides = (stride % resolution < tolerance) or (
        resolution - stride % resolution < tolerance
    )
    if not evenly_divides:
        msg = "Stride must be a multiple of resolution"
        raise ValueError(msg)

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
    return rt.RasterArray(
        data,
        transform,
        crs=crs,
        no_data_value=-1,
    )
