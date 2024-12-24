"""
Climate Data Utilities
----------------------

Utility functions for working with climate data.
"""

import numpy as np
import rasterra as rt
import xarray as xr
from affine import Affine


def to_raster(
    ds: xr.DataArray,
    no_data_value: float | int,
    lat_col: str = "lat",
    lon_col: str = "lon",
    crs: str = "EPSG:4326",
) -> rt.RasterArray:
    """Convert an xarray DataArray to a RasterArray.

    Parameters
    ----------
    ds
        The xarray DataArray to convert.
    no_data_value
        The value to use for missing data. This should be consistent with the dtype of the data.
    lat_col
        The name of the latitude coordinate in the dataset.
    lon_col
        The name of the longitude coordinate in the dataset.
    crs
        The coordinate reference system of the data.

    Returns
    -------
    rt.RasterArray
        The RasterArray representation of the input data.
    """
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
        no_data_value=no_data_value,
    )


def make_raster_template(
    x_min: int | float,
    y_min: int | float,
    stride: int | float,
    resolution: int | float,
    crs: str = "EPSG:4326",
) -> rt.RasterArray:
    """Create a raster template with the specified dimensions and resolution.

    A raster template is a RasterArray with a specified extent, resolution, and CRS. The data
    values are initialized to zero. This function is useful for creating a template to use
    when resampling another raster to a common grid.

    Parameters
    ----------
    x_min
        The minimum x-coordinate of the raster.
    y_min
        The minimum y-coordinate of the raster.
    stride
        The length of one side of the raster in the x and y directions measured in the units
        of the provided coordinate reference system.
    resolution
        The resolution of the raster in the units of the provided coordinate reference system.
    crs
        The coordinate reference system of the generated raster.

    Returns
    -------
    rt.RasterArray
        A raster template with the specified dimensions and resolution.
    """
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
