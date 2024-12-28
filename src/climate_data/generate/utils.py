import typing
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

#############################
# Standard unit conversions #
#############################


def kelvin_to_celsius(temperature_k: xr.Dataset) -> xr.Dataset:
    """Convert temperature from Kelvin to Celsius

    Parameters
    ----------
    temperature_k
        Temperature in Kelvin

    Returns
    -------
    xr.Dataset
        Temperature in Celsius
    """
    return temperature_k - 273.15


def meter_to_millimeter(rainfall_m: xr.Dataset) -> xr.Dataset:
    """Convert rainfall from meters to millimeters

    Parameters
    ----------
    rainfall_m
        Rainfall in meters

    Returns
    -------
    xr.Dataset
        Rainfall in millimeters
    """
    return 1000 * rainfall_m


def precipitation_flux_to_rainfall(precipitation_flux: xr.Dataset) -> xr.Dataset:
    """Convert precipitation flux to rainfall

    Parameters
    ----------
    precipitation_flux
        Precipitation flux in kg m-2 s-1

    Returns
    -------
    xr.Dataset
        Rainfall in mm/day
    """
    seconds_per_day = 86400
    mm_per_kg_m2 = 1
    return seconds_per_day * mm_per_kg_m2 * precipitation_flux


def scale_wind_speed_height(wind_speed_10m: xr.Dataset) -> xr.Dataset:
    """Scaling wind speed from a height of 10 meters to a height of 2 meters

    Reference: Bröde et al. (2012)
    https://doi.org/10.1007/s00484-011-0454-1

    Parameters
    ----------
    wind_speed_10m
        The 10m wind speed [m/s]. May be signed (ie a velocity component)

    Returns
    -------
    xr.DataSet
        The 2m wind speed [m/s]. May be signed (ie a velocity component)
    """
    scale_factor = np.log10(2 / 0.01) / np.log10(10 / 0.01)
    return scale_factor * wind_speed_10m  # type: ignore[no-any-return]


def identity(ds: xr.Dataset) -> xr.Dataset:
    """Identity transformation"""
    return ds


######################
# Standard summaries #
######################


def daily_mean(ds: xr.Dataset) -> xr.Dataset:
    return ds.groupby("time.date").mean()


def annual_mean(ds: xr.Dataset) -> xr.Dataset:
    return ds.groupby("date.year").mean()


def daily_max(ds: xr.Dataset) -> xr.Dataset:
    return ds.groupby("time.date").max()


def annual_max(ds: xr.Dataset) -> xr.Dataset:
    return ds.groupby("date.year").max()


def daily_min(ds: xr.Dataset) -> xr.Dataset:
    return ds.groupby("time.date").min()


def annual_min(ds: xr.Dataset) -> xr.Dataset:
    return ds.groupby("date.year").min()


def daily_sum(ds: xr.Dataset) -> xr.Dataset:
    return ds.groupby("time.date").sum()


def annual_sum(ds: xr.Dataset) -> xr.Dataset:
    return ds.groupby("date.year").sum()


def count_threshold(threshold: int | float) -> Callable[[xr.Dataset], xr.Dataset]:
    def count(ds: xr.Dataset) -> xr.Dataset:
        return ds > threshold

    return count


def count_between_threshold(
    lower_threshold: int | float, upper_threshold: int | float
) -> Callable[[xr.Dataset], xr.Dataset]:
    def count(ds: xr.Dataset) -> xr.Dataset:
        return (ds > lower_threshold) & (ds < upper_threshold)

    return count


def _load_suitability_curve(
    disease: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    df = pd.read_parquet(
        Path(__file__).parent / "supplementary_data" / f"{disease}_suitability.parquet"
    )
    t, s = df["temperature"].to_numpy(), df["suitability"].to_numpy()
    return t, s


def map_suitability(disease: str) -> Callable[[xr.Dataset], xr.Dataset]:
    diseases = ["dengue", "malaria"]
    if disease not in diseases:
        msg = f"Invalid disease: {disease}. Must be one of {', '.join(diseases)}"
        raise ValueError(msg)

    def smap(ds: xr.Dataset) -> xr.Dataset:
        t, s = _load_suitability_curve(disease)
        ds["value"] = (("date", "latitude", "longitude"), np.interp(ds["value"], t, s))
        return ds

    return smap


########################
# Data transformations #
########################


def vector_magnitude(x: xr.Dataset, y: xr.Dataset) -> xr.Dataset:
    """Calculate the magnitude of a vector."""
    return np.sqrt(x**2 + y**2)  # type: ignore[return-value]


def buck_vapor_pressure(temperature_c: xr.Dataset) -> xr.Dataset:
    """Approximate vapor pressure of water.

    https://en.wikipedia.org/wiki/Arden_Buck_equation
    https://journals.ametsoc.org/view/journals/apme/20/12/1520-0450_1981_020_1527_nefcvp_2_0_co_2.xml

    Parameters
    ----------
    temperature_c
        Temperature in Celsius

    Returns
    -------
    xr.Dataset
        Vapor pressure in hPa
    """
    over_water = 6.1121 * np.exp(
        (18.678 - temperature_c / 234.5) * (temperature_c / (257.14 + temperature_c))
    )
    over_ice = 6.1115 * np.exp(
        (23.036 - temperature_c / 333.7) * (temperature_c / (279.82 + temperature_c))
    )
    vp = xr.where(temperature_c > 0, over_water, over_ice)  # type: ignore[no-untyped-call]
    return vp  # type: ignore[no-any-return]


def rh_percent(
    temperature_c: xr.Dataset, dewpoint_temperature_c: xr.Dataset
) -> xr.Dataset:
    """Calculate relative humidity from temperature and dewpoint temperature.

    Parameters
    ----------
    temperature_c
        Temperature in Celsius
    dewpoint_temperature_c
        Dewpoint temperature in Celsius

    Returns
    -------
    xr.Dataset
        Relative humidity as a percentage
    """
    # saturation vapour pressure
    svp = buck_vapor_pressure(temperature_c)
    # actual vapour pressure
    vp = buck_vapor_pressure(dewpoint_temperature_c)
    return 100 * vp / svp


################
# Data cleanup #
################


def rename_val_column(ds: xr.Dataset) -> xr.Dataset:
    data_var = next(iter(ds))
    return ds.rename({data_var: "value"})


def interpolate_to_target_latlon(
    ds: xr.Dataset,
    method: str = "nearest",
    target_lon: xr.DataArray | None = None,
    target_lat: xr.DataArray | None = None,
) -> xr.Dataset:
    """Interpolate a dataset to a target latitude and longitude grid.

    Parameters
    ----------
    ds
        Dataset to interpolate
    method
        Interpolation method
    target_lon
        Target longitude grid
    target_lat
        Target latitude grid

    Returns
    -------
    xr.Dataset
        Interpolated dataset
    """
    valid_lat_lon = (
        (target_lat is not None and target_lon is not None)
        or (target_lat is None and target_lon is None)
    )
    if not valid_lat_lon:
        msg = "Both target_lat and target_lon must be provided or neither"
        raise ValueError(msg)
    if target_lon is None:
        # Avoid cyclic imports
        from climate_data.constants import TARGET_LON, TARGET_LAT  # noqa: I001
        target_lon = TARGET_LON
        target_lat = TARGET_LAT

        target_lon = np.round(np.arange(-180.0, 180.0, 0.1, dtype="float32"), 1)

    return (
        ds.interp(longitude=target_lon, latitude=target_lat, method=method)  # type: ignore[arg-type]
        .interpolate_na(dim="longitude", method="nearest", fill_value="extrapolate")
        .sortby("latitude")
        .interpolate_na(dim="latitude", method="nearest", fill_value="extrapolate")
        .sortby("latitude", ascending=False)
    )


class Transform:
    def __init__(
        self,
        source_variables: list[str],
        transform_funcs: list[typing.Callable[..., xr.Dataset]]
        | dict[str, list[typing.Callable[..., xr.Dataset]]],
        encoding_scale: float = 1.0,
        encoding_offset: float = 0.0,
    ):
        self.source_variables = source_variables
        self.transform_funcs = transform_funcs
        self.encoding_scale = encoding_scale
        self.encoding_offset = encoding_offset

    def __call__(self, *datasets: xr.Dataset, key: str | None = None) -> xr.Dataset:
        if len(datasets) > 1:
            # Enforce consistency in the spatial dimensions of the input datasets
            datasets = list(datasets)
            target_lat = datasets[0].latitude
            target_lon = datasets[0].longitude
            for i in range(1, len(datasets)):
                datasets[i] = interpolate_to_target_latlon(
                    datasets[i], method="linear", target_lon=target_lon, target_lat=target_lat
                )

        if isinstance(self.transform_funcs, dict):
            if key is None:
                msg = "Key must be provided for dict transform"
                raise ValueError(msg)
            transform_funcs = self.transform_funcs[key]
        else:
            transform_funcs = self.transform_funcs
        # first function is applied to the input data which can have multiple xarray datasets
        res = transform_funcs[0](*datasets)
        for transform_func in transform_funcs[1:]:
            res = transform_func(res)
        return res

    @property
    def encoding_kwargs(self) -> dict[str, float]:
        if self.encoding_offset != 0.0 or self.encoding_scale != 1:
            return {
                "add_offset": self.encoding_offset,
                "scale_factor": self.encoding_scale,
            }
        return {}
