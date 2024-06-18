from collections.abc import Callable

import numpy as np
import xarray as xr

REFERENCE_YEARS = list(range(2018, 2024))
REFERENCE_PERIOD = slice(f"{REFERENCE_YEARS[0]}-01-01", f"{REFERENCE_YEARS[-1]}-12-31")
TARGET_LON = xr.DataArray(
    np.round(np.arange(-180.0, 180.0, 0.1, dtype="float32"), 1), dims="longitude"
)
TARGET_LAT = xr.DataArray(
    np.round(np.arange(-90.0, 90.1, 0.1, dtype="float32"), 1), dims="latitude"
)

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


########################
# Data transformations #
########################


def vector_magnitude(x: xr.Dataset, y: xr.Dataset) -> xr.Dataset:
    """Calculate the magnitude of a vector."""
    return np.sqrt(x**2 + y**2)  # type: ignore[no-any-return]


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


def heat_index(
    temperature_c: xr.Dataset,
    relative_humidity_percent: xr.Dataset,
) -> xr.Dataset:
    """Calculate the heat index.

    https://www.weather.gov/media/ffc/ta_htindx.PDF

    Parameters
    ----------
    temperature_c
        Temperature in Celsius
    relative_humidity_percent
        Relative humidity as a percentage

    Returns
    -------
    xr.Dataset
        Heat index in Celsius
    """
    # Alias for simplicity in the formula
    t = temperature_c
    r = relative_humidity_percent

    # Heat index formula from canonical multi-variable regression
    hi_raw = (
        -8.784695
        + 1.61139411 * t
        + 2.338549 * r
        - 0.14611605 * t * r
        - 1.2308094e-2 * t**2
        - 1.6424828e-2 * r**2
        + 2.211732e-3 * t**2 * r
        + 7.2546e-4 * t * r**2
        - 3.582e-6 * t**2 * r**2
    )
    # Below 20 degrees, the heat index is the same as the temperature
    hi_threshold = 20
    hi = xr.where(t > hi_threshold, hi_raw, t)  # type: ignore[no-untyped-call]
    return hi  # type: ignore[no-any-return]


def humidex(
    temperature_c: xr.Dataset,
    relative_humidity_percent: xr.Dataset,
) -> xr.Dataset:
    """Calculate the humidex.

    https://en.wikipedia.org/wiki/Humidex

    Parameters
    ----------
    temperature_c
        Temperature in Celsius
    relative_humidity_percent
        Relative humidity as a percentage

    Returns
    -------
    xr.Dataset
        Humidex in Celsius
    """
    svp = buck_vapor_pressure(temperature_c)
    vp = relative_humidity_percent / 100 * svp
    return temperature_c + 0.5555 * (vp - 10)


def effective_temperature(
    temperature_c: xr.Dataset,
    relative_humidity_percent: xr.Dataset,
    wind_speed_m_s: xr.Dataset,
) -> xr.Dataset:
    """Calculate the effective temperature.

    https://www.sciencedirect.com/topics/engineering/effective-temperature

    Parameters
    ----------
    temperature_c
        Temperature in Celsius
    relative_humidity_percent
        Relative humidity as a percentage
    wind_speed_m_s
        Wind speed in m/s

    Returns
    -------
    xr.Dataset
        Effective temperature in Celsius
    """
    # Alias for simplicity in the formula
    t = temperature_c
    r = relative_humidity_percent
    v = wind_speed_m_s

    wind_adjustment = 1 / (1.76 + 1.4 * v**0.75)
    et = (
        37
        - ((37 - t) / (0.68 - 0.0014 * r + wind_adjustment))
        - 0.29 * t * (1 - 0.01 * r)
    )
    return et


################
# Data cleanup #
################


def rename_val_column(ds: xr.Dataset) -> xr.Dataset:
    data_var = next(iter(ds))
    return ds.rename({data_var: "value"})


def interpolate_to_target_latlon(
    ds: xr.Dataset,
    method: str = "nearest",
) -> xr.Dataset:
    return (
        ds.interp(longitude=TARGET_LON, latitude=TARGET_LAT, method=method)  # type: ignore[arg-type]
        .interpolate_na(dim="longitude", method="nearest", fill_value="extrapolate")
        .sortby("latitude")
        .interpolate_na(dim="latitude", method="nearest", fill_value="extrapolate")
        .sortby("latitude", ascending=False)
    )
