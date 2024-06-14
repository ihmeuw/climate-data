from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

TARGET_LON = xr.DataArray(
    np.round(np.arange(0.0, 360.0, 0.1, dtype="float32"), 1), dims="longitude"
)
TARGET_LAT = xr.DataArray(
    np.round(np.arange(90.0, -90.1, -0.1, dtype="float32"), 1), dims="latitude"
)


def kelvin_to_celsius(temperature_k):
    return temperature_k - 273.15


def m_to_mm(ds):
    return 1000 * ds


def scale_windspeed(windspeed):
    """Scaling wind speed from a height of 10 meters to a height of 2 meters

    Reference: Bröde et al. (2012)
    https://doi.org/10.1007/s00484-011-0454-1

    Parameters
    ----------
    ds
        The 10m wind speed [m/s]. May be signed (ie a velocity component)

    Returnds
    --------
    xr.DataSet
        The 2m wind speed [m/s]. May be signed (ie a velocity component)
    """
    scale_factor = np.log10(2 / 0.01) / np.log10(10 / 0.01)
    return scale_factor * windspeed


def identity(ds):
    return ds


def rename_val_column(ds):
    data_var = next(iter(ds))
    return ds.rename({data_var: "value"})


convert_map = {
    "10m_u_component_of_wind": scale_windspeed,
    "10m_v_component_of_wind": scale_windspeed,
    "2m_dewpoint_temperature": kelvin_to_celsius,
    "2m_temperature": kelvin_to_celsius,
    "surface_net_solar_radiation": identity,
    "surface_net_thermal_radiation": identity,
    "surface_pressure": identity,
    "surface_solar_radiation_downwards": identity,
    "surface_thermal_radiation_downwards": identity,
    "total_precipitation": m_to_mm,
    "total_sky_direct_solar_radiation_at_surface": identity,
}


def interpolate_to_target(ds):
    return ds.interp(
        longitude=TARGET_LON, latitude=TARGET_LAT, method="nearest"
    ).interpolate_na(dim="longitude", method="nearest", fill_value="extrapolate")


def load_variable(variable, year, month, dataset="single-levels"):
    root = Path("/mnt/share/erf/climate_downscale/extracted_data/era5")
    p = root / f"reanalysis-era5-{dataset}_{variable}_{year}_{month}.nc"
    if dataset == "land" and not p.exists():
        # Substitute the single level dataset pre-interpolated at the target resolution.
        p = root / f"reanalysis-era5-single-levels_{source_variable}_{year}_{month}.nc"
        ds = interpolate_to_target(xr.load_dataset(p))
    elif dataset == "land":
        ds = xr.load_dataset(p).assign_coords(latitude=TARGET_LAT, longitude=TARGET_LON)
    else:
        ds = xr.load_dataset(p)
    conversion = convert_map[variable]
    ds = conversion(rename_val_column(ds))
    return ds


########


def daily_mean(ds):
    return ds.groupby("time.date").mean()


def daily_max(ds):
    return ds.groupby("time.date").max()


def daily_min(ds):
    return ds.groupby("time.date").min()


def daily_sum(ds):
    return ds.groupby("time.date").sum()


def cdd(temperature_c):
    return np.maximum(temperature_c - 18, 0).groupby("time.date").mean()


def hdd(temperature_c):
    return np.maximum(18 - temperature_c, 0).groupby("time.date").mean()


def vector_magnitude(x, y):
    return np.sqrt(x**2 + y**2)


def buck_vapor_presure(temperature_c):
    """Approximate vapor pressure of water.

    https://en.wikipedia.org/wiki/Arden_Buck_equation
    https://journals.ametsoc.org/view/journals/apme/20/12/1520-0450_1981_020_1527_nefcvp_2_0_co_2.xml
    """
    over_water = 6.1121 * np.exp(
        (18.678 - temperature_c / 234.5) * (temperature_c / (257.14 + temperature_c))
    )
    over_ice = 6.1115 * np.exp(
        (23.036 - temperature_c / 333.7) * (temperature_c / (279.82 + temperature_c))
    )
    return xr.where(temperature_c > 0, over_water, over_ice)


def rh_percent(temperature_c, dewpoint_temperature_c):
    # saturated vapour pressure
    es = buck_vapor_pressure(temperature_c)
    # vapour pressure
    e = buck_vapor_pressure(dewpoint_temperature_c)
    rh = (e / es) * 100
    return rh


def heat_index(temperature_c, dewpoint_temperature_c):
    t = temperature_c  # Alias for simplicity in the formula
    r = rh_percent(temperature_c, dewpoint_temperature_c)

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
    hi = xr.where(t > 20, hi_raw, t)
    return hi


def humidex(temperature_c, dewpoint_temperature_c):
    vp = buck_vapor_pressure(dewpoint_temperature_c)
    return temperature_c + 0.5555 * (vp - 10)


def effective_temperature(temperature_c, dewpoint_temperature_c, uas, vas):
    """https://www.sciencedirect.com/topics/engineering/effective-temperature"""
    t = temperature_c
    r = rh_percent(temperature_c, dewpoint_temperature_c)
    v = vector_magnitude(uas, vas)

    wind_adjustment = 1 / (1.76 + 1.4 * v**0.75)
    et = (
        37
        - ((37 - t) / (0.68 - 0.0014 * r + wind_adjustment))
        - 0.29 * t * (1 - 0.01 * r)
    )
    return et


collapse_map = {
    "mean_temperature": (["2m_temperature"], daily_mean, (273.15, 0.01)),
    "max_temperature": (["2m_temperature"], daily_max, (273.15, 0.01)),
    "min_temperature": (["2m_temperature"], daily_min, (273.15, 0.01)),
    "cooling_degree_days": (["2m_temperature"], cdd, (0, 0.01)),
    "heating_degree_days": (["2m_temperature"], hdd, (0, 0.01)),
    "wind_speed": (
        ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        lambda x, y: daily_mean(vector_magnitude(x, y)),
        (0, 0.01),
    ),
    "relative_humidity": (
        ["2m_temperature", "2m_dewpoint_temperature"],
        lambda x, y: daily_mean(rh_percent(x, y)),
        (0, 0.01),
    ),
    "total_precipitation": (["total_precipitation"], daily_sum, (0, 0.1)),
    # "heat_index": (
    #     ["2m_temperature", "2m_dewpoint_temperature"], lambda x, y: daily_mean(heat_index(x, y)), (273.15, 0.01)
    # ),
    # "humidex": (
    #     ['2m_temperature', '2m_dewpoint_temperature'], lambda x, y: daily_mean(humidex(x, y)), (273.15, 0.01)
    # ),
    # "normal_effective_temperature": (
    #     ["2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
    #     lambda *args: daily_mean(effective_temperature(*args)), (273.15, 0.01)
    # ),
}

year = "1990"
month = "01"
target_variable = "wind_speed"

source_variables, collapse_fun, (e_offset, e_scale) = collapse_map[target_variable]

print("loading single-levels")
single_level = [
    load_variable(sv, year, month, "single-levels") for sv in source_variables
]
print("collapsing")
ds = collapse_fun(*single_level)
ds = ds.assign(date=pd.to_datetime(ds.date))

print("interpolating")
ds_land_res = interpolate_to_target(ds)

print("loading land")
land = [load_variable(sv, year, month, "land") for sv in source_variables]
print("collapsing")
ds_land = collapse_fun(*land)
ds_land = ds_land.assign(date=pd.to_datetime(ds_land.date))

print("combining")
combined = ds_land.combine_first(ds_land_res)

combined.to_netcdf(
    "compressed.nc",
    encoding={
        "value": {
            "dtype": "int16",
            "add_offset": e_offset,
            "scale_factor": e_scale,
            "_FillValue": -9999,
            "zlib": True,
            "complevel": 1,
        }
    },
)
