import typing
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr
from rra_tools import jobmon

from climate_downscale import cli_options as clio
from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData
from climate_downscale.generate import utils

TARGET_LON = xr.DataArray(
    np.round(np.arange(-180.0, 180.0, 0.1, dtype="float32"), 1), dims="longitude"
)
TARGET_LAT = xr.DataArray(
    np.round(np.arange(90.0, -90.1, -0.1, dtype="float32"), 1), dims="latitude"
)

# Map from source variable to a unit conversion function
CONVERT_MAP = {
    "10m_u_component_of_wind": utils.scale_wind_speed_height,
    "10m_v_component_of_wind": utils.scale_wind_speed_height,
    "2m_dewpoint_temperature": utils.kelvin_to_celsius,
    "2m_temperature": utils.kelvin_to_celsius,
    "surface_net_solar_radiation": utils.identity,
    "surface_net_thermal_radiation": utils.identity,
    "surface_pressure": utils.identity,
    "surface_solar_radiation_downwards": utils.identity,
    "surface_thermal_radiation_downwards": utils.identity,
    "total_precipitation": utils.meter_to_millimeter,
    "total_sky_direct_solar_radiation_at_surface": utils.identity,
}

# Map from target variable to:
#  - a list of source variables
#  - a transformation function
#  - a tuple of offset and scale factors for the output for serialization
TRANSFORM_MAP = {
    "mean_temperature": (
        ["2m_temperature"],
        utils.daily_mean,
        (273.15, 0.01),
    ),
    "max_temperature": (
        ["2m_temperature"],
        utils.daily_max,
        (273.15, 0.01),
    ),
    "min_temperature": (
        ["2m_temperature"],
        utils.daily_min,
        (273.15, 0.01),
    ),
    "cooling_degree_days": (
        ["2m_temperature"],
        utils.cdd,
        (0, 0.01),
    ),
    "heating_degree_days": (
        ["2m_temperature"],
        utils.hdd,
        (0, 0.01),
    ),
    "wind_speed": (
        ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        lambda x, y: utils.daily_mean(utils.vector_magnitude(x, y)),
        (0, 0.01),
    ),
    "relative_humidity": (
        ["2m_temperature", "2m_dewpoint_temperature"],
        lambda x, y: utils.daily_mean(utils.rh_percent(x, y)),
        (0, 0.01),
    ),
    "total_precipitation": (
        ["total_precipitation"],
        utils.daily_sum,
        (0, 0.1),
    ),
}

UNTESTED_TRANSFORM_MAP = {
    "heat_index": (
        ["2m_temperature", "2m_dewpoint_temperature"],
        lambda x, y: utils.daily_mean(utils.heat_index(x, y)),
        (273.15, 0.01),
    ),
    "humidex": (
        ["2m_temperature", "2m_dewpoint_temperature"],
        lambda x, y: utils.daily_mean(utils.humidex(x, y)),
        (273.15, 0.01),
    ),
    "effective_temperature": (
        [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        lambda t2m, t2d, uas, vas: utils.daily_mean(
            utils.effective_temperature(t2m, t2d, uas, vas)
        ),
        (273.15, 0.01),
    ),
}


_P = typing.ParamSpec("_P")
_T = typing.TypeVar("_T")


def with_variable(
    *,
    allow_all: bool = False,
) -> clio.ClickOption[_P, _T]:
    return clio.with_choice(
        "target-variable",
        "t",
        allow_all=allow_all,
        choices=list(TRANSFORM_MAP.keys()),
        help="Variable to generate.",
    )


def load_and_shift_longitude(ds_path: str | Path) -> xr.Dataset:
    ds = xr.load_dataset(ds_path)
    ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180).sortby(
        "longitude"
    )
    return ds


def load_variable(
    variable: str,
    year: str,
    month: str,
    dataset: str = "single-levels",
) -> xr.Dataset:
    root = Path("/mnt/share/erf/climate_downscale/extracted_data/era5")
    p = root / f"reanalysis-era5-{dataset}_{variable}_{year}_{month}.nc"
    if dataset == "land" and not p.exists():
        # Substitute the single level dataset pre-interpolated at the target resolution.
        p = root / f"reanalysis-era5-single-levels_{variable}_{year}_{month}.nc"
        ds = utils.interpolate_to_target_latlon(
            load_and_shift_longitude(p),
            target_lat=TARGET_LAT,
            target_lon=TARGET_LON,
        )
    elif dataset == "land":
        ds = load_and_shift_longitude(p).assign_coords(
            latitude=TARGET_LAT, longitude=TARGET_LON
        )
    else:
        ds = load_and_shift_longitude(p)
    conversion = CONVERT_MAP[variable]
    ds = conversion(utils.rename_val_column(ds))
    return ds


def generate_era5_daily_main(
    output_dir: str | Path,
    year: str,
    target_variable: str,
) -> None:
    source_variables, collapse_fun, (e_offset, e_scale) = TRANSFORM_MAP[target_variable]

    datasets = []
    for month in range(1, 13):
        month_str = f"{month:02d}"
        print("loading single-levels")
        single_level = [
            load_variable(sv, year, month_str, "single-levels")
            for sv in source_variables
        ]
        print("collapsing")
        ds = collapse_fun(*single_level)  # type: ignore[operator]
        # collapsing often screws the date dtype, so fix it
        ds = ds.assign(date=pd.to_datetime(ds.date))

        print("interpolating")
        ds_land_res = utils.interpolate_to_target_latlon(ds, TARGET_LAT, TARGET_LON)

        print("loading land")
        land = [load_variable(sv, year, month_str, "land") for sv in source_variables]
        print("collapsing")
        ds_land = collapse_fun(*land)  # type: ignore[operator]
        ds_land = ds_land.assign(date=pd.to_datetime(ds_land.date))

        print("combining")
        combined = ds_land.combine_first(ds_land_res)
        datasets.append(combined)

    ds_year = xr.concat(datasets, dim="date").sortby("date")

    cd_data = ClimateDownscaleData(output_dir)
    cd_data.save_era5_daily(
        ds_year, target_variable, year, add_offset=e_offset, scale_factor=e_scale
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_year()
@with_variable()
def generate_era5_daily_task(
    output_dir: str,
    year: str,
    target_variable: str,
) -> None:
    generate_era5_daily_main(output_dir, year, target_variable)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_year(allow_all=True)
@with_variable(allow_all=True)
@clio.with_queue()
def generate_era5_daily(
    output_dir: str,
    year: str,
    target_variable: str,
    queue: str,
) -> None:
    years = clio.VALID_YEARS if year == clio.RUN_ALL else [year]
    variables = (
        list(TRANSFORM_MAP.keys())
        if target_variable == clio.RUN_ALL
        else [target_variable]
    )

    jobmon.run_parallel(
        runner="cdtask",
        task_name="generate era5_daily",
        node_args={
            "year": years,
            "variable": variables,
        },
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "120m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
    )
