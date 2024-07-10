import itertools
from pathlib import Path

import click
import dask
import pandas as pd
import xarray as xr
from rra_tools import jobmon

from climate_downscale import cli_options as clio
from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData
from climate_downscale.generate import utils

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
    "mean_temperature": utils.Transform(
        source_variables=["2m_temperature"],
        transform_funcs=[utils.daily_mean],
        encoding_scale=0.01,
    ),
    "max_temperature": utils.Transform(
        source_variables=["2m_temperature"],
        transform_funcs=[utils.daily_max],
        encoding_scale=0.01,
    ),
    "min_temperature": utils.Transform(
        source_variables=["2m_temperature"],
        transform_funcs=[utils.daily_min],
        encoding_scale=0.01,
    ),
    "wind_speed": utils.Transform(
        source_variables=["10m_u_component_of_wind", "10m_v_component_of_wind"],
        transform_funcs=[utils.vector_magnitude, utils.daily_mean],
        encoding_scale=0.01,
    ),
    "relative_humidity": utils.Transform(
        source_variables=["2m_temperature", "2m_dewpoint_temperature"],
        transform_funcs=[utils.rh_percent, utils.daily_mean],
        encoding_scale=0.01,
    ),
    "total_precipitation": utils.Transform(
        source_variables=["total_precipitation"],
        transform_funcs=[utils.daily_sum],
        encoding_scale=0.1,
    ),
}


def load_and_shift_longitude(ds_path: str | Path) -> xr.Dataset:
    ds = xr.open_dataset(ds_path).chunk(time=24)
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):  # type: ignore[arg-type]
        ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180).sortby(
            "longitude"
        )
    return ds


def load_variable(
    cd_data: ClimateDownscaleData,
    variable: str,
    year: str,
    month: str,
    dataset: str = "single-levels",
) -> xr.Dataset:
    path = cd_data.extracted_era5_path(dataset, variable, year, month)
    if dataset == "land" and not path.exists():
        if variable != "total_sky_direct_solar_radiation_at_surface":
            # We only fallback for the one dataset, otherwise extraction failed.
            msg = f"Land dataset not found for {variable}. Extraction likely failed."
            raise ValueError(msg)
        # If the land dataset doesn't exist, fall back to the single-levels dataset
        path = cd_data.extracted_era5_path("single-levels", variable, year, month)
        ds = load_and_shift_longitude(path)
        # We expect this to already be in the correct grid, so interpolate.
        ds = utils.interpolate_to_target_latlon(ds)
    elif dataset == "land":
        ds = load_and_shift_longitude(path)
        # There are some slight numerical differences in the lat/long for some of
        # the land datasets. They are gridded consistently, so just tweak the
        # coordinates so things align.        
        ds = ds.assign_coords(latitude=utils.TARGET_LAT[::-1], longitude=utils.TARGET_LON)
    else:
        ds = load_and_shift_longitude(path)
    conversion = CONVERT_MAP[variable]
    ds = conversion(utils.rename_val_column(ds))
    return ds


def generate_historical_daily_main(
    output_dir: str | Path,
    year: str,
    target_variable: str,
) -> None:
    cd_data = ClimateDownscaleData(output_dir)

    transform = TRANSFORM_MAP[target_variable]
    datasets = []
    for month in range(1, 13):
        month_str = f"{month:02d}"
        print(f"loading single-levels for {month_str}")
        single_level = [
            load_variable(cd_data, sv, year, month_str, "single-levels")
            for sv in transform.source_variables
        ]
        print("collapsing")
        ds = transform(*single_level).compute()
        # collapsing often screws the date dtype, so fix it
        ds = ds.assign(date=pd.to_datetime(ds.date))

        print("interpolating")
        ds_land_res = utils.interpolate_to_target_latlon(ds)

        print(f"loading land for {month_str}")
        land = [
            load_variable(cd_data, sv, year, month_str, "land")
            for sv in transform.source_variables
        ]
        print("collapsing")
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):  # type: ignore[arg-type]
            ds_land = transform(*land).compute()
        ds_land = ds_land.assign(date=pd.to_datetime(ds_land.date))

        print("combining")
        combined = ds_land.combine_first(ds_land_res)
        datasets.append(combined)

    ds_year = xr.concat(datasets, dim="date").sortby("date")

    cd_data.save_daily_results(
        ds_year,
        scenario="historical",
        variable=target_variable,
        year=year,
        encoding_kwargs=transform.encoding_kwargs,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_year(years=clio.VALID_HISTORY_YEARS)
@clio.with_target_variable(variable_names=list(TRANSFORM_MAP))
def generate_historical_daily_task(
    output_dir: str,
    year: str,
    target_variable: str,
) -> None:
    generate_historical_daily_main(output_dir, year, target_variable)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_year(years=clio.VALID_HISTORY_YEARS, allow_all=True)
@clio.with_target_variable(variable_names=list(TRANSFORM_MAP), allow_all=True)
@clio.with_queue()
@clio.with_overwrite()
def generate_historical_daily(
    output_dir: str,
    year: str,
    target_variable: str,
    queue: str,
    overwrite: bool,  # noqa: FBT001
) -> None:
    cd_data = ClimateDownscaleData(output_dir)

    years = clio.VALID_HISTORY_YEARS if year == clio.RUN_ALL else [year]
    variables = (
        list(TRANSFORM_MAP.keys())
        if target_variable == clio.RUN_ALL
        else [target_variable]
    )
    years_and_variables = []
    complete = []
    for y, v in itertools.product(years, variables):
        path = cd_data.daily_results_path("historical", v, y)
        if not path.exists() or overwrite:
            years_and_variables.append((y, v))
        else:
            complete.append((y, v))

    print(
        f"{len(complete)} tasks already done. "
        f"Launching {len(years_and_variables)} tasks"
    )

    jobmon.run_parallel(
        runner="cdtask",
        task_name="generate historical_daily",
        flat_node_args=(
            ("year", "target-variable"),
            years_and_variables,
        ),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 5,
            "memory": "200G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
    )
