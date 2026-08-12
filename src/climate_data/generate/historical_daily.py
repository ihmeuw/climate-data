import itertools
from pathlib import Path

import click
import dask
import numpy as np
import pandas as pd
import xarray as xr

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData
from climate_data.generate import utils
from climate_data.jobmon_utils import run_parallel_maybe_dry_run

# Map from source variable to a unit conversion function
CONVERT_MAP = {
    cdc.ERA5_VARIABLES.u_component_of_wind: utils.scale_wind_speed_height,
    cdc.ERA5_VARIABLES.v_component_of_wind: utils.scale_wind_speed_height,
    cdc.ERA5_VARIABLES.dewpoint_temperature: utils.kelvin_to_celsius,
    cdc.ERA5_VARIABLES.temperature: utils.kelvin_to_celsius,
    cdc.ERA5_VARIABLES.surface_pressure: utils.identity,
    cdc.ERA5_VARIABLES.total_precipitation: utils.meter_to_millimeter,
    cdc.ERA5_VARIABLES.sea_surface_temperature: utils.kelvin_to_celsius,
}

# Map from target variable to:
#  - a list of source variables
#  - a transformation function
#  - a tuple of offset and scale factors for the output for serialization
TRANSFORM_MAP = {
    "mean_temperature": utils.Transform(
        source_variables=[cdc.ERA5_VARIABLES.temperature],
        transform_funcs=[utils.daily_mean],
        encoding_scale=0.01,
    ),
    "max_temperature": utils.Transform(
        source_variables=[cdc.ERA5_VARIABLES.temperature],
        transform_funcs=[utils.daily_max],
        encoding_scale=0.01,
    ),
    "min_temperature": utils.Transform(
        source_variables=[cdc.ERA5_VARIABLES.temperature],
        transform_funcs=[utils.daily_min],
        encoding_scale=0.01,
    ),
    "sea_surface_temperature": utils.Transform(
        source_variables=[cdc.ERA5_VARIABLES.sea_surface_temperature],
        transform_funcs=[utils.daily_mean],
        encoding_scale=0.01,
    ),
    "wind_speed": utils.Transform(
        source_variables=[
            cdc.ERA5_VARIABLES.u_component_of_wind,
            cdc.ERA5_VARIABLES.v_component_of_wind,
        ],
        transform_funcs=[utils.vector_magnitude, utils.daily_mean],
        encoding_scale=0.01,
    ),
    "relative_humidity": utils.Transform(
        source_variables=[
            cdc.ERA5_VARIABLES.temperature,
            cdc.ERA5_VARIABLES.dewpoint_temperature,
        ],
        transform_funcs=[utils.rh_percent, utils.daily_mean],
        encoding_scale=0.01,
    ),
    "total_precipitation": utils.Transform(
        source_variables=[cdc.ERA5_VARIABLES.total_precipitation],
        # Precipitation timestamps are hour-ending interval labels, so both branches
        # collapse with the interval-aware resample rather than a calendar-day groupby.
        transform_funcs={
            # Cumulative since 00Z: the day's total is the window's closing sample.
            cdc.ERA5_DATASETS.reanalysis_era5_land: [utils.daily_accumulation_last],
            # Per-hour increments: they sum.
            cdc.ERA5_DATASETS.reanalysis_era5_single_levels: [
                utils.daily_accumulation_sum
            ],
        },
        encoding_scale=0.1,
    ),
}


def load_and_shift_longitude(ds_path: str | Path) -> xr.Dataset:
    ds = xr.open_dataset(ds_path)
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    ds = ds.chunk(time=24)
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):  # type: ignore[arg-type]
        ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180).sortby(
            "longitude"
        )
    return ds


def load_variable(
    cdata: ClimateData,
    variable: str,
    year: str,
    month: str,
    dataset: str = cdc.ERA5_DATASETS.reanalysis_era5_single_levels,
) -> xr.Dataset:
    path = cdata.extracted_era5_path(dataset, variable, year, month)
    if dataset == cdc.ERA5_DATASETS.reanalysis_era5_land:
        ds = load_and_shift_longitude(path)
        # There are some slight numerical differences in the lat/long for some of
        # the land datasets. They are gridded consistently, so just tweak the
        # coordinates so things align.
        ds = ds.assign_coords(
            latitude=cdc.ERA5_LAND_LATITUDE[::-1],
            longitude=cdc.ERA5_LAND_LONGITUDE,
        )
    else:
        ds = load_and_shift_longitude(path)
    conversion = CONVERT_MAP[variable]
    ds = conversion(utils.rename_val_column(ds))
    return ds


# Target variables collapsed on hour-ending accumulation windows. Their final day is
# closed by the next month's first sample, so these need a look-ahead load and a trim.
# Everything else collapses instantaneous values on calendar days and needs neither.
LOOKAHEAD_VARIABLES = frozenset({"total_precipitation"})


def _next_month(year: str, month: int) -> tuple[str, str]:
    """The (year, month) following the one given, rolling December into January."""
    if month == 12:  # noqa: PLR2004
        return str(int(year) + 1), "01"
    return year, f"{month + 1:02d}"


def load_variable_with_lookahead(
    cdata: ClimateData,
    variable: str,
    year: str,
    month: str,
    dataset: str,
) -> xr.Dataset:
    """Load a month's hourly data plus the one sample that closes its final day.

    Accumulation windows are stamped by their end, so the last day of a month is closed
    by the next month's first sample -- for December, by the next *year's* January. That
    single extra sample completes the final bin without opening a new one.

    Raises if the look-ahead file is absent rather than degrading to the incomplete
    23-hour window: ERA5 extracts stop at 2023, so regenerating 2023 needs a January
    2024 file that does not exist, and that should stop the run rather than quietly
    shorten one day. Checked before loading the target month so the run fails fast.
    """
    next_year, next_month = _next_month(year, int(month))
    lookahead_path = cdata.extracted_era5_path(dataset, variable, next_year, next_month)
    if not lookahead_path.exists():
        msg = (
            f"Cannot close {year}-{month} for {variable}: the accumulation window of its"
            f" final day ends in the following month, whose extract is missing at"
            f" {lookahead_path}. Extract {dataset} {variable} {next_year}_{next_month}"
            f" before generating {year}."
        )
        raise FileNotFoundError(msg)

    ds = load_variable(cdata, variable, year, month, dataset)
    lookahead = load_variable(cdata, variable, next_year, next_month, dataset).isel(
        time=[0]
    )
    return xr.concat([ds, lookahead], dim="time")


def trim_to_month(ds: xr.Dataset, year: str, month: int) -> xr.Dataset:
    """Drop collapsed bins that fall outside the target month.

    The interval-aware collapse labels its first bin with the *previous* month's last
    day, since that bin holds that day's closing sample. Left in place, every month
    contributes a duplicate date at its seam.
    """
    dates = pd.to_datetime(ds.date.to_index())
    keep = np.flatnonzero((dates.year == int(year)) & (dates.month == month))
    return ds.isel(date=keep)


def validate_output(ds: xr.Dataset, year: str) -> None:  # noqa: C901
    error_msg_parts = []

    attrs_to_check = (
        ("dims", "dimensions", {"date", "latitude", "longitude"}),
        ("coords", "coordinates", {"date", "latitude", "longitude"}),
        ("data_vars", "data variables", {"value"}),
    )
    for attr, name, expected in attrs_to_check:
        actual = set(getattr(ds, attr))
        missing = expected - actual
        extra = actual - expected
        if missing:
            error_msg_parts.append(f"Missing {name}: {missing}")
        if extra:
            error_msg_parts.append(f"Extra {name}: {extra}")

    if ds.date.min() != pd.Timestamp(f"{year}-01-01"):
        error_msg_parts.append(f"Unexpected start date: {ds.date.min()}")
    if ds.date.max() != pd.Timestamp(f"{year}-12-31"):
        error_msg_parts.append(f"Unexpected end date: {ds.date.max()}")
    num_days = 366 if int(year) % 4 == 0 else 365
    if ds.dims["date"] != num_days:
        error_msg_parts.append(f"Unexpected number of days: {ds.dims['date']}")

    if not (ds.latitude == cdc.TARGET_LATITUDE[::-1]).all().item():
        error_msg_parts.append("Unexpected latitude")
    if not (ds.longitude == cdc.TARGET_LONGITUDE).all().item():
        error_msg_parts.append("Unexpected longitude")

    if str(ds["value"].dtype) not in ["float32", "float64"]:
        error_msg_parts.append(f"Unexpected dtype: {ds['value'].dtype}")

    if ds["value"].isnull().any().item():  # noqa: PD003
        error_msg_parts.append("Unexpected NaNs")

    if error_msg_parts:
        errors = "\n".join(error_msg_parts)
        msg = f"{len(error_msg_parts)} errors in output for {year}:" + "\n" + errors
        raise ValueError(msg)


def generate_historical_daily_main(
    target_variable: str,
    year: str,
    output_dir: str | Path,
) -> None:
    cdata = ClimateData(output_dir)

    transform = TRANSFORM_MAP[target_variable]
    # Accumulated variables need the next month's first sample to close their final day,
    # and the resulting out-of-month bins at each end trimmed off afterwards.
    needs_lookahead = target_variable in LOOKAHEAD_VARIABLES
    loader = load_variable_with_lookahead if needs_lookahead else load_variable

    datasets = []
    for month in range(1, 13):
        month_str = f"{month:02d}"
        print(f"month {month_str}")
        print("    loading single-levels")
        single_level = [
            loader(
                cdata,
                sv,
                year,
                month_str,
                cdc.ERA5_DATASETS.reanalysis_era5_single_levels,
            )
            for sv in transform.source_variables
        ]
        print("    collapsing")
        ds_single_level = transform(
            *single_level, key=cdc.ERA5_DATASETS.reanalysis_era5_single_levels
        ).compute()
        # collapsing often screws the date dtype, so fix it
        ds_single_level = ds_single_level.assign(
            date=pd.to_datetime(ds_single_level.date)
        )
        if needs_lookahead:
            ds_single_level = trim_to_month(ds_single_level, year, month)

        if target_variable == cdc.ERA5_VARIABLES.sea_surface_temperature:
            print("    interpolating")
            ds_single_level = utils.interpolate_to_target_latlon(
                ds_single_level, method="nearest"
            )

            # sea surface temperature is only available in the single-level dataset
            datasets.append(ds_single_level)
        else:
            print("    loading land")
            land = [
                loader(
                    cdata, sv, year, month_str, cdc.ERA5_DATASETS.reanalysis_era5_land
                )
                for sv in transform.source_variables
            ]

            print("    collapsing")
            with dask.config.set(**{"array.slicing.split_large_chunks": False}):  # type: ignore[arg-type]
                ds_land = transform(
                    *land, key=cdc.ERA5_DATASETS.reanalysis_era5_land
                ).compute()
            ds_land = ds_land.assign(date=pd.to_datetime(ds_land.date))
            if needs_lookahead:
                ds_land = trim_to_month(ds_land, year, month)

            print("    interpolating single level")
            ds_single_level = utils.interpolate_to_target_latlon(
                ds_single_level,
                method="linear",
                target_lon=ds_land.longitude,
                target_lat=ds_land.latitude,
            )

            print("    combining")
            combined = ds_land.fillna(ds_single_level).combine_first(ds_single_level)

            print("    interpolating combined")
            combined = utils.interpolate_to_target_latlon(combined, method="linear")

            datasets.append(combined)

    ds_year = xr.concat(datasets, dim="date").sortby("date")
    if "number" in ds_year:
        ds_year = ds_year.drop_vars("number")

    validate_output(ds_year, year)

    cdata.save_daily_results(
        ds_year,
        scenario="historical",
        variable=target_variable,
        year=year,
        encoding_kwargs=transform.encoding_kwargs,
    )


@click.command()
@clio.with_target_variable(TRANSFORM_MAP)
@clio.with_year(cdc.HISTORY_YEARS)
@clio.with_output_directory(cdc.MODEL_ROOT)
def generate_historical_daily_task(
    target_variable: str,
    year: str,
    output_dir: str,
) -> None:
    generate_historical_daily_main(target_variable, year, output_dir)


@click.command()
@clio.with_target_variable(TRANSFORM_MAP, allow_all=True)
@clio.with_year(cdc.HISTORY_YEARS, allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_queue()
@clio.with_overwrite()
@clio.with_dry_run()
def generate_historical_daily(
    target_variable: list[str],
    year: list[str],
    output_dir: str,
    queue: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    cdata = ClimateData(output_dir)

    years_and_variables = []
    complete = []
    for v, y in itertools.product(target_variable, year):
        path = cdata.daily_results_path("historical", v, y)
        if not path.exists() or overwrite:
            years_and_variables.append((y, v))
        else:
            complete.append((y, v))

    print(
        f"{len(complete)} tasks already done. "
        f"Launching {len(years_and_variables)} tasks"
    )

    run_parallel_maybe_dry_run(
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
            "memory": "150G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        max_attempts=2,
        dry_run=dry_run,
    )
