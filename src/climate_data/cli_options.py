"""
Climate Data CLI Options
------------------------

This module provides a set of CLI options for extracting climate data from the ERA5 and CMIP6 datasets.
These options are used to specify the data to extract, such as the year, month, variable, and dataset.
It also provides global variables representing the full space of valid values for these options.
"""

from typing import ParamSpec, TypeVar

import click
from rra_tools.cli_tools import (
    RUN_ALL,
    ClickOption,
    with_choice,
    with_debugger,
    with_input_directory,
    with_num_cores,
    with_output_directory,
    with_progress_bar,
    with_queue,
    with_verbose,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")


VALID_FULL_HISTORY_YEARS = [str(y) for y in range(1950, 2024)]
VALID_HISTORY_YEARS = [str(y) for y in range(1990, 2021)]
VALID_REFERENCE_YEARS = VALID_HISTORY_YEARS[-5:]
VALID_FORECAST_YEARS = [str(y) for y in range(2024, 2101)]


def with_year(
    *,
    years: list[str],
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    """Create a CLI option for selecting a year."""
    return with_choice(
        "year",
        "y",
        allow_all=allow_all,
        choices=years,
        help="Year to extract data for.",
    )


VALID_MONTHS = [f"{i:02d}" for i in range(1, 13)]


def with_month(
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "month",
        "m",
        allow_all=allow_all,
        choices=VALID_MONTHS,
        help="Month to extract data for.",
    )


VALID_ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "total_precipitation",
    "sea_surface_temperature",
]


def with_era5_variable(
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "era5-variable",
        "x",
        allow_all=allow_all,
        choices=VALID_ERA5_VARIABLES,
        help="Variable to extract.",
    )


VALID_ERA5_DATASETS = ["reanalysis-era5-land", "reanalysis-era5-single-levels"]


def with_era5_dataset(
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "era5-dataset",
        "d",
        allow_all=allow_all,
        choices=VALID_ERA5_DATASETS,
        help="Dataset to extract.",
    )


VALID_CMIP6_SOURCES = [
    "ACCESS-CM2",
    "AWI-CM-1-1-MR",
    "BCC-CSM2-MR",
    "CAMS-CSM1-0",
    "CESM2-WACCM",
    "CMCC-CM2-SR5",
    "CMCC-ESM2",
    "CNRM-CM6-1",
    "CNRM-CM6-1-HR",
    "CNRM-ESM2-1",
    "FGOALS-g3",
    "GFDL-ESM4",
    "GISS-E2-1-G",
    "IITM-ESM",
    "INM-CM4-8",
    "INM-CM5-0",
    "MIROC-ES2L",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NorESM2-MM",
]


def with_cmip6_source(
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "cmip6-source",
        "s",
        allow_all=allow_all,
        choices=VALID_CMIP6_SOURCES,
        help="CMIP6 source to extract.",
    )


VALID_CMIP6_EXPERIMENTS = [
    "ssp126",
    "ssp245",
    "ssp585",
]


def with_cmip6_experiment(
    *,
    allow_all: bool = False,
    allow_historical: bool = False,
) -> ClickOption[_P, _T]:
    choices = VALID_CMIP6_EXPERIMENTS[:]
    if allow_historical:
        choices.append("historical")
    return with_choice(
        "cmip6-experiment",
        "e",
        allow_all=allow_all,
        choices=choices,
        help="CMIP6 experiment to extract.",
    )


def with_target_variable(
    *,
    variable_names: list[str],
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "target-variable",
        "t",
        allow_all=allow_all,
        choices=variable_names,
        help="Variable to generate.",
    )


VALID_DRAWS = [str(d) for d in range(100)]


def with_draw(
    *,
    draws: list[str] = VALID_DRAWS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "draw",
        "d",
        allow_all=allow_all,
        choices=draws,
        help="Draw to extract data for.",
    )


STRIDE = 30
LATITUDES = [str(lat) for lat in range(-90, 90, STRIDE)]
LONGITUDES = [str(lon) for lon in range(-180, 180, STRIDE)]


def with_lat_start(
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "lat-start",
        allow_all=allow_all,
        choices=LATITUDES,
        help="Latitude of the top-left corner of the tile.",
    )


def with_lon_start(
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "lon-start",
        allow_all=allow_all,
        choices=LONGITUDES,
        help="Longitude of the top-left corner of the tile.",
    )


def with_overwrite() -> ClickOption[_P, _T]:
    return click.option(
        "--overwrite",
        is_flag=True,
        help="Overwrite existing files.",
    )


__all__ = [
    "LATITUDES",
    "LONGITUDES",
    "RUN_ALL",
    "STRIDE",
    "VALID_CMIP6_EXPERIMENTS",
    "VALID_CMIP6_SOURCES",
    "VALID_DRAWS",
    "VALID_ERA5_DATASETS",
    "VALID_ERA5_VARIABLES",
    "VALID_FORECAST_YEARS",
    "VALID_HISTORY_YEARS",
    "VALID_MONTHS",
    "VALID_REFERENCE_YEARS",
    "ClickOption",
    "with_choice",
    "with_cmip6_experiment",
    "with_cmip6_source",
    "with_debugger",
    "with_draw",
    "with_era5_dataset",
    "with_era5_variable",
    "with_input_directory",
    "with_lat_start",
    "with_lon_start",
    "with_month",
    "with_num_cores",
    "with_output_directory",
    "with_overwrite",
    "with_progress_bar",
    "with_queue",
    "with_verbose",
    "with_year",
]
