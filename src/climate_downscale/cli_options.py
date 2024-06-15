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


VALID_YEARS = [str(y) for y in range(1990, 2024)]


def with_year(
    *,
    years: list[str] = VALID_YEARS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
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
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_pressure",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "total_precipitation",
    "total_sky_direct_solar_radiation_at_surface",
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
    "CAMS-CSM1-0",
    "CanESM5",
    "CNRM-ESM2-1",
    "GFDL-ESM4",
    "GISS-E2-1-G",
    "MIROC-ES2L",
    "MIROC6",
    "MRI-ESM2-0",
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
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
]


def with_cmip6_experiment(
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "cmip6-experiment",
        "e",
        allow_all=allow_all,
        choices=VALID_CMIP6_EXPERIMENTS,
        help="CMIP6 experiment to extract.",
    )


VALID_CMIP6_VARIABLES = [
    "uas",
    "vas",
    "hurs",
    "tas",
    "pr",
]


def with_cmip6_variable(
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "cmip6-variable",
        "x",
        allow_all=allow_all,
        choices=VALID_CMIP6_VARIABLES,
        help="CMIP6 variable to extract.",
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
    "VALID_YEARS",
    "VALID_MONTHS",
    "VALID_ERA5_VARIABLES",
    "VALID_ERA5_DATASETS",
    "VALID_CMIP6_SOURCES",
    "VALID_CMIP6_EXPERIMENTS",
    "VALID_CMIP6_VARIABLES",
    "STRIDE",
    "LATITUDES",
    "LONGITUDES",
    "with_year",
    "with_month",
    "with_era5_variable",
    "with_era5_dataset",
    "with_cmip6_source",
    "with_cmip6_experiment",
    "with_cmip6_variable",
    "with_lat_start",
    "with_lon_start",
    "with_output_directory",
    "with_queue",
    "with_verbose",
    "with_debugger",
    "with_input_directory",
    "with_num_cores",
    "with_progress_bar",
    "RUN_ALL",
    "ClickOption",
    "with_choice",
    "with_overwrite",
]
