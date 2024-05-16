from typing import ParamSpec, TypeVar

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
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "year",
        "y",
        allow_all=allow_all,
        choices=VALID_YEARS,
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


VALID_CLIMATE_VARIABLES = [
    "total_precipitation",
    "2m_temperature",
]


def with_climate_variable(
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "climate-variable",
        "x",
        allow_all=allow_all,
        choices=VALID_CLIMATE_VARIABLES,
        help="Variable to extract.",
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


__all__ = [
    "VALID_YEARS",
    "VALID_MONTHS",
    "VALID_CLIMATE_VARIABLES",
    "STRIDE",
    "LATITUDES",
    "LONGITUDES",
    "with_year",
    "with_month",
    "with_climate_variable",
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
]
