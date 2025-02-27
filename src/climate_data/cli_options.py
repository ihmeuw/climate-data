"""
Climate Data CLI Options
------------------------

This module provides a set of CLI options for extracting climate data from the ERA5 and CMIP6 datasets.
These options are used to specify the data to extract, such as the year, month, variable, and dataset.
It also provides global variables representing the full space of valid values for these options.
"""

from collections.abc import Callable, Collection

import click
from rra_tools.cli_tools import (
    RUN_ALL,
    with_choice,
    with_debugger,
    with_input_directory,
    with_num_cores,
    with_output_directory,
    with_overwrite,
    with_progress_bar,
    with_queue,
    with_verbose,
)

from climate_data import constants as cdc


def with_year[**P, T](
    years: Collection[str],
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Create a CLI option for selecting a year."""
    return with_choice(
        "year",
        "y",
        allow_all=allow_all,
        choices=years,
        help="Year to extract data for.",
        convert=allow_all,
    )


def with_month[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "month",
        "m",
        allow_all=allow_all,
        choices=cdc.MONTHS,
        help="Month to extract data for.",
        convert=allow_all,
    )


def with_era5_variable[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "era5-variable",
        "x",
        allow_all=allow_all,
        choices=cdc.ERA5_VARIABLES,
        help="Variable to extract.",
        convert=allow_all,
    )


def with_era5_dataset[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "era5-dataset",
        "d",
        allow_all=allow_all,
        choices=cdc.ERA5_DATASETS,
        help="Dataset to extract.",
        convert=allow_all,
    )


def with_cmip6_source[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "cmip6-source",
        "s",
        allow_all=allow_all,
        choices=cdc.CMIP6_SOURCES,
        help="CMIP6 source to extract.",
        convert=allow_all,
    )


def with_cmip6_experiment[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "cmip6-experiment",
        "e",
        allow_all=allow_all,
        choices=cdc.CMIP6_EXPERIMENTS,
        help="CMIP6 experiment to extract.",
        convert=allow_all,
    )


def with_cmip6_variable[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "cmip6-variable",
        "x",
        allow_all=allow_all,
        choices=[v.name for v in cdc.CMIP6_VARIABLES],
        help="CMIP6 variable to extract.",
        convert=allow_all,
    )


def with_target_variable[**P, T](
    variable_names: Collection[str],
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "target-variable",
        "t",
        allow_all=allow_all,
        choices=variable_names,
        help="Variable to generate.",
        convert=allow_all,
    )


def with_draw[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "draw",
        allow_all=allow_all,
        choices=cdc.DRAWS,
        help="Draw to process.",
        convert=allow_all,
    )


def with_scenario[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "scenario",
        allow_all=allow_all,
        choices=cdc.SCENARIOS,
        help="Scenario to process.",
        convert=allow_all,
    )


def with_gcm_member[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    return click.option(
        "--gcm-member",
        "-g",
        type=click.STRING,
        help="GCM member to process.",
    )


__all__ = [
    "RUN_ALL",
    "with_choice",
    "with_cmip6_experiment",
    "with_cmip6_source",
    "with_debugger",
    "with_draw",
    "with_era5_dataset",
    "with_era5_variable",
    "with_gcm_member",
    "with_input_directory",
    "with_month",
    "with_num_cores",
    "with_output_directory",
    "with_overwrite",
    "with_progress_bar",
    "with_queue",
    "with_scenario",
    "with_target_variable",
    "with_verbose",
    "with_year",
]
