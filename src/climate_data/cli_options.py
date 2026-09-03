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
    convert_choice,
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
    choices: Collection[str] = cdc.SCENARIOS,
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "scenario",
        allow_all=allow_all,
        choices=choices,
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


def with_agg_version[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Add aggregation version option to a command."""
    return click.option(
        "--agg-version",
        help="Aggregation version to process.",
        required=True,
    )


def with_block_key[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Add block key option to a command."""
    return with_choice(
        "block-key",
        allow_all=allow_all,
        choices=None,  # Will be populated at runtime
        help="Block key to process.",
    )


def with_hierarchy[**P, T](
    choices: Collection[str] = cdc.HIERARCHY_MAP,
    *,
    allow_all: bool = False,
    default: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Add hierarchy option to a command.

    Pass ``default`` to give the (single-value) option a default; this builds
    the option directly since ``with_choice`` always supplies its own default.
    """
    if default is not None:
        return click.option(
            "--hierarchy",
            type=click.Choice(list(choices)),
            default=default,
            show_default=True,
            help="Hierarchy to process.",
        )
    return with_choice(
        "hierarchy",
        allow_all=allow_all,
        choices=choices,
        help="Hierarchy to process.",
        convert=allow_all,
    )


def with_agg_measure[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Add aggregation measure option to a command."""
    return with_choice(
        "agg-measure",
        allow_all=allow_all,
        choices=cdc.AGGREGATION_MEASURES,
        help="Climate measure to process.",
    )


def with_agg_scenario[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Add aggregation scenario option to a command."""
    return with_choice(
        "agg-scenario",
        allow_all=allow_all,
        choices=cdc.AGGREGATION_SCENARIOS,
        help="Climate scenario to process.",
    )


def with_location_id[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Add location ID option to a command."""
    return click.option(
        "--location-id",
        "-l",
        type=click.INT,
        help="Location ID to process.",
    )


def with_dry_run[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Add dry-run flag to a command."""
    return click.option(
        "--dry-run/--no-dry-run",
        "-n",
        default=False,
        show_default=True,
        help="Print sbatch-like previews instead of submitting jobs.",
    )


def with_concurrency_limit[**P, T](
    *,
    default: int | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Add the jobmon concurrency-limit option to a command.

    Pass ``default`` to throttle a runner out of the box; leaving it unset means
    the option defaults to ``None``, which ``run_parallel_maybe_dry_run`` drops
    so jobmon applies its own default (10000, effectively unthrottled).
    """
    return click.option(
        "--concurrency-limit",
        type=click.INT,
        default=default,
        show_default=True,
        help=(
            "Cap on tasks running at once, to keep write latency on shared "
            "storage manageable. jobmon's own default is 10000, which is "
            "effectively unthrottled."
        ),
    )


def with_run_mode[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Add the run-mode option selecting the storage-root profile."""
    return click.option(
        "--run-mode",
        type=click.Choice(list(cdc.RUN_MODES)),
        default="forecast",
        show_default=True,
        help=(
            "Storage-root profile: 'forecast' (production roots) or 'historical' "
            "(geospatial roots for the GBD historical exposure product)."
        ),
    )


def resolve_run_mode_root(
    param_name: str, value: str, run_mode: str, *, aggregate: bool = False
) -> str:
    """Resolve a directory option to the run-mode root unless the user overrode it.

    ``--run-mode`` selects the storage-root profile; an explicitly-passed
    ``--<param>`` still wins (detected via click's parameter source).
    """
    ctx = click.get_current_context(silent=True)
    overridden = (
        ctx is not None
        and ctx.get_parameter_source(param_name)
        is not click.core.ParameterSource.DEFAULT
    )
    if overridden:
        return value
    root = cdc.aggregate_root(run_mode) if aggregate else cdc.model_root(run_mode)
    return str(root)


__all__ = [
    "RUN_ALL",
    "convert_choice",
    "resolve_run_mode_root",
    "with_agg_measure",
    "with_agg_scenario",
    "with_agg_version",
    "with_block_key",
    "with_choice",
    "with_cmip6_experiment",
    "with_cmip6_source",
    "with_concurrency_limit",
    "with_debugger",
    "with_draw",
    "with_dry_run",
    "with_era5_dataset",
    "with_era5_variable",
    "with_gcm_member",
    "with_hierarchy",
    "with_input_directory",
    "with_location_id",
    "with_month",
    "with_num_cores",
    "with_output_directory",
    "with_overwrite",
    "with_progress_bar",
    "with_queue",
    "with_run_mode",
    "with_scenario",
    "with_target_variable",
    "with_verbose",
    "with_year",
]
