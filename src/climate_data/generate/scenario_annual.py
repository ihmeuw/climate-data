import itertools
from pathlib import Path

import click
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from rra_tools import jobmon

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData
from climate_data.generate import utils
from climate_data.generate.scenario_daily import generate_scenario_daily_main

TEMP_THRESHOLDS = [30]


TRANSFORM_MAP = {
    "mean_temperature": utils.Transform(
        source_variables=["mean_temperature"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    "mean_high_temperature": utils.Transform(
        source_variables=["max_temperature"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    "mean_low_temperature": utils.Transform(
        source_variables=["min_temperature"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    **{
        f"days_over_{temp}C": utils.Transform(
            source_variables=["mean_temperature"],
            transform_funcs=[utils.count_threshold(temp), utils.annual_sum],
        )
        for temp in TEMP_THRESHOLDS
    },
    **{
        f"{disease}_suitability": utils.Transform(
            source_variables=["mean_temperature"],
            transform_funcs=[
                utils.map_suitability(disease),
                utils.annual_sum,
            ],
        )
        for disease in ["malaria", "dengue"]
    },
    "wind_speed": utils.Transform(
        source_variables=["wind_speed"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    "relative_humidity": utils.Transform(
        source_variables=["relative_humidity"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    "total_precipitation": utils.Transform(
        source_variables=["total_precipitation"],
        transform_funcs=[utils.annual_sum],
        encoding_scale=10,
    ),
    "precipitation_days": utils.Transform(
        source_variables=["total_precipitation"],
        transform_funcs=[utils.count_threshold(0.1), utils.annual_sum],
    ),
}

# Notes about what to do:
# We want to leave the interface for this function/entry point essentially the same.  We'll add in
# a `draw` argument to the task function, but otherwise we'll keep the same interface.
# The idea here is to take a target variable in annual space, get all the source variables,
# compute the daily source variables in memory, then collapse them to the annual target variable.


def generate_scenario_annual_main(
    target_variable: str,
    scenario: str,
    year: str,
    draw: str,
    output_dir: str | Path,
    progress_bar: bool = False,
) -> None:
    cdata = ClimateData(output_dir)
    transform = TRANSFORM_MAP[target_variable]

    print("Loading files")
    if scenario == "historical":
        ds = transform(
            *[
                xr.open_dataset(
                    cdata.daily_results_path(scenario, source_variable, year)
                )
                for source_variable in transform.source_variables
            ]
        )
    else:
        ds = transform(
            *[
                generate_scenario_daily_main(
                    output_dir=output_dir,
                    year=year,
                    draw=draw,
                    target_variable=source_variable,
                    cmip6_experiment=scenario,
                    write_output=False,
                )
                for source_variable in transform.source_variables
            ]
        )
    if progress_bar:
        with ProgressBar():  # type: ignore[no-untyped-call]
            ds = ds.compute()
    else:
        ds = ds.compute()

    print("Saving files")
    cdata.save_annual_results(
        ds,
        scenario=scenario,
        variable=target_variable,
        year=year,
        draw=draw,
        encoding_kwargs=transform.encoding_kwargs,
    )

    if scenario == "historical":
        # Symlink all the other draws to the same file
        source = cdata.annual_results_path(scenario, target_variable, year, "0")
        for d in clio.VALID_DRAWS:
            if d == "0":
                continue
            destination = cdata.annual_results_path(scenario, target_variable, year, d)
            if destination.exists():
                destination.unlink()
            destination.symlink_to(source)


@click.command()  # type: ignore[arg-type]
@clio.with_target_variable(list(TRANSFORM_MAP))
@clio.with_scenario()
@clio.with_year(cdc.FULL_HISTORY_YEARS + cdc.FORECAST_YEARS)
@clio.with_draw()
@clio.with_output_directory(cdc.MODEL_ROOT)
def generate_scenario_annual_task(
    target_variable: str,
    scenario: str,
    year: str,
    draw: str,
    output_dir: str,
) -> None:
    if year in cdc.HISTORY_YEARS and scenario != "historical":
        msg = "Historical years must use the 'historical' experiment."
        raise ValueError(msg)
    if year in cdc.FORECAST_YEARS and scenario == "historical":
        msg = (
            f"Forecast years must use a future experiment: " f"{cdc.CMIP6_EXPERIMENTS}."
        )
        raise ValueError(msg)

    if scenario == "historical" and draw != "0":
        msg = "Historical years must use draw 0."
        raise ValueError(msg)

    generate_scenario_annual_main(output_dir, target_variable, scenario, year, draw)


def build_arg_list(
    target_variable: str,
    scenario: str,
    draw: str,
    output_dir: str,
    overwrite: bool,
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    cdata = ClimateData(output_dir)

    variables = (
        list(TRANSFORM_MAP.keys())
        if target_variable == clio.RUN_ALL
        else [target_variable]
    )
    scenarios = cdc.SCENARIOS if scenario == clio.RUN_ALL else [scenario]
    draws = cdc.DRAWS if draw == clio.RUN_ALL else [draw]

    to_run, complete = [], []
    trc, cc = 0, 0

    print_template = "{v:<30} {e:<12} {tra:>10} {ca:>10}"
    print(
        print_template.format(v="VARIABLE", e="EXPERIMENT", tra="TO_RUN", ca="COMPLETE")
    )

    for v, s in itertools.product(variables, scenarios):
        if s == "historical":
            years = cdc.FULL_HISTORY_YEARS
            run_draws = ["0"]
        else:
            years = cdc.FORECAST_YEARS
            run_draws = draws

        for y, d in itertools.product(years, run_draws):
            path = cdata.annual_results_path(scenario=s, variable=v, year=y, draw=d)
            if not path.exists():
                to_run.append((v, s, y, d))
            else:
                complete.append((v, s, y, d))

        tra, ca = len(to_run) - trc, len(complete) - cc
        trc, cc = len(to_run), len(complete)
        print(print_template.format(v=v, e=s, tra=tra, ca=ca))

    if overwrite:
        to_run += complete
        complete = []

    return to_run, complete


@click.command()  # type: ignore[arg-type]
@clio.with_target_variable(list(TRANSFORM_MAP), allow_all=True)
@clio.with_scenario(allow_all=True)
@clio.with_draw(allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_queue()
@clio.with_overwrite()
def generate_scenario_annual(
    output_dir: str,
    target_variable: str,
    scenario: str,
    draw: str,
    queue: str,
    overwrite: bool,
) -> None:
    to_run, complete = build_arg_list(
        target_variable,
        scenario,
        draw,
        output_dir,
        overwrite,
    )

    print(f"{len(complete)} tasks already done. {len(to_run)} tasks to do.")

    if not to_run:
        return

    jobmon.run_parallel(
        runner="cdtask",
        task_name="generate scenario_annual",
        flat_node_args=(
            ("target-variable", "scenario", "year", "draw"),
            to_run,
        ),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "200G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
    )
