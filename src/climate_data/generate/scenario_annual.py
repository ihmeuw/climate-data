import itertools
from pathlib import Path

import click
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from rra_tools import jobmon

from climate_data import cli_options as clio
from climate_data.data import DEFAULT_ROOT, ClimateData
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
    output_dir: str | Path,
    target_variable: str,
    scenario: str,
    year: str,
    draw: str,
    progress_bar: bool = False,
) -> None:
    cd_data = ClimateData(output_dir)
    transform = TRANSFORM_MAP[target_variable]

    print("Loading files")
    if scenario == "historical":
        ds = transform(
            *[
                xr.open_dataset(
                    cd_data.daily_results_path(scenario, source_variable, year)
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
    cd_data.save_annual_results(
        ds,
        scenario=scenario,
        variable=target_variable,
        year=year,
        draw=draw,
        encoding_kwargs=transform.encoding_kwargs,
    )

    if scenario == "historical":
        # Symlink all the other draws to the same file
        source = cd_data.annual_results_path(scenario, target_variable, year, "0")
        for d in clio.VALID_DRAWS:
            if d == "0":
                continue
            destination = cd_data.annual_results_path(
                scenario, target_variable, year, d
            )
            if destination.exists():
                destination.unlink()
            destination.symlink_to(source)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_target_variable(variable_names=list(TRANSFORM_MAP))
@clio.with_cmip6_experiment(allow_historical=True)
@clio.with_year(years=clio.VALID_FULL_HISTORY_YEARS + clio.VALID_FORECAST_YEARS)
@clio.with_draw(allow_all=False)
def generate_scenario_annual_task(
    output_dir: str,
    target_variable: str,
    cmip6_experiment: str,
    year: str,
    draw: str,
) -> None:
    if year in clio.VALID_HISTORY_YEARS and cmip6_experiment != "historical":
        msg = "Historical years must use the 'historical' experiment."
        raise ValueError(msg)
    if year in clio.VALID_FORECAST_YEARS and cmip6_experiment == "historical":
        msg = (
            f"Forecast years must use a future experiment: "
            f"{clio.VALID_CMIP6_EXPERIMENTS}."
        )
        raise ValueError(msg)

    if cmip6_experiment == "historical" and draw != "0":
        msg = "Historical years must use draw 0."
        raise ValueError(msg)

    generate_scenario_annual_main(
        output_dir, target_variable, cmip6_experiment, year, draw
    )


def build_arg_list(
    target_variable: str,
    cmip6_experiment: str,
    draw: str,
    output_dir: str,
    overwrite: bool,
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    cd_data = ClimateData(output_dir)

    variables = (
        list(TRANSFORM_MAP.keys())
        if target_variable == clio.RUN_ALL
        else [target_variable]
    )
    experiments = (
        [*clio.VALID_CMIP6_EXPERIMENTS, "historical"]
        if cmip6_experiment == clio.RUN_ALL
        else [cmip6_experiment]
    )
    draws = clio.VALID_DRAWS if draw == clio.RUN_ALL else [draw]

    to_run, complete = [], []
    trc, cc = 0, 0

    print_template = "{v:<30} {e:<12} {tra:>10} {ca:>10}"
    print(
        print_template.format(v="VARIABLE", e="EXPERIMENT", tra="TO_RUN", ca="COMPLETE")
    )

    for v, e in itertools.product(variables, experiments):
        if e == "historical":
            if v in ["wind_speed", "relative_humidity"]:
                years = clio.VALID_HISTORY_YEARS
            else:
                years = clio.VALID_FULL_HISTORY_YEARS
            run_draws = ["0"]
        else:
            years = clio.VALID_FORECAST_YEARS
            run_draws = draws

        for y, d in itertools.product(years, run_draws):
            path = cd_data.annual_results_path(scenario=e, variable=v, year=y, draw=d)
            if not path.exists():
                to_run.append((v, e, y, d))
            else:
                complete.append((v, e, y, d))

        tra, ca = len(to_run) - trc, len(complete) - cc
        trc, cc = len(to_run), len(complete)
        print(print_template.format(v=v, e=e, tra=tra, ca=ca))

    if overwrite:
        to_run += complete
        complete = []

    return to_run, complete


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_target_variable(variable_names=list(TRANSFORM_MAP), allow_all=True)
@clio.with_cmip6_experiment(allow_all=True, allow_historical=True)
@clio.with_draw(allow_all=True)
@clio.with_queue()
@clio.with_overwrite()
def generate_scenario_annual(
    output_dir: str,
    target_variable: str,
    cmip6_experiment: str,
    draw: str,
    queue: str,
    overwrite: bool,
) -> None:
    to_run, complete = build_arg_list(
        target_variable,
        cmip6_experiment,
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
            ("target-variable", "cmip6-experiment", "year", "draw"),
            to_run,
        ),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "100G",
            "runtime": "120m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
    )
