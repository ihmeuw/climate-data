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
from climate_data.generate.scenario_daily import (
    TRANSFORM_MAP as DAILY_TRANSFORM_MAP,
)
from climate_data.generate.scenario_daily import (
    generate_scenario_daily_main,
)

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
    gcm_member: str,
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
                    gcm_member=gcm_member,
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
    cdata.save_raw_annual_results(
        ds,
        scenario=scenario,
        variable=target_variable,
        year=year,
        gcm_member=gcm_member,
        encoding_kwargs=transform.encoding_kwargs,
    )


@click.command()
@clio.with_target_variable(list(TRANSFORM_MAP))
@clio.with_scenario()
@clio.with_year(cdc.HISTORY_YEARS + cdc.FORECAST_YEARS)
@clio.with_gcm_member()
@clio.with_output_directory(cdc.MODEL_ROOT)
def generate_scenario_annual_task(
    target_variable: str,
    scenario: str,
    year: str,
    gcm_member: str,
    output_dir: str,
) -> None:
    history_flags = [
        year in cdc.HISTORY_YEARS,
        scenario == "historical",
        gcm_member == "era5",
    ]
    if any(history_flags) and not all(history_flags):
        msg = f"Historical years must use the 'historical' experiment and era5 GCM member. {year} {scenario} {gcm_member}"
        raise ValueError(msg)

    generate_scenario_annual_main(
        target_variable, scenario, year, gcm_member, output_dir, progress_bar=False
    )


def build_arg_list(
    target_variables: list[str],
    scenarios: list[str],
    output_dir: str,
    overwrite: bool,
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    cdata = ClimateData(output_dir)
    to_run, complete = [], []
    trc, cc = 0, 0

    print_template = "{v:<30} {e:<12} {tra:>10} {ca:>10}"
    print(
        print_template.format(v="VARIABLE", e="EXPERIMENT", tra="TO_RUN", ca="COMPLETE")
    )

    for v, s in itertools.product(target_variables, scenarios):
        if s == "historical":
            years = cdc.HISTORY_YEARS
            gcm_members = ["era5"]
        else:
            years = cdc.FORECAST_YEARS
            annual_source_variables = TRANSFORM_MAP[v].source_variables
            daily_source_variables = itertools.chain(
                *[
                    DAILY_TRANSFORM_MAP[source_variable][0].source_variables
                    for source_variable in annual_source_variables
                ]
            )
            gcm_members = cdata.get_gcms(list(daily_source_variables))

        for y, g in itertools.product(years, gcm_members):
            path = cdata.raw_annual_results_path(
                scenario=s, variable=v, year=y, gcm_member=g
            )
            if not path.exists():
                to_run.append((v, s, y, g))
            else:
                complete.append((v, s, y, g))

        tra, ca = len(to_run) - trc, len(complete) - cc
        trc, cc = len(to_run), len(complete)
        print(print_template.format(v=v, e=s, tra=tra, ca=ca))

    if overwrite:
        to_run += complete
        complete = []

    return to_run, complete


@click.command()
@clio.with_target_variable(TRANSFORM_MAP, allow_all=True)
@clio.with_scenario(allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_queue()
@clio.with_overwrite()
def generate_scenario_annual(
    target_variable: list[str],
    scenario: list[str],
    output_dir: str,
    queue: str,
    overwrite: bool,
) -> None:
    to_run, complete = build_arg_list(
        target_variable,
        scenario,
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
            ("target-variable", "scenario", "year", "gcm-member"),
            to_run,
        ),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "120G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
    )
