import itertools
from pathlib import Path

import click
from dask.diagnostics.progress import ProgressBar
from rra_tools import jobmon

from climate_data import cli_options as clio
from climate_data.data import DEFAULT_ROOT, ClimateData
from climate_data.generate import utils
from climate_data.generate.scenario_daily import generate_scenario_daily_main

TEMP_THRESHOLDS = list(range(20, 35))
BETWEEN_TEMP_THRESHOLDS = [
    (15, 30),
    (20, 30),
    (15, 35),
    (20, 35),
]


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
        f"days_under_{upper}C_over_{lower}C": utils.Transform(
            source_variables=["mean_temperature"],
            transform_funcs=[
                utils.count_between_threshold(lower, upper),
                utils.annual_sum,
            ],
        )
        for lower, upper in BETWEEN_TEMP_THRESHOLDS
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
    "mean_heat_index": utils.Transform(
        source_variables=["heat_index"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    **{
        f"days_over_{temp}C_heat_index": utils.Transform(
            source_variables=["heat_index"],
            transform_funcs=[
                utils.count_threshold(temp),
                utils.annual_sum,
            ],
        )
        for temp in TEMP_THRESHOLDS
    },
    **{
        f"days_under_{upper}C_over_{lower}C_heat_index": utils.Transform(
            source_variables=["heat_index"],
            transform_funcs=[
                utils.count_between_threshold(lower, upper),
                utils.annual_sum,
            ],
        )
        for lower, upper in BETWEEN_TEMP_THRESHOLDS
    },
    "mean_humidex": utils.Transform(
        source_variables=["humidex"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    **{
        f"days_over_{temp}C_humidex": utils.Transform(
            source_variables=["humidex"],
            transform_funcs=[
                utils.count_threshold(temp),
                utils.annual_sum,
            ],
        )
        for temp in TEMP_THRESHOLDS
    },
    **{
        f"days_under_{upper}C_over_{lower}C_humidex": utils.Transform(
            source_variables=["humidex"],
            transform_funcs=[
                utils.count_between_threshold(lower, upper),
                utils.annual_sum,
            ],
        )
        for lower, upper in BETWEEN_TEMP_THRESHOLDS
    },
    "mean_effective_temperature": utils.Transform(
        source_variables=["effective_temperature"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    **{
        f"days_over_{temp}C_effective_temperature": utils.Transform(
            source_variables=["effective_temperature"],
            transform_funcs=[
                utils.count_threshold(temp),
                utils.annual_sum,
            ],
        )
        for temp in TEMP_THRESHOLDS
    },
    **{
        f"days_under_{upper}C_over_{lower}C_effective_temperature": utils.Transform(
            source_variables=["effective_temperature"],
            transform_funcs=[
                utils.count_between_threshold(lower, upper),
                utils.annual_sum,
            ],
        )
        for lower, upper in BETWEEN_TEMP_THRESHOLDS
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

# Remove all annual variables dependent on derived_daily variables
_excluded_source_variables = ["heat_index", "humidex", "effective_temperature"]

TRANSFORM_MAP = {
    k: v
    for k, v in TRANSFORM_MAP.items()
    if not any(
        source_variable in _excluded_source_variables
        for source_variable in v.source_variables
    )
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


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_target_variable(variable_names=list(TRANSFORM_MAP))
@clio.with_cmip6_experiment(allow_historical=True)
@clio.with_year(years=clio.VALID_HISTORY_YEARS + clio.VALID_FORECAST_YEARS)
@clio.with_draw(draws=clio.VALID_DRAWS, allow_all=False)
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

    generate_scenario_annual_main(
        output_dir, target_variable, cmip6_experiment, year, draw
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_target_variable(variable_names=list(TRANSFORM_MAP), allow_all=True)
@clio.with_cmip6_experiment(allow_all=True, allow_historical=True)
@clio.with_draw(draws=clio.VALID_DRAWS, allow_all=True)
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
    cd_data = ClimateData(output_dir)

    draws = clio.VALID_DRAWS if draw == clio.RUN_ALL else [draw]
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

    vey = []
    complete = []
    for d, v, e in itertools.product(draws, variables, experiments):
        year_list = (
            clio.VALID_HISTORY_YEARS if e == "historical" else clio.VALID_FORECAST_YEARS
        )
        for y in year_list:
            path = cd_data.annual_results_path(scenario=e, variable=v, year=y, draw=d)
            if not path.exists() or overwrite:
                vey.append((v, e, y, d))
            else:
                complete.append((v, e, y, d))

    print(f"{len(complete)} tasks already done. {len(vey)} tasks to do.")
    if not vey:
        return

    jobmon.run_parallel(
        runner="cdtask",
        task_name="generate scenario_annual",
        flat_node_args=(
            ("target-variable", "cmip6-experiment", "year", "draw"),
            vey,
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
