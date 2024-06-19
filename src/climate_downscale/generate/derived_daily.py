import itertools

import click
from rra_tools import jobmon

from climate_downscale import cli_options as clio
from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData
from climate_downscale.generate import utils

TRANSFORM_MAP = {
    "heat_index": utils.Transform(
        source_variables=["mean_temperature", "relative_humidity"],
        transform_funcs=[utils.heat_index],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    "humidex": utils.Transform(
        source_variables=["mean_temperature", "relative_humidity"],
        transform_funcs=[utils.humidex],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    "effective_temperature": utils.Transform(
        source_variables=["mean_temperature", "relative_humidity", "wind_speed"],
        transform_funcs=[utils.effective_temperature],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
}


def generate_derived_daily_main(
    output_dir: str,
    target_variable: str,
    scenario: str,
    year: str,
) -> None:
    cd_data = ClimateDownscaleData(output_dir)
    transform = TRANSFORM_MAP[target_variable]

    ds = transform(
        *[
            cd_data.load_daily_results(scenario, source_variable, year)
            for source_variable in transform.source_variables
        ]
    )
    cd_data.save_daily_results(
        ds,
        scenario=scenario,
        variable=target_variable,
        year=year,
        encoding_kwargs=transform.encoding_kwargs,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_target_variable(variable_names=list(TRANSFORM_MAP))
@clio.with_cmip6_experiment(allow_historical=True)
@clio.with_year(years=clio.VALID_HISTORY_YEARS + clio.VALID_FORECAST_YEARS)
def generate_derived_daily_task(
    output_dir: str,
    target_variable: str,
    cmip6_experiment: str,
    year: str,
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
    generate_derived_daily_main(output_dir, target_variable, cmip6_experiment, year)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_target_variable(variable_names=list(TRANSFORM_MAP), allow_all=True)
@clio.with_cmip6_experiment(allow_all=True, allow_historical=True)
@clio.with_queue()
@clio.with_overwrite()
def generate_derived_daily(
    output_dir: str,
    target_variable: str,
    cmip6_experiment: str,
    queue: str,
    overwrite: bool,  # noqa: FBT001
) -> None:
    cd_data = ClimateDownscaleData(output_dir)

    variables = (
        list(TRANSFORM_MAP.keys())
        if target_variable == clio.RUN_ALL
        else [target_variable]
    )
    experiments = (
        list(clio.VALID_CMIP6_EXPERIMENTS)
        if cmip6_experiment == clio.RUN_ALL
        else [cmip6_experiment]
    )

    vey = []
    complete = []
    for v, e in itertools.product(variables, experiments):
        year_list = (
            clio.VALID_HISTORY_YEARS if e == "historical" else clio.VALID_FORECAST_YEARS
        )
        for y in year_list:
            path = cd_data.annual_results_path(scenario=e, variable=v, year=y)
            if not path.exists() or overwrite:
                vey.append((v, e, y))
            else:
                complete.append((v, e, y))

    print(f"{len(complete)} tasks already done. {len(vey)} tasks to do.")
    if not vey:
        return

    jobmon.run_parallel(
        runner="cdtask",
        task_name="generate derived_daily",
        flat_node_args=(
            ("target-variable", "cmip6-experiment", "year"),
            vey,
        ),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 2,
            "memory": "100G",
            "runtime": "120m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
    )
