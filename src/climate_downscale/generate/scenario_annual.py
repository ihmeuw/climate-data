import itertools
import typing
from pathlib import Path

import click
import xarray as xr
from rra_tools import jobmon

from climate_downscale import cli_options as clio
from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData
from climate_downscale.generate import utils

TEMP_THRESHOLDS = list(range(20, 35))


TRANSFORM_MAP = {
    "mean_temperature": utils.Transform(
        source_variables=["mean_temperature"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    "mean_high_temperature": utils.Transform(
        source_variables=["max_temperature"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    "mean_low_temperature": utils.Transform(
        source_variables=["min_temperature"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    **{
        f"days_over_{temp}C": utils.Transform(
            source_variables=["mean_temperature"],
            transform_funcs=[utils.count_threshold(temp), utils.annual_sum],
        )
        for temp in TEMP_THRESHOLDS
    },
    "mean_heat_index": utils.Transform(
        source_variables=["mean_temperature", "relative_humidity"],
        transform_funcs=[utils.heat_index, utils.annual_mean],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    **{
        f"days_over_{temp}C_heat_index": utils.Transform(
            source_variables=["mean_temperature", "relative_humidity"],
            transform_funcs=[
                utils.heat_index,
                utils.count_threshold(temp),
                utils.annual_sum,
            ],
        )
        for temp in TEMP_THRESHOLDS
    },
    "mean_humidex": utils.Transform(
        source_variables=["mean_temperature", "relative_humidity"],
        transform_funcs=[utils.humidex, utils.annual_mean],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    **{
        f"days_over_{temp}C_humidex": utils.Transform(
            source_variables=["mean_temperature", "relative_humidity"],
            transform_funcs=[
                utils.humidex,
                utils.count_threshold(temp),
                utils.annual_sum,
            ],
        )
        for temp in TEMP_THRESHOLDS
    },
    "mean_effective_temperature": utils.Transform(
        source_variables=["mean_temperature", "relative_humidity", "wind_speed"],
        transform_funcs=[utils.effective_temperature, utils.annual_mean],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    **{
        f"days_over_{temp}C_effective_temperature": utils.Transform(
            source_variables=["mean_temperature", "relative_humidity", "wind_speed"],
            transform_funcs=[
                utils.effective_temperature,
                utils.count_threshold(temp),
                utils.annual_sum,
            ],
        )
        for temp in TEMP_THRESHOLDS
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
        encoding_scale=0.1,
    ),
}


_P = typing.ParamSpec("_P")
_T = typing.TypeVar("_T")


def with_target_variable(
    *,
    allow_all: bool = False,
) -> clio.ClickOption[_P, _T]:
    return clio.with_choice(
        "target-variable",
        "t",
        allow_all=allow_all,
        choices=list(TRANSFORM_MAP.keys()),
        help="Variable to generate.",
    )


def generate_scenario_annual_main(
    output_dir: str | Path, target_variable: str, scenario: str, year: str
) -> None:
    cd_data = ClimateDownscaleData(output_dir)
    transform = TRANSFORM_MAP[target_variable]

    ds = transform(
        *[
            xr.open_dataset(cd_data.daily_results_path(scenario, source_variable, year))
            for source_variable in transform.source_variables
        ]
    )
    cd_data.save_annual_results(
        ds,
        scenario=scenario,
        variable=target_variable,
        year=year,
        encoding_kwargs=transform.encoding_kwargs,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@with_target_variable()
@clio.with_cmip6_experiment(allow_historical=True)
@clio.with_year(years=clio.VALID_HISTORY_YEARS + clio.VALID_FORECAST_YEARS)
def generate_scenario_annual_task(
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

    generate_scenario_annual_main(output_dir, target_variable, cmip6_experiment, year)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@with_target_variable(allow_all=True)
@clio.with_cmip6_experiment(allow_all=True, allow_historical=True)
@clio.with_queue()
@clio.with_overwrite()
def generate_scenario_annual(
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
        task_name="generate scenario_annual",
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
