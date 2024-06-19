import itertools
import typing
from pathlib import Path

import click
import xarray as xr
from rra_tools import jobmon

from climate_downscale import cli_options as clio
from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData
from climate_downscale.generate import utils
from climate_downscale.generate.scenario_daily import VALID_YEARS

YEARS = {
    "historical": clio.VALID_YEARS,
    "scenario": VALID_YEARS,
}
TEMP_THRESHOLDS = list(range(20, 35))


class Transform:
    def __init__(
        self,
        source_variables: list[str],
        transform_funcs: list[typing.Callable[..., xr.Dataset]] = [utils.annual_mean],  # noqa: B006
        encoding_scale: float = 1.0,
        encoding_offset: float = 0.0,
    ):
        self.source_variables = source_variables
        self.transform_funcs = transform_funcs
        self.encoding_scale = encoding_scale
        self.encoding_offset = encoding_offset

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.source_variables)

    def __call__(self, *datasets: xr.Dataset) -> xr.Dataset:
        res = self.transform_funcs[0](*datasets)
        for transform_func in self.transform_funcs[1:]:
            res = transform_func(res)
        return res

    @property
    def encoding_kwargs(self) -> dict[str, float]:
        if self.encoding_offset != 0. or self.encoding_scale != 1:
            return {"add_offset": self.encoding_offset, "scale_factor": self.encoding_scale}
        return {}


TRANSFORM_MAP = {
    "mean_temperature": Transform(
        source_variables=["mean_temperature"],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    "mean_high_temperature": Transform(
        source_variables=["max_temperature"],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    "mean_low_temperature": Transform(
        source_variables=["min_temperature"],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    **{
        f"days_over_{temp}C": Transform(
            source_variables=["mean_temperature"],
            transform_funcs=[utils.count_threshold(temp), utils.annual_sum],
        )
        for temp in TEMP_THRESHOLDS
    },
    "mean_heat_index": Transform(
        source_variables=["mean_temperature", "relative_humidity"],
        transform_funcs=[utils.heat_index, utils.annual_mean],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    **{
        f"days_over_{temp}C_heat_index": Transform(
            source_variables=["mean_temperature", "relative_humidity"],
            transform_funcs=[
                utils.heat_index,
                utils.count_threshold(temp),
                utils.annual_sum,
            ],
        )
        for temp in TEMP_THRESHOLDS
    },
    "mean_humidex": Transform(
        source_variables=["mean_temperature", "relative_humidity"],
        transform_funcs=[utils.humidex, utils.annual_mean],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    **{
        f"days_over_{temp}C_humidex": Transform(
            source_variables=["mean_temperature", "relative_humidity"],
            transform_funcs=[
                utils.humidex,
                utils.count_threshold(temp),
                utils.annual_sum,
            ],
        )
        for temp in TEMP_THRESHOLDS
    },
    "mean_effective_temperature": Transform(
        source_variables=["mean_temperature", "relative_humidity", "wind_speed"],
        transform_funcs=[utils.effective_temperature, utils.annual_mean],
        encoding_scale=0.01,
        encoding_offset=273.15,
    ),
    **{
        f"days_over_{temp}C_effective_temperature": Transform(
            source_variables=["mean_temperature", "relative_humidity", "wind_speed"],
            transform_funcs=[
                utils.effective_temperature,
                utils.count_threshold(temp),
                utils.annual_sum,
            ],
        )
        for temp in TEMP_THRESHOLDS
    },
    "wind_speed": Transform(
        source_variables=["wind_speed"],
        encoding_scale=0.01,
    ),
    "relative_humidity": Transform(
        source_variables=["relative_humidity"],
        encoding_scale=0.01,
    ),
    "total_precipitation": Transform(
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
    output_dir: str | Path,
    target_variable: str,
    scenario: str,
) -> None:
    cd_data = ClimateDownscaleData(output_dir)

    transform = TRANSFORM_MAP[target_variable]

    
    variables = []
    for source_variable in transform:
        paths = []
        for scenario_label, year_list in YEARS.items():
            s = "historical" if scenario_label == "historical" else scenario
            for year in year_list:            
                paths.append(cd_data.daily_results_path(s, source_variable, year))
        variables.append(
            xr.open_mfdataset(
                paths, 
                parallel=True, 
                chunks={'date': -1, 'latitude': 601, 'longitude': 1200},
            )
        )
    ds = transform(*variables).compute()
    
    
    cd_data.save_annual_results(
        ds,
        scenario=scenario,
        variable=target_variable,
        encoding_kwargs=transform.encoding_kwargs,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@with_target_variable()
@clio.with_cmip6_experiment()
def generate_scenario_annual_task(
    output_dir: str,
    target_variable: str,
    cmip6_experiment: str,
) -> None:
    generate_scenario_annual_main(output_dir, target_variable, cmip6_experiment)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@with_target_variable(allow_all=True)
@clio.with_cmip6_experiment(allow_all=True)
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

    ve = []
    complete = []
    for v, e in itertools.product(variables, experiments):
        path = cd_data.annual_results_path(scenario=e, variable=v)
        if not path.exists() or overwrite:
            ve.append((v, e))
        else:
            complete.append((v, e))

    print(f"{len(complete)} tasks already done. {len(ve)} tasks to do.")
    if not ve:
        return

    jobmon.run_parallel(
        runner="cdtask",
        task_name="generate scenario_annual",
        flat_node_args=(
            ("target-variable", "cmip6-experiment"),
            ve,
        ),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 20,
            "memory": "250G",
            "runtime": "600m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
    )
