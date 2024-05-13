from pathlib import Path
from typing import ParamSpec, TypeVar

import cdsapi
import click
from rra_tools import jobmon
from rra_tools.cli_tools import (
    RUN_ALL,
    ClickOption,
    with_choice,
    with_output_directory,
    with_queue,
)

from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData

VALID_YEARS = [str(y) for y in range(1990, 2024)]
VALID_MONTHS = [f"{i:02d}" for i in range(1, 13)]
VALID_VARIABLES = [
    "total_precipitation",
    "2m_temperature",
]

_T = TypeVar("_T")
_P = ParamSpec("_P")


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


def with_variable(
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "variable",
        "x",
        allow_all=allow_all,
        choices=VALID_VARIABLES,
        help="Variable to extract.",
    )


def extract_era5_main(
    output_dir: str | Path,
    year: int | str,
    month: str,
    variable: str,
) -> None:
    cddata = ClimateDownscaleData(output_dir)
    cred_path = cddata.credentials_root / "copernicus.txt"
    url, key = cred_path.read_text().strip().split("\n")

    copernicus = cdsapi.Client(url=url, key=key)
    kwargs = {
        "dataset": "reanalysis-era5-land",
        "product_type": "reanalysis",
        "statistic": "daily_mean",
        "variable": "total_precipitation",
        "year": "2020",
        "month": "01",
        "time_zone": "UTC+00:00",
        "frequency": "1-hourly",
        "grid": "0.1/0.1",
        "area": {"lat": [-90, 90], "lon": [-180, 180]},
    }
    result = copernicus.service(
        "tool.toolbox.orchestrator.workflow",
        params={
            "realm": "user-apps",
            "project": "app-c3s-daily-era5-statistics",
            "version": "master",
            "kwargs": kwargs,
            "workflow_name": "application",
        },
    )

    out_path = cddata.era5 / f"{variable}_{year}_{month}.nc"
    copernicus.download(result, [out_path])


@click.command()  # type: ignore[arg-type]
@with_output_directory(DEFAULT_ROOT)
@with_year()
@with_month()
@with_variable()
def extract_era5_task(year: str, month: str, variable: str) -> None:
    extract_era5_main(DEFAULT_ROOT, year, month, variable)


@click.command()  # type: ignore[arg-type]
@with_output_directory(DEFAULT_ROOT)
@with_year(allow_all=True)
@with_variable(allow_all=True)
@with_queue()
def extract_era5(
    output_dir: str,
    year: str,
    variable: str,
    queue: str,
) -> None:
    years = VALID_YEARS if year == RUN_ALL else [year]
    variables = VALID_VARIABLES if variable == RUN_ALL else [variable]

    jobmon.run_parallel(
        task_name="extract_era5",
        node_args={
            "output-dir": [output_dir],
            "year": years,
            "variable": variables,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        runner="cdtask",
    )
