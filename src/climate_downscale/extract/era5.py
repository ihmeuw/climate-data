from pathlib import Path

import cdsapi
import click
from rra_tools import jobmon
from rra_tools.shell_tools import touch

from climate_downscale import cli_options as clio
from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData


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

    out_path = cddata.era5_temperature_daily_mean / f"{variable}_{year}_{month}.nc"
    touch(out_path, exist_ok=True)
    copernicus.download(result, [out_path])


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_year()
@clio.with_month()
@clio.with_climate_variable()
def extract_era5_task(year: str, month: str, climate_variable: str) -> None:
    extract_era5_main(DEFAULT_ROOT, year, month, climate_variable)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_year(allow_all=True)
@clio.with_climate_variable(allow_all=True)
@clio.with_queue()
def extract_era5(
    output_dir: str,
    year: str,
    variable: str,
    queue: str,
) -> None:
    years = clio.VALID_YEARS if year == clio.RUN_ALL else [year]
    variables = clio.VALID_CLIMATE_VARIABLES if variable == clio.RUN_ALL else [variable]

    jobmon.run_parallel(
        runner="cdtask",
        task_name="extract_era5",
        node_args={
            "year": years,
            "variable": variables,
        },
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
    )
