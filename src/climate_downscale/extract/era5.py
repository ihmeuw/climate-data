from pathlib import Path

import cdsapi
import click
from rra_tools import jobmon
from rra_tools.shell_tools import touch

from climate_downscale import cli_options as clio
from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData


def extract_era5_main(
    output_dir: str | Path,
    era5_dataset: str,
    climate_variable: str,
    year: int | str,
    month: str,
) -> None:
    cddata = ClimateDownscaleData(output_dir)
    cred_path = cddata.credentials_root / "copernicus.txt"
    url, key = cred_path.read_text().strip().split("\n")

    copernicus = cdsapi.Client(url=url, key=key)
    kwargs = {
        "product_type": "reanalysis",
        "variable": climate_variable,
        "year": year,
        "month": month,
        "time": [f"{h:02d}:00" for h in range(24)],
        "format": "netcdf",
    }
    out_path = cddata.era5_path(era5_dataset, climate_variable, year, month)
    touch(out_path, exist_ok=True)

    copernicus.retrieve(
        era5_dataset,
        kwargs,
        out_path,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_era5_dataset()
@clio.with_climate_variable()
@clio.with_year()
@clio.with_month()
def extract_era5_task(
    output_dir: str,
    era5_dataset: str,
    climate_variable: str,
    year: str,
    month: str,
) -> None:
    extract_era5_main(
        output_dir,
        era5_dataset,
        climate_variable,
        year,
        month,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_era5_dataset(allow_all=True)
@clio.with_climate_variable(allow_all=True)
@clio.with_year(allow_all=True)
@clio.with_month(allow_all=True)
@clio.with_queue()
def extract_era5(  # noqa: PLR0913
    output_dir: str,
    era5_dataset: str,
    climate_variable: str,
    year: str,
    month: str,
    queue: str,
) -> None:
    datasets = (
        clio.VALID_ERA5_DATASETS if era5_dataset == clio.RUN_ALL else [era5_dataset]
    )
    variables = (
        clio.VALID_CLIMATE_VARIABLES
        if climate_variable == clio.RUN_ALL
        else [climate_variable]
    )
    years = clio.VALID_YEARS if year == clio.RUN_ALL else [year]
    months = clio.VALID_MONTHS if month == clio.RUN_ALL else [month]

    jobmon.run_parallel(
        runner="cdtask",
        task_name="extract_era5",
        node_args={
            "era5-dataset": datasets,
            "climate-variable": variables,
            "year": years,
            "month": months,
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
