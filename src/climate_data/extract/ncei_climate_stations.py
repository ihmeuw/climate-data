import shutil
from pathlib import Path

import click
import pandas as pd
from rra_tools import jobmon
from rra_tools.shell_tools import mkdir, wget

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData

URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/{year}.tar.gz"
)


def extract_ncei_climate_stations_main(year: int | str, output_dir: str | Path) -> None:
    cdata = ClimateData(output_dir)

    gz_path = cdata.ncei_climate_stations / f"{year}.tar.gz"
    if gz_path.exists():
        gz_path.unlink()
    year_dir = cdata.ncei_climate_stations / str(year)
    mkdir(year_dir, exist_ok=True)

    url = URL_TEMPLATE.format(year=year)
    wget(url, str(gz_path))
    shutil.unpack_archive(str(gz_path), year_dir)

    data = pd.concat([pd.read_csv(f) for f in year_dir.glob("*.csv")])
    data["STATION"] = data["STATION"].astype(str)
    cdata.save_ncei_climate_stations(data, year)

    gz_path.unlink()
    shutil.rmtree(year_dir)


@click.command()
@clio.with_year(years=cdc.HISTORY_YEARS)
@clio.with_output_directory(cdc.MODEL_ROOT)
def extract_ncei_climate_stations_task(year: str, output_dir: str) -> None:
    extract_ncei_climate_stations_main(year, output_dir)


@click.command()
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_queue()
def extract_ncei_climate_stations(output_dir: str, queue: str) -> None:
    jobmon.run_parallel(
        runner="cdtask",
        task_name="extract ncei",
        node_args={
            "year": cdc.HISTORY_YEARS,
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
