import shutil
from pathlib import Path

import click
import pandas as pd
from rra_tools import jobmon
from rra_tools.cli_tools import with_choice, with_output_directory, with_queue
from rra_tools.shell_tools import mkdir, wget

from climate_data.data import DEFAULT_ROOT, ClimateData

EXTRACTION_YEARS = [str(y) for y in range(1990, 2024)]
URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/{year}.tar.gz"
)


def extract_ncei_climate_stations_main(output_dir: str | Path, year: str) -> None:
    cd_data = ClimateData(output_dir)

    gz_path = cd_data.ncei_climate_stations / f"{year}.tar.gz"
    if gz_path.exists():
        gz_path.unlink()
    year_dir = cd_data.ncei_climate_stations / year
    mkdir(year_dir, exist_ok=True)

    url = URL_TEMPLATE.format(year=year)
    wget(url, str(gz_path))
    shutil.unpack_archive(str(gz_path), year_dir)

    data = pd.concat([pd.read_csv(f) for f in year_dir.glob("*.csv")])
    data["STATION"] = data["STATION"].astype(str)
    cd_data.save_ncei_climate_stations(data, year)

    gz_path.unlink()
    shutil.rmtree(year_dir)


@click.command()  # type: ignore[arg-type]
@with_choice(
    "year",
    "y",
    allow_all=False,
    choices=EXTRACTION_YEARS,
    help="Year to extract data for.",
)
@with_output_directory(DEFAULT_ROOT)
def extract_ncei_climate_stations_task(output_dir: str, year: str) -> None:
    extract_ncei_climate_stations_main(output_dir, year)


@click.command()  # type: ignore[arg-type]
@with_output_directory(DEFAULT_ROOT)
@with_queue()
def extract_ncei_climate_stations(output_dir: str, queue: str) -> None:
    jobmon.run_parallel(
        runner="cdtask",
        task_name="extract ncei",
        node_args={
            "year": EXTRACTION_YEARS,
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
