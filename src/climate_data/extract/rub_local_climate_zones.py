from pathlib import Path

import click
from rra_tools.shell_tools import wget

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData

URL_TEMPLATE = "https://zenodo.org/records/8419340/files/{file_name}?download=1"
FILES = [
    "00_global_map_of_LCZs.png",
    "lcz_filter_v3.tif",
    "lcz_probability_v3.tif",
    "lcz_v3.tif",
    "readme.txt",
]


def extract_rub_local_climate_zones_main(output_dir: str | Path) -> None:
    data = ClimateData(output_dir)
    out_root = data.rub_local_climate_zones

    for file_name in FILES:
        print(f"Downloading {file_name}")
        url = URL_TEMPLATE.format(file_name=file_name)
        wget(url, out_root / file_name)


@click.command()
@clio.with_output_directory(cdc.MODEL_ROOT)
def extract_rub_local_climate_zones(output_dir: str) -> None:
    extract_rub_local_climate_zones_main(output_dir)
