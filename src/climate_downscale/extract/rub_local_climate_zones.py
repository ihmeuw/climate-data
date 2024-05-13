from pathlib import Path

import click
from rra_tools.shell_tools import wget

from climate_downscale.data import ClimateDownscaleData

URL_TEMPLATE = "https://zenodo.org/records/8419340/files/{file_name}?download=1"
FILES = [
    "00_global_map_of_LCZs.png",
    "lcz_filter_v3.tif",
    "lcz_probability_v3.tif",
    "lcz_v3.tif",
    "readme.txt",
]


def extract_rub_local_climate_zones_main(output_dir: str | Path) -> None:
    data = ClimateDownscaleData(output_dir)
    out_root = data.rub_local_climate_zones

    for file_name in FILES:
        url = URL_TEMPLATE.format(file_name=file_name)
        wget(url, out_root / file_name)


@click.command()
def extract_rub_local_climate_zones() -> None:
    raise NotImplementedError
