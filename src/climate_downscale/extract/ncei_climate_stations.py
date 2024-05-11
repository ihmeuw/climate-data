from rra_tools.shell_tools import wget
from rra_tools.cli_tools import (
    with_output_directory
)
import shutil
import tempfile
from pathlib import Path
import pandas as pd

import click

from climate_downscale.data import ClimateDownscaleData, DEFAULT_ROOT


EXTRACTION_YEARS = list(range(1990, 1992))
URL_TEMPLATE = 'https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/{year}.tar.gz'


def extract_ncei_climate_stations_main(output_dir: str | Path):
    cd_data = ClimateDownscaleData(output_dir)

    data = []
    for year in EXTRACTION_YEARS:
        with tempfile.NamedTemporaryFile(suffix='.tar.gz') as f:
            url = URL_TEMPLATE.format(year=year)

            wget(url, f.name)

            with tempfile.TemporaryDirectory() as outdir:
                shutil.unpack_archive(f.name, outdir)
                data.append(
                    pd.concat([pd.read_csv(f) for f in Path(outdir).glob('*.csv')])
                )
    data = pd.concat(data)

    data.to_parquet(cd_data.ncei_climate_stations / 'climate_stations.parquet')


@click.command()
@with_output_directory(DEFAULT_ROOT)
def extract_ncei_climate_stations(output_dir: str):
    extract_ncei_climate_stations_main(output_dir)
