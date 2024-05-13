import shutil
import tempfile
from pathlib import Path

import click
import pandas as pd
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import wget

from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData

EXTRACTION_YEARS = list(range(1990, 1992))
URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/{year}.tar.gz"
)


def extract_ncei_climate_stations_main(output_dir: str | Path) -> None:
    cd_data = ClimateDownscaleData(output_dir)

    dfs = []
    for year in EXTRACTION_YEARS:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as f:
            url = URL_TEMPLATE.format(year=year)

            wget(url, f.name)

            with tempfile.TemporaryDirectory() as outdir:
                shutil.unpack_archive(f.name, outdir)
                dfs.append(
                    pd.concat([pd.read_csv(f) for f in Path(outdir).glob("*.csv")])
                )
    data = pd.concat(dfs)

    data.to_parquet(cd_data.ncei_climate_stations / "climate_stations.parquet")


@click.command()  # type: ignore[arg-type]
@with_output_directory(DEFAULT_ROOT)
def extract_ncei_climate_stations(output_dir: str) -> None:
    extract_ncei_climate_stations_main(output_dir)
