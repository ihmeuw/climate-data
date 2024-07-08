import warnings
from pathlib import Path
from typing import Any

import click
import pandas as pd
import xarray as xr
from rra_tools import parallel

from climate_downscale import cli_options as clio
from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData

warnings.filterwarnings("ignore")


def extract_metadata(data_path: Path) -> tuple[Any]:
    meta = data_path.stem.split("_")
    ds = xr.open_dataset(data_path)
    year_start = ds["time.year"].min().item()
    year_end = ds["time.year"].max().item()
    return *meta, year_start, year_end, str(data_path)


def generate_scenario_inclusion_main(
    output_dir: str | Path, *, num_cores: int = 1, progress_bar: bool = False
) -> None:
    cd_data = ClimateDownscaleData(output_dir)
    paths = list(cd_data.extracted_cmip6.glob("*.nc"))

    meta_list = parallel.run_parallel(
        extract_metadata,
        paths,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )

    columns = ["variable", "scenario", "source", "variant", "year_start", "year_end"]
    meta_df = pd.DataFrame(meta_list, columns=columns)
    meta_df["all_years"] = (meta_df.year_start <= 2020) & (meta_df.year_end >= 2099)  # noqa: PLR2004
    meta_df["year_range"] = meta_df.apply(
        lambda r: f"{r.loc['year_start']}_{r.loc['year_end']}", axis=1
    )

    valid_scenarios = (
        meta_df.set_index(["variable", "source", "variant", "scenario"])
        .all_years.unstack()
        .fillna(value=False)
        .sum(axis=1)
        .rename("valid_scenarios")
    )
    year_range = (
        meta_df.set_index(["variable", "source", "variant", "scenario"])
        .year_range.unstack()
        .fillna("")
    )
    inclusion_df = pd.concat([year_range, valid_scenarios], axis=1).reset_index()
    inclusion_df["include"] = inclusion_df.valid_scenarios == 5  # noqa: PLR2004

    cd_data.save_scenario_metadata(meta_df)
    cd_data.save_scenario_inclusion_metadata(inclusion_df)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_num_cores(default=10)
@clio.with_progress_bar()
def generate_scenario_inclusion(
    output_dir: str,
    num_cores: int,
    progress_bar: bool,
) -> None:
    generate_scenario_inclusion_main(
        output_dir,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )
