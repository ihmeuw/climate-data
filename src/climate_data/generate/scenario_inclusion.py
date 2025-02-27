import warnings
from pathlib import Path
from typing import Any

import click
import pandas as pd
import xarray as xr
from rra_tools import parallel

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData

warnings.filterwarnings("ignore")


def extract_metadata(data_path: Path) -> tuple[Any, ...]:
    meta = data_path.stem.split("_")
    try:
        ds = xr.open_dataset(data_path)
        year_start = ds["time.year"].min().item()
        year_end = ds["time.year"].max().item()
        duplicates = []
        for coord in ["lat", "lon", "time"]:
            if coord in ds.coords:
                duplicates.append(bool(pd.Index(ds[coord]).duplicated().any()))
            else:
                duplicates.append(False)
        can_load = True
    except (ValueError, RuntimeError):
        year_start, year_end, can_load, duplicates = (
            -1,
            -1,
            False,
            [False, False, False],
        )
    return *meta, year_start, year_end, can_load, *duplicates


def generate_scenario_inclusion_main(
    output_dir: str | Path, *, num_cores: int = 1, progress_bar: bool = False
) -> None:
    cdata = ClimateData(output_dir)
    paths = list(cdata.extracted_cmip6.glob("*.nc"))

    meta_list = parallel.run_parallel(
        extract_metadata,
        paths,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )
    columns = [
        "variable",
        "scenario",
        "source",
        "variant",
        "year_start",
        "year_end",
        "can_load",
        "lat_duplicates",
        "lon_duplicates",
        "time_duplicates",
    ]
    idx_columns = ["variable", "source", "variant", "scenario"]
    meta_df = pd.DataFrame(meta_list, columns=columns).set_index(idx_columns)
    meta_df["all_years"] = (meta_df.year_start <= 2020) & (meta_df.year_end >= 2099)  # noqa: PLR2004
    meta_df["no_duplicates"] = ~meta_df[
        ["lat_duplicates", "lon_duplicates", "time_duplicates"]
    ].any(axis=1)
    meta_df["valid"] = (
        meta_df["all_years"] & meta_df["can_load"] & meta_df["no_duplicates"]
    )
    inclusion_df = (
        meta_df["valid"]
        .unstack()
        .fillna(value=False)
        .sum(axis=1)
        .rename("valid_scenarios")
        .reset_index()
    )

    inclusion_df["include"] = inclusion_df.valid_scenarios == len(cdc.CMIP6_EXPERIMENTS)
    inclusion_df = (
        inclusion_df.loc[inclusion_df.include]
        .set_index(["source", "variant", "variable"])
        .include.unstack()
        .fillna(value=False)
    )
    cdata.save_scenario_metadata(meta_df)
    cdata.save_scenario_inclusion_metadata(inclusion_df)


@click.command()
@clio.with_output_directory(cdc.MODEL_ROOT)
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
