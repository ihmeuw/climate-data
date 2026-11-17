"""Compile stage for the CLIMATE-48 rerun: one member's person-days in production's layout.

Production's person-days (July 2026, ``compile_person_days``) were built with the latitude
defect of CLIMATE-48. To rebuild them with the fix, the history
(``history_person_days.py``) and forecast (``forecast_person_days.py``) stages run on the
GBD pixel hierarchy (``gbd_2023``) and write most-detailed person-days, one file per year.
This stage reads one member's years, sums them up the hierarchy, and writes one file per
subset hierarchy in ``cdc.HIERARCHY_MAP`` (``gbd_2023`` and ``fhs_2023``), as production's
compile does.

The files are laid out like July's, so their consumers can read them unchanged:

- years 1990-2100;
- keys (``location_id``, ``year_id``, ``temperature_zone``) as an int64 index;
- every one of the 60 temperature zones on every location-year, zeros included;
- the 800 bins as float64 columns labelled by floats (``-35.0`` ... ``44.9``).

Unlike ``compile_country_person_days`` (CLIMATE-45's US files), it keeps all-zero rows and
float labels, and has no GEOIDs or level names.
"""

import resource
import time
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
import pandas as pd

from climate_data import cli_options as clio
from climate_data import constants as cdc
from climate_data.data import PopulationModelData, save_parquet
from climate_data.special import utils
from climate_data.special.compile_country_person_days import person_days_year_path
from climate_data.special.history_person_days import (
    TEMPERATURE_COLUMNS,
    TEMPERATURE_ZONE_BINS,
)

# The years of the July 2026 production files.
PRODUCTION_YEARS = list(range(1990, 2101))
KEY_COLUMNS = ["location_id", "year_id", "temperature_zone"]
# The pixel hierarchies this compile can run on: those with subset views.
MEMBER_COMPILE_HIERARCHIES = ["gbd_2023"]


def compiled_member_path(
    root: str | Path,
    subset_hierarchy: str,
    scenario: str,
    gcm_member: str,
) -> Path:
    """``<root>/compiled/<subset_hierarchy>/<scenario>_<member>.parquet``."""
    return (
        Path(root) / "compiled" / subset_hierarchy / f"{scenario}_{gcm_member}.parquet"
    )


def _load_year(path: Path, year: int) -> pd.DataFrame:
    """One year's stage file, checked to hold that year and only that year."""
    df = pd.read_parquet(path).reset_index().rename(columns={"year": "year_id"})
    if df.empty:
        msg = f"The person-days file for {year} has no rows: {path}."
        raise ValueError(msg)
    found = sorted(set(df["year_id"].astype(int)))
    if found != [year]:
        msg = f"The person-days file for {year} holds years {found}: {path}."
        raise ValueError(msg)
    return df


def _all_zones(
    aggregated: pd.DataFrame, year: int, locations: npt.NDArray[np.int64]
) -> pd.DataFrame:
    """One year's rows on the full (location, zone) grid, absent rows as zeros.

    ``locations`` are every location of the hierarchy, so a location with no person-days
    in this year (no population, for example) still gets its zero rows.
    """
    full_index = pd.MultiIndex.from_product(
        [
            locations,
            np.array([year], dtype=np.int64),
            TEMPERATURE_ZONE_BINS.astype(np.int64),
        ],
        names=KEY_COLUMNS,
    )
    keyed = aggregated.astype(
        {"location_id": np.int64, "year_id": np.int64, "temperature_zone": np.int64}
    )
    keyed = keyed.set_index(KEY_COLUMNS)[list(TEMPERATURE_COLUMNS)]
    return keyed.reindex(full_index, fill_value=0.0)


def compile_member_person_days(
    hierarchy_df: pd.DataFrame,
    years: list[int],
    year_path: Callable[[int], Path],
) -> pd.DataFrame:
    """One member's person-days at every level of ``hierarchy_df``, for ``years``.

    Indexed by ``KEY_COLUMNS`` (int64), sorted; every zone in ``TEMPERATURE_ZONE_BINS`` on
    every location-year; the 800 bins as float64 columns labelled by ``TEMPERATURE_COLUMNS``.
    """
    if not years:
        msg = "No years to compile: years is empty."
        raise ValueError(msg)
    missing = []
    for year in years:
        if not year_path(year).exists():
            missing.append(year)
    if missing:
        msg = f"No person-days file for {len(missing)} years: {missing}."
        raise FileNotFoundError(msg)

    locations = np.sort(hierarchy_df["location_id"].unique()).astype(np.int64)
    frames = []
    for year in years:
        detailed = _load_year(year_path(year), year)
        aggregated = utils.aggregate_to_hierarchy(detailed, hierarchy_df)
        frames.append(_all_zones(aggregated, year, locations))
    result = pd.concat(frames).sort_index()
    result.columns = pd.Index(TEMPERATURE_COLUMNS, dtype=np.float64)
    return result.astype(np.float64)


def subset_to_hierarchy(
    compiled: pd.DataFrame, subset_hierarchy_df: pd.DataFrame
) -> pd.DataFrame:
    """The rows of ``compiled`` whose location is in the subset hierarchy."""
    keep = compiled.index.get_level_values("location_id").isin(
        subset_hierarchy_df["location_id"]
    )
    return compiled.loc[keep]


def compile_member_person_days_main(
    scenario: str,
    gcm_member: str,
    hierarchy: str,
    person_days_root: str | Path,
    population_model_root: str | Path,
    output_root: str | Path,
    *,
    years: list[int] | None = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compile and write one member's subset files. Returns the written ones by subset.

    Without ``overwrite``, writes only the subset files that don't exist yet, and returns an
    empty dict, writing nothing, when they all exist.
    """
    if years is not None and not years:
        msg = "No years to compile: years is empty."
        raise ValueError(msg)
    subsets = cdc.HIERARCHY_MAP[hierarchy]
    paths = {}
    for subset in subsets:
        path = compiled_member_path(output_root, subset, scenario, gcm_member)
        if overwrite or not path.exists():
            paths[subset] = path
    if not paths:
        print(f"All {len(subsets)} compiled files exist; skipping.")
        return {}

    pm_data = PopulationModelData(population_model_root)
    hierarchy_df = pm_data.load_subset_hierarchy(hierarchy)

    def year_path(year: int) -> Path:
        return person_days_year_path(
            person_days_root, hierarchy, scenario, gcm_member, year
        )

    compiled = compile_member_person_days(
        hierarchy_df, PRODUCTION_YEARS if years is None else years, year_path
    )
    written = {}
    for subset, path in paths.items():
        subset_df = subset_to_hierarchy(compiled, pm_data.load_subset_hierarchy(subset))
        path.parent.mkdir(parents=True, exist_ok=True)
        save_parquet(subset_df, path, atomic=True)
        written[subset] = subset_df
    return written


@click.command()
@clio.with_cmip6_experiment()
@clio.with_gcm_member()
@clio.with_hierarchy(choices=MEMBER_COMPILE_HIERARCHIES, default="gbd_2023")
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@click.option(
    "--person-days-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Root holding history/ and forecast/. Defaults to --output-dir.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Root for compiled/<subset hierarchy>/. No default, by design.",
)
@clio.with_overwrite()
def compile_member_person_days_task(
    cmip6_experiment: str,
    gcm_member: str,
    hierarchy: str,
    population_model_dir: str,
    person_days_dir: str | None,
    output_dir: str,
    *,
    overwrite: bool,
) -> None:
    """Compile one member-scenario's person-days for every subset hierarchy, 1990-2100."""
    start = time.time()
    written = compile_member_person_days_main(
        cmip6_experiment,
        gcm_member,
        hierarchy,
        person_days_dir or output_dir,
        population_model_dir,
        output_dir,
        overwrite=overwrite,
    )
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    for subset, df in written.items():
        total = float(np.sum(df.to_numpy()))
        n_locations = df.index.get_level_values("location_id").nunique()
        print(
            f"compile_member_person_days {subset} {cmip6_experiment} {gcm_member}: "
            f"{len(df)} rows, {n_locations} locations, {total:.6e} person-days"
        )
    print(f"{time.time() - start:.1f} s, peak RSS {peak_gb:.2f} GB")
