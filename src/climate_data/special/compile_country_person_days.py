"""Compile stage of the CLIMATE-45 person-days: one country, every level, 1950-2100.

The history (``history_person_days.py``) and forecast (``forecast_person_days.py``) stages
write admin2 person-days, one file per year, for every location in the world. This stage
reads one country's admin2 rows from each year, sums them up the hierarchy to the country,
and writes one file with the location's level and name on every row. Rows with no
person-days are dropped. For the US it adds Census GEOIDs (a zero-padded string, 2 digits
for a state and 5 for a county) from USHD's crosswalk; the national row has none.

The 800 bin columns are labelled by strings (``'-35.0'`` ... ``'44.9'``), as in the
Delaware pilot, both in memory and on disk. Parquet column names are strings, and with the
keys as ordinary columns pandas would write float labels as strings anyway; production's
files read back with float labels only because their keys are the index.

One country at a time keeps the memory to one country's rows: all 47,855 admin2 units over
151 years do not fit in one job.
"""

import resource
import time
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np
import pandas as pd

from climate_data import cli_options as clio
from climate_data import constants as cdc
from climate_data.data import PopulationModelData, save_parquet
from climate_data.special import utils
from climate_data.special.forecast_person_days import (
    FORECAST_PERSON_DAYS_YEARS,
    FORECAST_START_YEAR,
    forecast_person_days_path,
)
from climate_data.special.history_person_days import (
    HISTORY_PERSON_DAYS_YEARS,
    TEMPERATURE_COLUMNS,
    history_person_days_path,
)

COMPILE_YEARS = HISTORY_PERSON_DAYS_YEARS + FORECAST_PERSON_DAYS_YEARS
# The bin labels of the compiled file: the bins' left edges, to one decimal.
BIN_LABELS = []
for _edge in TEMPERATURE_COLUMNS:
    BIN_LABELS.append(f"{_edge:.1f}")
KEY_COLUMNS = ["location_id", "year_id", "temperature_zone"]

US_LOCATION_ID = 102
# USHD's FIPS <-> location_id crosswalk (dated 9 Feb 2023). Its location_ids are LSAE's.
US_GEOID_CROSSWALK = Path(
    "/home/j/Project/us_counties/locations/counties/locations.csv"
)
# Level labels by depth below the country.
US_LEVEL_NAMES = {0: "national", 1: "state", 2: "county"}
US_GEOID_WIDTHS = {1: 2, 2: 5}
# The hierarchies the compile can sum on. lsae_1578 differs from lsae_1285 only in one
# parent (Chile); both have the same admin2 units, so lsae_1285's weights serve both.
COMPILE_HIERARCHIES = ["lsae_1285", "lsae_1578"]


def person_days_year_path(
    root: str | Path,
    weights_hierarchy: str,
    scenario: str,
    gcm_member: str,
    year: int,
) -> Path:
    """The history file before 2024, the member's forecast file from 2024."""
    if year < FORECAST_START_YEAR:
        return history_person_days_path(root, weights_hierarchy, year)
    return forecast_person_days_path(
        root, weights_hierarchy, scenario, gcm_member, year
    )


def compiled_country_path(
    root: str | Path,
    hierarchy: str,
    country_id: int,
    scenario: str,
    gcm_member: str,
) -> Path:
    """``<root>/compiled/<hierarchy>/<country_id>/<scenario>_<member>.parquet``."""
    return (
        Path(root)
        / "compiled"
        / hierarchy
        / str(country_id)
        / f"{scenario}_{gcm_member}.parquet"
    )


def country_locations(hierarchy_df: pd.DataFrame, country_id: int) -> pd.DataFrame:
    """The country's row and every location under it, from ``path_to_top_parent``."""
    in_country = []
    for path in hierarchy_df["path_to_top_parent"].astype(str):
        in_country.append(str(country_id) in path.split(","))
    locations = hierarchy_df.loc[in_country]
    if country_id not in set(locations["location_id"]):
        msg = f"Location {country_id} is not in the hierarchy."
        raise ValueError(msg)
    return locations


def country_admin2_ids(hierarchy_df: pd.DataFrame, country_id: int) -> list[int]:
    """The country's most-detailed locations: the admin2 units the stages wrote."""
    locations = country_locations(hierarchy_df, country_id)
    ids = locations.loc[locations["most_detailed"] == 1, "location_id"]
    return sorted(int(i) for i in ids)


def level_names(locations: pd.DataFrame, country_id: int) -> pd.Series:
    """``location_id`` -> level label: national/state/county for the US, else admin0-2."""
    country_level = int(
        locations.loc[locations["location_id"] == country_id, "level"].iloc[0]
    )
    labels = []
    for level in locations["level"]:
        depth = int(level) - country_level
        if country_id == US_LOCATION_ID:
            labels.append(US_LEVEL_NAMES[depth])
        else:
            labels.append(f"admin{depth}")
    return pd.Series(labels, index=locations["location_id"].to_numpy())


def _load_year(path: Path, admin2_ids: list[int]) -> pd.DataFrame:
    df = pd.read_parquet(path, filters=[("location_id", "in", admin2_ids)])
    df = df.reset_index().rename(columns={"year": "year_id"})
    return df


def compile_country_person_days(
    hierarchy_df: pd.DataFrame,
    country_id: int,
    years: list[int],
    year_path: Callable[[int], Path],
    geoids: pd.Series | None = None,
) -> pd.DataFrame:
    """One country's person-days at every level, for ``years``.

    Columns: ``location_id``, [``geoid``,] ``level``, ``location_name``, ``year_id``,
    ``temperature_zone``, then the 800 bins labelled by ``BIN_LABELS``. One row per
    (location, year, zone) with person-days; sorted by those keys.
    """
    missing = []
    for year in years:
        if not year_path(year).exists():
            missing.append(year)
    if missing:
        msg = f"No person-days file for {len(missing)} years: {missing}."
        raise FileNotFoundError(msg)

    locations = country_locations(hierarchy_df, country_id)
    admin2_ids = country_admin2_ids(hierarchy_df, country_id)
    keep = set(locations["location_id"].tolist())
    bins = list(TEMPERATURE_COLUMNS)

    frames = []
    for year in years:
        admin2 = _load_year(year_path(year), admin2_ids)
        aggregated = utils.aggregate_to_hierarchy(admin2, locations)
        # The helper also sums the country into its parent; keep the country and below.
        aggregated = aggregated.loc[aggregated["location_id"].isin(keep)]
        frames.append(aggregated[KEY_COLUMNS + bins])
    result = pd.concat(frames, ignore_index=True)
    result = result.loc[result[bins].to_numpy().sum(axis=1) > 0]
    result = result.sort_values(KEY_COLUMNS).reset_index(drop=True)

    names = locations.set_index("location_id")["location_name"]
    labels = level_names(locations, country_id)
    result.insert(1, "level", labels.loc[result["location_id"]].to_numpy())
    result.insert(2, "location_name", names.loc[result["location_id"]].to_numpy())
    result = result.rename(columns=dict(zip(bins, BIN_LABELS, strict=True)))
    if geoids is not None:
        result = insert_geoid(result, geoids)
    return result


def _current_fips(crosswalk_path: str | Path) -> pd.Series:
    """The crosswalk's current rows with a FIPS: ``location_id`` -> FIPS (a float)."""
    crosswalk = pd.read_csv(crosswalk_path)
    crosswalk = crosswalk.loc[crosswalk["current"] == 1, ["location_id", "fips"]]
    fips = crosswalk.dropna(subset=["fips"]).set_index("location_id")["fips"]
    if fips.index.duplicated().any():
        dupes = sorted(fips.index[fips.index.duplicated()].unique().tolist())
        msg = f"Location ids with more than one current FIPS: {dupes}."
        raise ValueError(msg)
    return fips


def _check_counties_in_states(geoids: pd.Series, parents: pd.Series) -> None:
    """Fail unless every county's GEOID starts with its state's."""
    outside = []
    for location_id, geoid in zip(geoids.index, geoids.to_numpy(), strict=True):
        if len(geoid) == US_GEOID_WIDTHS[2]:
            state = int(parents.loc[location_id])
            if not geoid.startswith(geoids.loc[state]):
                outside.append(int(location_id))
    if outside:
        msg = f"Counties whose GEOID is not in their state's: {outside[:20]}."
        raise ValueError(msg)


def load_us_geoids(crosswalk_path: str | Path, hierarchy_df: pd.DataFrame) -> pd.Series:
    """``location_id`` -> Census GEOID for every US state and county in the hierarchy.

    Uses the crosswalk's current rows. Fails unless every state and county maps, no two
    share a GEOID, and every county's GEOID starts with its state's.
    """
    locations = country_locations(hierarchy_df, US_LOCATION_ID)
    country_level = int(
        locations.loc[locations["location_id"] == US_LOCATION_ID, "level"].iloc[0]
    )
    fips = _current_fips(crosswalk_path)

    geoids = {}
    unmapped = []
    for location_id, level in zip(
        locations["location_id"], locations["level"], strict=True
    ):
        depth = int(level) - country_level
        if depth == 0:
            continue
        if location_id not in fips.index:
            unmapped.append(int(location_id))
            continue
        width = US_GEOID_WIDTHS[depth]
        geoids[int(location_id)] = str(int(fips.loc[location_id])).zfill(width)
    if unmapped:
        msg = f"{len(unmapped)} US locations have no FIPS: {unmapped[:20]}."
        raise ValueError(msg)

    result = pd.Series(geoids, name="geoid").sort_index()
    if result.duplicated().any():
        msg = f"GEOIDs used twice: {sorted(set(result[result.duplicated()]))}."
        raise ValueError(msg)
    _check_counties_in_states(result, locations.set_index("location_id")["parent_id"])
    return result


def insert_geoid(compiled: pd.DataFrame, geoids: pd.Series) -> pd.DataFrame:
    """Add ``geoid`` after ``location_id``; null where the location has none (the nation)."""
    result = compiled.copy()
    values = result["location_id"].map(geoids).astype(object)
    values = values.where(values.notna(), None)
    result.insert(1, "geoid", values)
    return result


def compile_country_person_days_main(
    country_id: int,
    scenario: str,
    gcm_member: str,
    hierarchy: str,
    weights_hierarchy: str,
    person_days_root: str | Path,
    population_model_root: str | Path,
    output_path: Path,
    *,
    years: list[int] | None = None,
    crosswalk_path: str | Path = US_GEOID_CROSSWALK,
    overwrite: bool = False,
) -> pd.DataFrame | None:
    """Compile and write one country's file. Returns it, or None if it already exists."""
    if output_path.exists() and not overwrite:
        print(f"{output_path} exists; skipping.")
        return None
    hierarchy_df = PopulationModelData(population_model_root).load_subset_hierarchy(
        hierarchy
    )
    geoids = None
    if country_id == US_LOCATION_ID:
        geoids = load_us_geoids(crosswalk_path, hierarchy_df)

    def year_path(year: int) -> Path:
        return person_days_year_path(
            person_days_root, weights_hierarchy, scenario, gcm_member, year
        )

    result = compile_country_person_days(
        hierarchy_df, country_id, years or COMPILE_YEARS, year_path, geoids
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(result, output_path, atomic=True)
    return result


@click.command()
@clio.with_cmip6_experiment()
@clio.with_gcm_member()
@click.option(
    "--country-id",
    type=int,
    default=US_LOCATION_ID,
    show_default=True,
    help="The country (LSAE level 1 location_id) to compile.",
)
@clio.with_hierarchy(choices=COMPILE_HIERARCHIES, default="lsae_1578")
@click.option(
    "--weights-hierarchy",
    type=click.Choice(["lsae_1285"]),
    default="lsae_1285",
    show_default=True,
    help="The hierarchy the history and forecast stages ran on (names their directories).",
)
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
    help="Root for compiled/<hierarchy>/<country>/. No default, by design.",
)
@clio.with_overwrite()
def compile_country_person_days_task(
    cmip6_experiment: str,
    gcm_member: str,
    country_id: int,
    hierarchy: str,
    weights_hierarchy: str,
    population_model_dir: str,
    person_days_dir: str | None,
    output_dir: str,
    *,
    overwrite: bool,
) -> None:
    """Compile one country's person-days for one member-scenario, 1950-2100."""
    start = time.time()
    output_path = compiled_country_path(
        output_dir, hierarchy, country_id, cmip6_experiment, gcm_member
    )
    result = compile_country_person_days_main(
        country_id,
        cmip6_experiment,
        gcm_member,
        hierarchy,
        weights_hierarchy,
        person_days_dir or output_dir,
        population_model_dir,
        output_path,
        overwrite=overwrite,
    )
    if result is None:
        return
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    total = float(np.sum(result[BIN_LABELS].to_numpy()))
    print(
        f"compile_country_person_days {country_id} {cmip6_experiment} {gcm_member}: "
        f"{len(result)} rows, {result['location_id'].nunique()} locations, "
        f"{total:.6e} person-days, {time.time() - start:.1f} s, "
        f"peak RSS {peak_gb:.2f} GB"
    )
