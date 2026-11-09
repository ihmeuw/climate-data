"""History stage of the CLIMATE-45 person-days: global admin2, per year, from ERA5.

For one year from 1950 to 2023, this bins every (location, climate cell) pair of the
population weight tables (``weight_table.py``) against the ERA5 daily mean temperature
and a temperature zone, and writes person-days keyed by (location, year, zone). It is the
same quantity as the production per-block files, summed over blocks.

The history is shared by every member and scenario. Production bins a scenario's years
before 2024 against that member's compiled ``temperature_zone``, and for 1990-2023 every
member checked equals the ERA5 10-year trailing mean, stored as int16 with a 0.01 scale
(``scratch/climate-45/section3_zone_check.py``). So the zone here is that trailing mean,
rounded the same way. Years before the first full window (1950-1958) take the first full
window's zone (1959).

Inherited on purpose: the dailies on disk are int16 with a 0.01 scale, so temperatures
on a bin edge bin one 0.1 degree bin low, exactly as in production (CLIMATE-40, comment
521591).
"""

import resource
import time
from pathlib import Path
from typing import Any

import click
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from climate_data import cli_options as clio
from climate_data import constants as cdc
from climate_data.data import ClimateData, PopulationModelData, save_parquet
from climate_data.jobmon_utils import run_parallel_maybe_dry_run
from climate_data.special import utils
from climate_data.special.weight_table import (
    GLOBAL_LATITUDE,
    GLOBAL_LONGITUDE,
    missing_blocks,
    weight_table_path,
)

# The same bins as temperature_person_days_main.
TEMPERATURE_BINS = np.arange(-35, 45, 0.1)
TEMPERATURE_ZONE_BINS = np.arange(-25, 35, 1)
# Production's column labels: the bin left edges, rounded to 0.1.
TEMPERATURE_COLUMNS = TEMPERATURE_BINS.round(1)
INDEX_NAMES = ["location_id", "year", "temperature_zone"]

ZONE_WINDOW = 10
# The compiled temperature_zone files store the zone as int16 with this scale.
ZONE_SCALE = 0.01

HISTORY_PERSON_DAYS_YEARS = [int(y) for y in cdc.HISTORY_YEARS]


def history_person_days_path(
    output_root: str | Path,
    hierarchy: str,
    year: int,
) -> Path:
    """Where one year lives: ``<root>/history/<hierarchy>/<year>.parquet``."""
    return Path(output_root) / "history" / hierarchy / f"{year}.parquet"


def missing_years(
    years: list[int],
    output_root: str | Path,
    hierarchy: str,
) -> list[int]:
    """The years whose history file has not been written yet, in input order."""
    missing = []
    for year in years:
        if not history_person_days_path(output_root, hierarchy, year).exists():
            missing.append(year)
    return missing


def check_global_grid(ds: xr.Dataset, name: str) -> None:
    """Fail unless ``ds`` is on the global grid of the daily files, in the same order.

    Cell ids in the weight tables are ``row * 3600 + col`` on that grid, so a dataset in
    another order would bin each pair against the wrong cell.
    """
    for coord, expected in (
        ("latitude", GLOBAL_LATITUDE),
        ("longitude", GLOBAL_LONGITUDE),
    ):
        actual = ds[coord].to_numpy()
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, atol=1e-6
        ):
            msg = f"{name}: {coord} is not the global grid of the daily files."
            raise ValueError(msg)


def round_zone_storage(values: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Round a zone as storing it does: int16 with a 0.01 scale, then decoded.

    Verified on 09Nov2026 to equal a member's compiled zone exactly for 1990, 2005
    and 2023. NaN stays NaN.
    """
    values = np.asarray(values, dtype=np.float64)
    rounded: npt.NDArray[np.float64] = np.round(values / ZONE_SCALE) * ZONE_SCALE
    return rounded


def trailing_zone(annual: xr.Dataset, year: int) -> npt.NDArray[np.float64]:
    """The temperature zone for ``year``: a rounded 10-year trailing mean.

    Parameters
    ----------
    annual
        Annual mean temperature with dims (year, latitude, longitude), every year from
        the first to the last.
    year
        The year to return. Years before the first full window take the first full
        window's zone.

    Returns
    -------
    npt.NDArray[np.float64]
        The zone on the (latitude, longitude) grid of ``annual``.
    """
    years = annual["year"].to_numpy().astype(int)
    first, last = int(years.min()), int(years.max())
    if not np.array_equal(np.sort(years), np.arange(first, last + 1)):
        msg = f"Annual mean temperature years are not contiguous: {years.tolist()}"
        raise ValueError(msg)
    if year < first or year > last:
        msg = f"Year {year} is outside the annual mean temperature, {first}-{last}."
        raise ValueError(msg)
    # The rolling mean over the whole series, as temperature_zone.py computes it, so the
    # floating-point sums (and so the rounding) match the compiled zone.
    rolled = annual["value"].sortby("year").rolling(year=ZONE_WINDOW).mean()
    zone_year = max(year, first + ZONE_WINDOW - 1)
    return round_zone_storage(rolled.sel(year=zone_year).to_numpy())


def decode_int16(
    raw: npt.NDArray[Any],
    scale_factor: float,
    add_offset: float,
    fill_value: int,
) -> npt.NDArray[np.float64]:
    """Decode packed int16 values as xarray does: float64, scale, offset, fill to NaN.

    Reading the raw int16 and decoding only the cells needed keeps memory down: the
    whole decoded year is 19 GB of float64.
    """
    out = np.asarray(raw).astype(np.float64)
    out *= scale_factor
    out += add_offset
    out[raw == fill_value] = np.nan
    return out


def load_year_weights(
    weights_root: str | Path,
    hierarchy: str,
    year: int,
) -> pd.DataFrame:
    """One year's population per (location, global cell), from every block's table.

    A (location, cell) pair can appear in two blocks when a location crosses a block
    edge, so the pairs are summed again.
    """
    directory = weight_table_path(weights_root, hierarchy, "any").parent
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        msg = f"No weight tables in {directory}."
        raise FileNotFoundError(msg)
    frames = []
    for path in paths:
        table = pd.read_parquet(
            path,
            columns=["location_id", "cell", "population"],
            filters=[("year_id", "==", year)],
        )
        if len(table):
            frames.append(table)
    if not frames:
        msg = f"No weight table in {directory} has rows for {year}."
        raise ValueError(msg)
    weights = (
        pd.concat(frames, ignore_index=True)
        .groupby(["location_id", "cell"], sort=True)["population"]
        .sum()
        .reset_index()
    )
    return weights[["location_id", "cell", "population"]]


def used_cells(weights: pd.DataFrame) -> npt.NDArray[np.int64]:
    """The global cells that some pair uses, sorted. Temperature is read for these only."""
    return np.unique(weights["cell"].to_numpy().astype(np.int64))


@numba.njit
def person_days_by_location_zone(
    pair_row: npt.NDArray[Any],
    pair_col: npt.NDArray[Any],
    pair_pop: npt.NDArray[Any],
    temp_idx: npt.NDArray[Any],
    out: npt.NDArray[Any],
) -> None:
    # pair_row is (pair) with values of output row: one (location, zone) row
    # pair_col is (pair) with values of the compacted cell (dim 0 of temp_idx)
    # pair_pop is (pair) with values of population
    # temp_idx is (compacted cell, day) with values of output column (temperature bin)
    # out has shape num_location_zone_rows x num_temperature_bins
    for p in range(pair_row.shape[0]):
        row = pair_row[p]
        col = pair_col[p]
        pop = pair_pop[p]
        for day in range(temp_idx.shape[1]):
            out[row, temp_idx[col, day]] += pop


def history_person_days(
    weights: pd.DataFrame,
    zone: npt.NDArray[Any],
    temperature: npt.NDArray[Any],
    year: int,
) -> pd.DataFrame:
    """Person-days by (location, zone) for one year.

    Parameters
    ----------
    weights
        ``location_id``, ``cell`` (global cell id) and ``population``, one row per pair.
    zone
        The temperature zone of every global cell, flat (``row * 3600 + col``).
    temperature
        Daily temperature, (day, cell), for exactly the cells of ``used_cells(weights)``,
        in that order.
    year
        The year, for the index.

    Returns
    -------
    pd.DataFrame
        Index (``location_id``, ``year``, ``temperature_zone``), one column per
        temperature bin, as in production. Only zones a location has are rows.
    """
    cells = used_cells(weights)
    if temperature.ndim != 2 or temperature.shape[1] != len(cells):  # noqa: PLR2004
        msg = (
            f"Temperature has shape {temperature.shape}, but the weights use "
            f"{len(cells)} cells."
        )
        raise ValueError(msg)

    pair_cell = weights["cell"].to_numpy().astype(np.int64)
    pair_col = np.searchsorted(cells, pair_cell)
    zone_idx = utils._to_idx(np.asarray(zone)[cells], TEMPERATURE_ZONE_BINS)  # noqa: SLF001
    pair_zone = zone_idx[pair_col].astype(np.int64)

    n_zones = len(TEMPERATURE_ZONE_BINS)
    location_id = weights["location_id"].to_numpy().astype(np.int64)
    row_keys, pair_row = np.unique(
        location_id * n_zones + pair_zone, return_inverse=True
    )
    row_location, row_zone = np.divmod(row_keys, n_zones)

    # (cell, day), so the kernel reads one cell's days contiguously.
    temp_idx = np.ascontiguousarray(
        utils._to_idx(temperature, TEMPERATURE_BINS).astype(np.int16).T  # noqa: SLF001
    )
    out = np.zeros((len(row_keys), len(TEMPERATURE_BINS)), dtype=np.float64)
    person_days_by_location_zone(
        pair_row.astype(np.int64),
        pair_col.astype(np.int64),
        weights["population"].to_numpy().astype(np.float64),
        temp_idx,
        out,
    )

    index = pd.MultiIndex.from_arrays(
        [
            row_location,
            np.full(len(row_keys), year, dtype=np.int64),
            TEMPERATURE_ZONE_BINS[row_zone].astype(np.int64),
        ],
        names=INDEX_NAMES,
    )
    result = pd.DataFrame(
        out, index=index, columns=pd.Index(TEMPERATURE_COLUMNS, dtype=np.float64)
    )
    return result.loc[out.any(axis=1)]


def load_history_temperature(
    cd_data: ClimateData,
    year: int,
    cells: npt.NDArray[Any],
) -> npt.NDArray[np.float64]:
    """ERA5 daily mean temperature for ``year``, (day, cell), for ``cells`` only.

    Reads the packed int16 values (4.7 GB for a year), keeps the needed cells, then
    decodes them as ``xr.open_dataset`` would.
    """
    path = cd_data.daily_results_path("historical", "mean_temperature", year)
    with xr.open_dataset(path, mask_and_scale=False) as ds:
        check_global_grid(ds, str(path))
        value = ds["value"]
        if value.dims != ("date", "latitude", "longitude"):
            msg = f"{path}: unexpected dims {value.dims}."
            raise ValueError(msg)
        raw = value.to_numpy()
        attrs = value.attrs
    packed = raw.reshape(raw.shape[0], -1)[:, cells]
    del raw
    return decode_int16(
        packed,
        float(attrs.get("scale_factor", 1.0)),
        float(attrs.get("add_offset", 0.0)),
        int(attrs["_FillValue"]),
    )


def history_person_days_main(
    year: int,
    hierarchy: str,
    climate_data_root: str | Path,
    weights_root: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Build one year of history person-days and write it to ``output_path``.

    Refuses to replace an existing file unless ``overwrite`` is set.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        msg = f"{output_path} exists; pass overwrite=True to replace it."
        raise FileExistsError(msg)
    cd_data = ClimateData(climate_data_root, read_only=True)

    weights = load_year_weights(weights_root, hierarchy, year)

    annual = cd_data.load_raw_annual_mfdataset("historical", "mean_temperature")
    check_global_grid(annual, "ERA5 annual mean temperature")
    zone = trailing_zone(annual, year).reshape(-1)
    del annual

    temperature = load_history_temperature(cd_data, year, used_cells(weights))
    result = history_person_days(weights, zone, temperature, year)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(result, output_path)
    return result


@click.command()
@clio.with_year(cdc.HISTORY_YEARS)
@clio.with_hierarchy(default="lsae_1285")
@clio.with_input_directory("climate-data", cdc.MODEL_ROOT)
@click.option(
    "--weights-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Root holding weights/<hierarchy>/. Defaults to --output-dir.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Root for history/<hierarchy>/<year>.parquet. No default, by design.",
)
@clio.with_overwrite()
def build_history_person_days_task(
    year: str,
    hierarchy: str,
    climate_data_dir: str,
    weights_dir: str | None,
    output_dir: str,
    *,
    overwrite: bool,
) -> None:
    """Build one year of global admin2 history person-days."""
    start = time.time()
    output_path = history_person_days_path(output_dir, hierarchy, int(year))
    result = history_person_days_main(
        int(year),
        hierarchy,
        climate_data_dir,
        weights_dir or output_dir,
        output_path,
        overwrite=overwrite,
    )
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    n_locations = result.index.get_level_values("location_id").nunique()
    print(
        f"history_person_days {year}: {len(result)} rows, {n_locations} locations, "
        f"{result.to_numpy().sum():.6e} person-days, "
        f"{time.time() - start:.1f} s, peak RSS {peak_gb:.2f} GB"
    )


@click.command()
@clio.with_year(cdc.HISTORY_YEARS, allow_all=True)
@clio.with_hierarchy(default="lsae_1285")
@clio.with_input_directory("climate-data", cdc.MODEL_ROOT)
@click.option(
    "--weights-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Root holding weights/<hierarchy>/. Defaults to --output-dir.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Root for history/<hierarchy>/<year>.parquet. No default, by design.",
)
@clio.with_queue()
@clio.with_concurrency_limit(default=100)
@clio.with_dry_run()
def build_history_person_days(
    year: list[str],
    hierarchy: str,
    climate_data_dir: str,
    weights_dir: str | None,
    output_dir: str,
    queue: str,
    concurrency_limit: int,
    dry_run: bool,
) -> None:
    """Build history person-days for every year that does not have a file yet."""
    weights_root = weights_dir or output_dir
    # Every block must have its weight table, or a year would silently miss people.
    pm_data = PopulationModelData(cdc.POPULATION_MODEL_ROOT)
    all_blocks = pm_data.load_modeling_frame()["block_key"].unique().tolist()
    absent = missing_blocks(all_blocks, weights_root, hierarchy)
    if absent:
        msg = f"{len(absent)} blocks have no weight table, e.g. {absent[:3]}."
        raise click.ClickException(msg)

    years = []
    for y in year:
        years.append(int(y))
    jobs = missing_years(years, output_dir, hierarchy)
    print(f"Running {len(jobs)} of {len(years)} years")

    run_parallel_maybe_dry_run(
        runner="cdtask special",
        task_name="history_person_days",
        node_args={"year": jobs},
        task_args={
            "hierarchy": hierarchy,
            "climate-data-dir": climate_data_dir,
            "weights-dir": weights_root,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "40G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        log_root=Path(output_dir) / "logs" / "history_person_days",
        max_attempts=3,
        concurrency_limit=concurrency_limit,
        dry_run=dry_run,
    )
