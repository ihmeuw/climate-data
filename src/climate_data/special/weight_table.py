"""Population weight tables for temperature person-days (CLIMATE-45).

``compute_person_days`` loops over every 100 m population pixel and every day. It uses a
pixel's population only through two keys: the pixel's location and its 0.1 degree climate
cell. Summing population over the pixels that share a (location, cell) pair, once per
block and year, gives a much smaller **weight table**. Person-days for every location
then come from a loop over pairs instead of pixels, and one global day grid can serve
every block.

Cells are numbered on the global grid of the daily files: row-major, latitude descending
(89.95 to -89.95), longitude ascending (-179.95 to 179.95), so
``cell = row * 3600 + col``.
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
from climate_data.data import PopulationModelData, save_parquet
from climate_data.jobmon_utils import run_parallel_maybe_dry_run
from climate_data.special import utils

# The daily files store latitude descending; the constant is ascending.
GLOBAL_LATITUDE = cdc.TARGET_LATITUDE.to_numpy()[::-1]
GLOBAL_LONGITUDE = cdc.TARGET_LONGITUDE.to_numpy()

WEIGHT_TABLE_COLUMNS = ["year_id", "location_id", "cell", "population"]
WEIGHT_TABLE_YEARS = [int(y) for y in [*cdc.HISTORY_YEARS, *cdc.FORECAST_YEARS]]
POPULATION_RTOL = 1e-12


def weight_table_path(
    output_root: str | Path,
    hierarchy: str,
    block_key: str,
) -> Path:
    """Where one block's weight table lives: ``<root>/weights/<hierarchy>/<block>.parquet``."""
    return Path(output_root) / "weights" / hierarchy / f"{block_key}.parquet"


def missing_blocks(
    block_keys: list[str],
    output_root: str | Path,
    hierarchy: str,
) -> list[str]:
    """The blocks whose weight table has not been written yet, in input order."""
    missing = []
    for block_key in block_keys:
        if not weight_table_path(output_root, hierarchy, block_key).exists():
            missing.append(block_key)
    return missing


def check_population(
    table_total: float,
    raster_total: float,
    year: int,
    block_key: str,
) -> None:
    """Fail if a year's weight table does not hold the raster's population.

    ``raster_total`` is the float64 sum of the block's population inside locations, with
    nodata as zero. The table should equal it up to summation order.
    """
    if not np.isclose(table_total, raster_total, rtol=POPULATION_RTOL, atol=0.0):
        msg = (
            f"Weight table for {block_key} {year} holds {table_total} people, "
            f"but the raster has {raster_total} inside locations."
        )
        raise ValueError(msg)


def _window_offset(
    block_coord: npt.NDArray[Any],
    global_coord: npt.NDArray[Any],
    name: str,
) -> int:
    """Index of the block window's first value in the global coordinate.

    The whole window must equal a contiguous run of the global coordinate, so a window
    on a different grid (or in the opposite order) fails loudly rather than mapping
    cells to the wrong place.
    """
    matches = np.flatnonzero(np.isclose(global_coord, block_coord[0], atol=1e-6))
    if len(matches) != 1:
        msg = f"Block {name} {block_coord[0]} is not on the global grid."
        raise ValueError(msg)
    start = int(matches[0])
    window = global_coord[start : start + len(block_coord)]
    if len(window) != len(block_coord) or not np.allclose(
        window, block_coord, atol=1e-6
    ):
        msg = f"Block {name} is not a contiguous window of the global grid."
        raise ValueError(msg)
    return start


def global_cell_ids(
    local_cells: npt.NDArray[Any],
    block_latitude: npt.NDArray[Any],
    block_longitude: npt.NDArray[Any],
    global_latitude: npt.NDArray[Any] = GLOBAL_LATITUDE,
    global_longitude: npt.NDArray[Any] = GLOBAL_LONGITUDE,
) -> npt.NDArray[np.int32]:
    """Convert block-local climate cell indices to global cell ids.

    Parameters
    ----------
    local_cells
        Cell indices as returned by ``utils.get_temperature_coordinates``: row-major on
        the block's climate window, ``x + y * len(block_longitude)``.
    block_latitude, block_longitude
        The block's climate window coordinates.
    global_latitude, global_longitude
        The full grid, in the order of the daily files.

    Returns
    -------
    npt.NDArray[np.int32]
        ``row * len(global_longitude) + col`` on the global grid.
    """
    row0 = _window_offset(block_latitude, global_latitude, "latitude")
    col0 = _window_offset(block_longitude, global_longitude, "longitude")
    y, x = np.divmod(np.asarray(local_cells, dtype=np.int64), len(block_longitude))
    ids: npt.NDArray[np.int64] = (row0 + y) * len(global_longitude) + (col0 + x)
    return ids.astype(np.int32)


def aggregate_weights(
    location_idx: npt.NDArray[Any],
    cells: npt.NDArray[Any],
    population: npt.NDArray[Any],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Sum pixel population per (location index, cell) pair.

    NaN population (nodata) counts as zero, as in ``temperature_person_days_main``.
    Pixels outside every location (index -1) and pairs whose population is zero are
    dropped: they add nothing to any person-days sum.

    Returns
    -------
    tuple
        ``(pair_loc, pair_cell, pair_pop)``, one entry per non-empty pair.
    """
    location_idx = np.asarray(location_idx, dtype=np.int64)
    cells = np.asarray(cells, dtype=np.int64)
    population = np.nan_to_num(np.asarray(population, dtype=np.float64), nan=0.0)

    keep = (location_idx >= 0) & (population != 0)
    loc = location_idx[keep]
    cell = cells[keep]
    pop = population[keep]
    if len(loc) == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.zeros(0, dtype=np.float64)

    n_cells = int(cell.max()) + 1
    key = loc * n_cells + cell
    unique_keys, inverse = np.unique(key, return_inverse=True)
    pair_pop = np.bincount(inverse, weights=pop, minlength=len(unique_keys))
    pair_loc, pair_cell = np.divmod(unique_keys, n_cells)

    nonzero = pair_pop != 0
    return pair_loc[nonzero], pair_cell[nonzero], pair_pop[nonzero]


@numba.njit
def compute_person_days_from_weights(
    pair_loc: npt.NDArray[Any],
    pair_cell: npt.NDArray[Any],
    pair_pop: npt.NDArray[Any],
    temp_idx: npt.NDArray[Any],
    tz_idx: npt.NDArray[Any],
    out: npt.NDArray[Any],
) -> None:
    # pair_loc, pair_cell, pair_pop are (pair): output index (dim 0), cell, population
    # temp_idx is (days, cell) with values of output index (dim 1)
    # tz_idx is (cell) with values of output index (dim 2)
    # out has shape num_locations x num_temperature_bins x num_temperature_zones
    for p in range(pair_loc.shape[0]):
        loc = pair_loc[p]
        cell = pair_cell[p]
        pop = pair_pop[p]
        tz = tz_idx[cell]
        for day in range(temp_idx.shape[0]):
            out[loc, temp_idx[day, cell], tz] += pop


def _block_climate_window(
    climate_slice: dict[str, slice],
) -> tuple[xr.Dataset, npt.NDArray[Any], npt.NDArray[Any]]:
    """The block's climate window, cut from the global grid without reading data."""
    grid = xr.Dataset(
        coords={"latitude": GLOBAL_LATITUDE, "longitude": GLOBAL_LONGITUDE}
    ).sel(**climate_slice)  # type: ignore[arg-type]
    return grid, grid["latitude"].to_numpy(), grid["longitude"].to_numpy()


def build_block_weight_table(
    block_key: str,
    hierarchy: str,
    years: list[int],
    pm_data: PopulationModelData,
) -> pd.DataFrame:
    """Build one block's weight table: population per (year, location, global cell).

    The location index and the cell map are built once; population is loaded for each
    year's first quarter, as ``temperature_person_days_main`` does.
    """
    climate_slice, location_ids, location_idx = utils.build_location_index(
        hierarchy, block_key, pm_data
    )
    window, block_lat, block_lon = _block_climate_window(climate_slice)
    local_cells = utils.get_temperature_coordinates(block_key, pm_data, window)
    cells = global_cell_ids(local_cells, block_lat, block_lon)
    location_id_arr = np.asarray(location_ids, dtype=np.int32)

    inside = location_idx >= 0
    frames = []
    for year in years:
        population = pm_data.load_results(f"{year}q1", block_key)._ndarray.flatten()  # noqa: SLF001
        pair_loc, pair_cell, pair_pop = aggregate_weights(
            location_idx, cells, population
        )
        raster_total = float(np.nansum(population[inside].astype(np.float64)))
        check_population(float(pair_pop.sum()), raster_total, year, block_key)
        frames.append(
            pd.DataFrame(
                {
                    "year_id": np.full(len(pair_loc), year, dtype=np.int16),
                    "location_id": location_id_arr[pair_loc],
                    "cell": pair_cell.astype(np.int32),
                    "population": pair_pop,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[WEIGHT_TABLE_COLUMNS]


def build_block_weight_table_main(
    block_key: str,
    hierarchy: str,
    years: list[int],
    population_model_root: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Build one block's weight table and write it to ``output_path``.

    Refuses to replace an existing file unless ``overwrite`` is set.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        msg = f"{output_path} exists; pass overwrite=True to replace it."
        raise FileExistsError(msg)
    pm_data = PopulationModelData(population_model_root)
    table = build_block_weight_table(block_key, hierarchy, years, pm_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(table, output_path, atomic=True)
    return table


@click.command()
@clio.with_block_key()
@clio.with_hierarchy(default="lsae_1285")
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Root for weights/<hierarchy>/<block>.parquet. No default, by design.",
)
@clio.with_overwrite()
def build_weight_table_task(
    block_key: str,
    hierarchy: str,
    population_model_dir: str,
    output_dir: str,
    *,
    overwrite: bool,
) -> None:
    """Build one block's population weight table for 1950-2100."""
    start = time.time()
    output_path = weight_table_path(output_dir, hierarchy, block_key)
    table = build_block_weight_table_main(
        block_key,
        hierarchy,
        WEIGHT_TABLE_YEARS,
        population_model_dir,
        output_path,
        overwrite=overwrite,
    )
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(
        f"weight_table {block_key}: {len(table)} rows, "
        f"{table.location_id.nunique()} locations, "
        f"{time.time() - start:.1f} s, peak RSS {peak_gb:.2f} GB"
    )


@click.command()
@clio.with_block_key(allow_all=True)
@clio.with_hierarchy(default="lsae_1285")
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Root for weights/<hierarchy>/<block>.parquet. No default, by design.",
)
@clio.with_queue()
@clio.with_concurrency_limit(default=200)
@clio.with_dry_run()
def build_weight_tables(
    block_key: str,
    hierarchy: str,
    population_model_dir: str,
    output_dir: str,
    queue: str,
    concurrency_limit: int,
    dry_run: bool,
) -> None:
    """Build population weight tables for every block that does not have one yet."""
    pm_data = PopulationModelData(population_model_dir)
    all_blocks = pm_data.load_modeling_frame()["block_key"].unique().tolist()
    block_keys = clio.convert_choice(block_key, all_blocks)
    jobs = missing_blocks(block_keys, output_dir, hierarchy)
    print(f"Running {len(jobs)} of {len(block_keys)} blocks")

    run_parallel_maybe_dry_run(
        runner="cdtask special",
        task_name="weight_table",
        node_args={"block-key": jobs},
        task_args={
            "hierarchy": hierarchy,
            "population-model-dir": population_model_dir,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "20G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        log_root=Path(output_dir) / "logs" / "weight_table",
        max_attempts=3,
        concurrency_limit=concurrency_limit,
        dry_run=dry_run,
    )
