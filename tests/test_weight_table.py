"""Tests for the CLIMATE-45 population weight table.

The weight table replaces the per-pixel loop of ``compute_person_days`` with a loop over
(location, climate cell) pairs whose population has been summed in advance. These tests
check that the two give the same person-days, and that block-local climate cells map to
the right cells of the global grid.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from climate_data.special.utils import compute_person_days
from climate_data.special.weight_table import (
    aggregate_weights,
    compute_person_days_from_weights,
    global_cell_ids,
)

RNG_SEED = 20261106
NAN_FRACTION = 0.1


def _pairs_as_dict(
    pair_loc: npt.NDArray[Any],
    pair_cell: npt.NDArray[Any],
    pair_pop: npt.NDArray[Any],
) -> dict[tuple[int, int], float]:
    out = {}
    for loc, cell, pop in zip(pair_loc, pair_cell, pair_pop, strict=True):
        out[(int(loc), int(cell))] = float(pop)
    return out


def test_aggregate_weights_sums_population_per_location_and_cell() -> None:
    # Five pixels: two share (loc 0, cell 3), one is (loc 1, cell 3), one is (loc 0,
    # cell 7), and one belongs to no location.
    location_idx = np.array([0, 0, 1, 0, -1], dtype=np.int64)
    cells = np.array([3, 3, 3, 7, 3], dtype=np.int64)
    population = np.array([1.5, 2.5, 4.0, 8.0, 100.0], dtype=np.float64)

    pair_loc, pair_cell, pair_pop = aggregate_weights(location_idx, cells, population)

    assert _pairs_as_dict(pair_loc, pair_cell, pair_pop) == {
        (0, 3): 4.0,
        (1, 3): 4.0,
        (0, 7): 8.0,
    }


def test_aggregate_weights_treats_nan_as_zero_and_drops_empty_pairs() -> None:
    location_idx = np.array([0, 0, 1, 2], dtype=np.int64)
    cells = np.array([5, 5, 6, 9], dtype=np.int64)
    population = np.array([3.0, np.nan, np.nan, 0.0], dtype=np.float64)

    pair_loc, pair_cell, pair_pop = aggregate_weights(location_idx, cells, population)

    # (0, 5) keeps the finite pixel; (1, 6) is all-NaN and (2, 9) is zero: both dropped.
    assert _pairs_as_dict(pair_loc, pair_cell, pair_pop) == {(0, 5): 3.0}
    assert np.isfinite(pair_pop).all()


def test_aggregate_weights_conserves_population_inside_locations() -> None:
    rng = np.random.default_rng(RNG_SEED)
    n_pix = 5_000
    location_idx = rng.integers(-1, 6, size=n_pix)
    cells = rng.integers(0, 40, size=n_pix)
    population = rng.gamma(1.0, 10.0, size=n_pix)
    population[rng.random(n_pix) < NAN_FRACTION] = np.nan

    _, _, pair_pop = aggregate_weights(location_idx, cells, population)

    inside = location_idx >= 0
    expected = np.nansum(population[inside])
    assert pair_pop.sum() == pytest.approx(expected, rel=1e-12)


def test_weights_kernel_matches_pixel_kernel() -> None:
    """The pair loop gives the same person-days as the pixel loop."""
    rng = np.random.default_rng(RNG_SEED)
    n_pix, n_loc, n_cells, n_days = 4_000, 7, 30, 12
    n_temp, n_tz = 25, 5

    location_idx = rng.integers(-1, n_loc, size=n_pix)
    temp_coords = rng.integers(0, n_cells, size=n_pix)
    population = rng.gamma(1.0, 10.0, size=n_pix)
    temp_idx = rng.integers(0, n_temp, size=(n_days, n_cells))
    tz_idx = rng.integers(0, n_tz, size=n_cells)

    expected = np.zeros((n_loc, n_temp, n_tz), dtype=np.float64)
    compute_person_days(
        location_idx, temp_idx, tz_idx, population, temp_coords, expected
    )

    pair_loc, pair_cell, pair_pop = aggregate_weights(
        location_idx, temp_coords, population
    )
    actual = np.zeros_like(expected)
    compute_person_days_from_weights(
        pair_loc, pair_cell, pair_pop, temp_idx, tz_idx, actual
    )

    assert expected.sum() > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def _toy_global_grid() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Like the daily files: latitude descending, longitude ascending, 0.1 degree cells.
    latitude = np.round(np.arange(89.95, -90.0, -0.1), 2)
    longitude = np.round(np.arange(-179.95, 180.0, 0.1), 2)
    return latitude, longitude


def test_global_cell_ids_maps_block_window_into_global_grid() -> None:
    global_lat, global_lon = _toy_global_grid()
    # A 3 x 4 window starting at global row 500, column 1000.
    block_lat = global_lat[500:503]
    block_lon = global_lon[1000:1004]
    # Block-local ids are row-major on the window: x + y * n_lon.
    local = np.array([0, 3, 4, 11], dtype=np.int64)

    ids = global_cell_ids(local, block_lat, block_lon, global_lat, global_lon)

    n_lon = len(global_lon)
    expected = np.array(
        [
            500 * n_lon + 1000,
            500 * n_lon + 1003,
            501 * n_lon + 1000,
            502 * n_lon + 1003,
        ]
    )
    np.testing.assert_array_equal(ids, expected)
    assert ids.dtype == np.int32


def test_global_cell_ids_round_trips_a_coordinate() -> None:
    global_lat, global_lon = _toy_global_grid()
    block_lat = global_lat[517:530]  # 38.25 down to 37.05
    block_lon = global_lon[1046:1060]  # -75.35 to -74.05
    y, x = 4, 9
    local = np.array([x + y * len(block_lon)], dtype=np.int64)

    (cell,) = global_cell_ids(local, block_lat, block_lon, global_lat, global_lon)

    row, col = divmod(int(cell), len(global_lon))
    assert global_lat[row] == block_lat[y]
    assert global_lon[col] == block_lon[x]


def test_global_cell_ids_rejects_window_not_on_the_grid() -> None:
    global_lat, global_lon = _toy_global_grid()
    block_lat = global_lat[10:13] + 0.05  # off-grid by half a cell
    block_lon = global_lon[20:24]

    with pytest.raises(ValueError, match="latitude"):
        global_cell_ids(np.array([0]), block_lat, block_lon, global_lat, global_lon)
