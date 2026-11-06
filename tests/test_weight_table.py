"""Tests for the CLIMATE-45 population weight table.

The weight table replaces the per-pixel loop of ``compute_person_days`` with a loop over
(location, climate cell) pairs whose population has been summed in advance. These tests
check that the two give the same person-days, and that block-local climate cells map to
the right cells of the global grid.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from climate_data.special import utils
from climate_data.special.utils import compute_person_days
from climate_data.special.weight_table import (
    GLOBAL_LATITUDE,
    GLOBAL_LONGITUDE,
    aggregate_weights,
    build_block_weight_table,
    check_population,
    compute_person_days_from_weights,
    global_cell_ids,
    missing_blocks,
    weight_table_path,
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


# --- Section 2: path, population check, job list, build with stubs, CLI wiring ---


def test_weight_table_path_layout(tmp_path: Path) -> None:
    path = weight_table_path(tmp_path, "lsae_1285", "B-0014X-0002Y")
    assert path == tmp_path / "weights" / "lsae_1285" / "B-0014X-0002Y.parquet"


def test_check_population_accepts_matching_totals() -> None:
    check_population(1_000_000.0, 1_000_000.0 * (1 + 1e-14), 2000, "B-0001X-0001Y")


def test_check_population_rejects_lost_population() -> None:
    with pytest.raises(ValueError, match="B-0001X-0001Y.*2000"):
        check_population(999_000.0, 1_000_000.0, 2000, "B-0001X-0001Y")


def test_missing_blocks_skips_existing_tables(tmp_path: Path) -> None:
    done = weight_table_path(tmp_path, "lsae_1285", "B-0000X-0000Y")
    done.parent.mkdir(parents=True)
    done.touch()

    missing = missing_blocks(
        ["B-0000X-0000Y", "B-0001X-0000Y", "B-0002X-0000Y"], tmp_path, "lsae_1285"
    )

    assert missing == ["B-0001X-0000Y", "B-0002X-0000Y"]


class _StubRaster:
    def __init__(self, values: npt.NDArray[Any]) -> None:
        self._ndarray = values


class _StubPopulation:
    """Four population pixels; the value scales with the year."""

    def load_results(self, time_point: str, block_key: str) -> _StubRaster:
        year = int(time_point[:4])
        scale = year - 1949
        return _StubRaster(np.array([1.0, 2.0, np.nan, 4.0]) * scale)


def test_build_block_weight_table_with_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 2 x 2 climate window at 37.95/37.85 N, -75.35/-75.25 E.
    climate_slice = {
        "latitude": slice(38.0, 37.8),
        "longitude": slice(-75.4, -75.2),
    }
    location_ids = [101, 102]
    location_idx = np.array([0, 0, 1, -1], dtype=np.int64)
    local_cells = np.array([0, 0, 3, 1], dtype=np.int64)
    monkeypatch.setattr(
        utils,
        "build_location_index",
        lambda *_: (climate_slice, location_ids, location_idx),
    )
    monkeypatch.setattr(utils, "get_temperature_coordinates", lambda *_: local_cells)

    table = build_block_weight_table(
        "B-0001X-0001Y",
        "lsae_1285",
        [1950, 1951],
        _StubPopulation(),  # type: ignore[arg-type]
    )

    assert list(table.columns) == ["year_id", "location_id", "cell", "population"]
    lat, lon = GLOBAL_LATITUDE, GLOBAL_LONGITUDE
    row0 = int(np.flatnonzero(np.isclose(lat, 37.95))[0])
    col0 = int(np.flatnonzero(np.isclose(lon, -75.35))[0])
    cell_0 = row0 * len(lon) + col0
    # Pixels 0 and 1 (loc 101, cell 0) sum to 3 x scale; pixel 2 is NaN; pixel 3 is
    # outside every location.
    got = table.set_index(["year_id", "location_id", "cell"])["population"].to_dict()
    assert got == {(1950, 101, cell_0): 3.0, (1951, 101, cell_0): 6.0}


def test_cli_commands_are_registered_and_require_output_dir() -> None:
    from click.testing import CliRunner

    from climate_data.special import RUNNERS, TASK_RUNNERS

    runner = CliRunner()
    for command in (RUNNERS["weight_table"], TASK_RUNNERS["weight_table"]):
        assert runner.invoke(command, ["--help"]).exit_code == 0
    result = runner.invoke(
        TASK_RUNNERS["weight_table"], ["--block-key", "B-0001X-0001Y"]
    )
    assert result.exit_code != 0
    assert "--output-dir" in result.output
