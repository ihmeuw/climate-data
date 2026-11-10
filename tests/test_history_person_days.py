"""Tests for the CLIMATE-45 history stage: global admin2 person-days from ERA5, per year.

The history stage bins every (location, climate cell) pair of the weight tables against
the ERA5 dailies and a temperature zone shared by every member. These tests check the zone
rule, the decoding of the int16 dailies, the reading of the weight tables, and that the
compacted (location, zone) kernel gives the same person-days as the dense one.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_data.special.history_person_days import (
    TEMPERATURE_BINS,
    TEMPERATURE_COLUMNS,
    TEMPERATURE_ZONE_BINS,
    check_global_grid,
    decode_int16,
    history_person_days,
    history_person_days_path,
    load_weights_by_year,
    load_year_weights,
    missing_years,
    round_zone_storage,
    trailing_zone,
    used_cells,
)
from climate_data.special.weight_table import (
    WEIGHT_TABLE_COLUMNS,
    compute_person_days_from_weights,
    weight_table_path,
)

RNG_SEED = 20261109


# --- The temperature zone ---


def test_round_zone_storage_matches_int16_scale_factor() -> None:
    values = np.array([14.996, 14.994, -3.125, np.nan])

    rounded = round_zone_storage(values)

    np.testing.assert_array_equal(rounded[:2], [15.0, 14.99])
    # The same arithmetic as the stored file: round(value / 0.01), times 0.01.
    assert rounded[2] == np.round(-3.125 / 0.01) * 0.01
    assert np.isnan(rounded[3])


def _toy_annual(first_year: int, n_years: int) -> xr.Dataset:
    """Annual mean temperature on a 2 x 3 grid; each year's value is distinct."""
    rng = np.random.default_rng(RNG_SEED)
    years = np.arange(first_year, first_year + n_years)
    values = rng.normal(15.0, 8.0, size=(n_years, 2, 3))
    return xr.Dataset(
        {"value": (("year", "latitude", "longitude"), values)},
        coords={
            "year": years,
            "latitude": [0.05, -0.05],
            "longitude": [-0.05, 0.05, 0.15],
        },
    )


def test_trailing_zone_is_rounded_ten_year_mean() -> None:
    annual = _toy_annual(1950, 20)

    zone = trailing_zone(annual, 1962)

    expected = round_zone_storage(
        annual["value"].sel(year=slice(1953, 1962)).mean("year").to_numpy()
    )
    np.testing.assert_allclose(zone, expected, rtol=0, atol=1e-12)
    assert zone.shape == (2, 3)


def _stored_like_annual(n_years: int, n_cells: int) -> xr.Dataset:
    """Annual means as the stored files hold them: multiples of 0.01, dims (year, cell)."""
    rng = np.random.default_rng(RNG_SEED)
    values = np.round(rng.normal(15.0, 8.0, size=(n_years, n_cells)) / 0.01) * 0.01
    return xr.Dataset(
        {"value": (("year", "cell"), values)},
        coords={"year": np.arange(2000, 2000 + n_years)},
    )


def test_trailing_zone_sums_the_window_left_to_right() -> None:
    # Production's zone is xarray's rolling mean on a C-ordered global grid, which sums each
    # window from its first year to its last (checked 09Nov2026: equal to the stored zone
    # at every cell). On a column-major array, which is what selecting cells with
    # ``values[:, cells]`` returns, xarray sums pairwise instead, and that rounds
    # differently at the 0.005 halfway points. The zone must not depend on memory layout.
    annual = _stored_like_annual(12, 60_000)
    window = annual["value"].sel(year=slice(2002, 2011)).to_numpy()
    left_to_right = window[0].copy()
    for row in window[1:]:
        left_to_right += row
    pairwise = np.sum(np.ascontiguousarray(window.T), axis=1)
    # The data must be able to tell the two orders apart.
    assert (
        round_zone_storage(left_to_right / 10) != round_zone_storage(pairwise / 10)
    ).any()
    expected = round_zone_storage(left_to_right / 10)

    np.testing.assert_array_equal(trailing_zone(annual, 2011), expected)
    cells = np.arange(0, 60_000, 7)
    column_major = xr.Dataset(
        {"value": (("year", "cell"), annual["value"].to_numpy()[:, cells])},
        coords={"year": annual["year"]},
    )
    assert column_major["value"].to_numpy().flags["F_CONTIGUOUS"]
    np.testing.assert_array_equal(trailing_zone(column_major, 2011), expected[cells])


def test_trailing_zone_is_nan_when_the_window_has_a_nan() -> None:
    annual = _stored_like_annual(12, 4)
    annual["value"][5, 1] = np.nan

    zone = trailing_zone(annual, 2011)

    assert np.isnan(zone[1])
    assert np.isfinite(zone[[0, 2, 3]]).all()


def test_trailing_zone_holds_first_full_window_for_early_years() -> None:
    annual = _toy_annual(1950, 20)

    first_full = trailing_zone(annual, 1959)
    for year in range(1950, 1959):
        np.testing.assert_array_equal(trailing_zone(annual, year), first_full)


def test_trailing_zone_rejects_years_outside_the_series() -> None:
    annual = _toy_annual(1950, 20)

    with pytest.raises(ValueError, match="1949"):
        trailing_zone(annual, 1949)
    with pytest.raises(ValueError, match="1970"):
        trailing_zone(annual, 1970)


def test_trailing_zone_rejects_a_gap_in_the_years() -> None:
    annual = _toy_annual(1950, 20).drop_sel(year=1955)

    with pytest.raises(ValueError, match="contiguous"):
        trailing_zone(annual, 1965)


def test_check_global_grid_rejects_ascending_latitude() -> None:
    lat = np.round(np.arange(-89.95, 90.0, 0.1), 2)
    lon = np.round(np.arange(-179.95, 180.0, 0.1), 2)
    ds = xr.Dataset(coords={"latitude": lat, "longitude": lon})

    with pytest.raises(ValueError, match="latitude"):
        check_global_grid(ds, "toy")


# --- The daily temperature ---


def test_decode_int16_matches_xarray(tmp_path: Path) -> None:
    rng = np.random.default_rng(RNG_SEED)
    values = rng.normal(10.0, 15.0, size=(4, 5))
    values[0, 0] = np.nan
    values[1, 2] = 12.345  # on a 0.005 edge before rounding to int16
    path = tmp_path / "toy.nc"
    xr.Dataset({"value": (("date", "cell"), values)}).to_netcdf(
        path,
        encoding={
            "value": {
                "dtype": "int16",
                "_FillValue": -32767,
                "scale_factor": 0.01,
                "add_offset": 0.0,
            }
        },
    )

    with xr.open_dataset(path) as ds:
        expected = ds["value"].to_numpy()
    with xr.open_dataset(path, mask_and_scale=False) as ds:
        raw = ds["value"]
        actual = decode_int16(
            raw.to_numpy(),
            raw.attrs["scale_factor"],
            raw.attrs["add_offset"],
            raw.attrs["_FillValue"],
        )

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


# --- The weight tables ---

BLOCKS = ["B-0000X-0000Y", "B-0001X-0000Y", "B-0002X-0000Y"]


def _write_block(
    root: Path, block_key: str, rows: list[tuple[int, int, int, float]]
) -> None:
    table = pd.DataFrame(rows, columns=WEIGHT_TABLE_COLUMNS).astype(
        {"year_id": np.int16, "location_id": np.int32, "cell": np.int32}
    )
    path = weight_table_path(root, "lsae_1285", block_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path)


def test_load_year_weights_sums_a_pair_split_across_blocks(tmp_path: Path) -> None:
    _write_block(
        tmp_path,
        "B-0000X-0000Y",
        [(2000, 7, 100, 1.5), (2000, 7, 101, 2.0), (2001, 7, 100, 50.0)],
    )
    _write_block(tmp_path, "B-0001X-0000Y", [(2000, 7, 100, 0.5), (2000, 8, 101, 4.0)])
    _write_block(tmp_path, "B-0002X-0000Y", [])  # an ocean block: an empty table

    weights = load_year_weights(tmp_path, "lsae_1285", 2000, BLOCKS)

    assert list(weights.columns) == ["location_id", "cell", "population"]
    got = weights.set_index(["location_id", "cell"])["population"].to_dict()
    assert got == {(7, 100): 2.0, (7, 101): 2.0, (8, 101): 4.0}


def test_load_year_weights_reads_only_the_frame_blocks(tmp_path: Path) -> None:
    _write_block(tmp_path, "B-0000X-0000Y", [(2000, 7, 100, 1.5)])
    _write_block(tmp_path, "B-0001X-0000Y", [(2000, 7, 100, 0.5)])
    # A table for a block not in the frame (a stray file, or an old block layout).
    _write_block(tmp_path, "B-0009X-0009Y", [(2000, 7, 100, 1000.0)])

    weights = load_year_weights(
        tmp_path, "lsae_1285", 2000, ["B-0000X-0000Y", "B-0001X-0000Y"]
    )

    assert weights["population"].tolist() == [2.0]


def test_load_year_weights_fails_on_a_missing_frame_block(tmp_path: Path) -> None:
    _write_block(tmp_path, "B-0000X-0000Y", [(2000, 7, 100, 1.5)])

    with pytest.raises(FileNotFoundError, match="B-0001X-0000Y"):
        load_year_weights(
            tmp_path, "lsae_1285", 2000, ["B-0000X-0000Y", "B-0001X-0000Y"]
        )


def test_load_weights_by_year_equals_one_year_at_a_time(tmp_path: Path) -> None:
    _write_block(
        tmp_path,
        "B-0000X-0000Y",
        [(2000, 7, 100, 1.5), (2001, 7, 100, 50.0), (2002, 8, 102, 3.0)],
    )
    _write_block(
        tmp_path,
        "B-0001X-0000Y",
        [(2000, 7, 100, 0.5), (2001, 8, 101, 4.0), (2002, 7, 100, 9.0)],
    )
    _write_block(tmp_path, "B-0002X-0000Y", [])

    by_year = load_weights_by_year(tmp_path, "lsae_1285", [2000, 2002], BLOCKS)

    assert sorted(by_year) == [2000, 2002]
    for year in (2000, 2002):
        pd.testing.assert_frame_equal(
            by_year[year], load_year_weights(tmp_path, "lsae_1285", year, BLOCKS)
        )
    got = by_year[2002].set_index(["location_id", "cell"])["population"].to_dict()
    assert got == {(7, 100): 9.0, (8, 102): 3.0}


def test_load_weights_by_year_fails_on_a_year_with_no_rows(tmp_path: Path) -> None:
    _write_block(tmp_path, "B-0000X-0000Y", [(2000, 7, 100, 1.5)])
    _write_block(tmp_path, "B-0001X-0000Y", [])
    _write_block(tmp_path, "B-0002X-0000Y", [])

    with pytest.raises(ValueError, match="2001"):
        load_weights_by_year(tmp_path, "lsae_1285", [2000, 2001], BLOCKS)


def test_load_year_weights_fails_without_tables(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_year_weights(tmp_path, "lsae_1285", 2000, BLOCKS)


# --- The person-days ---


def _random_case(
    n_pairs: int = 300,
    n_loc: int = 9,
    n_cells: int = 200,
    n_days: int = 31,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Weights on a toy global grid, a zone per cell, and daily temperature per cell."""
    rng = np.random.default_rng(RNG_SEED)
    pairs = pd.DataFrame(
        {
            "location_id": rng.integers(1000, 1000 + n_loc, size=n_pairs),
            "cell": rng.integers(0, n_cells, size=n_pairs),
            "population": rng.gamma(1.0, 10.0, size=n_pairs),
        }
    )
    weights = pairs.groupby(["location_id", "cell"])["population"].sum().reset_index()
    zone = rng.uniform(-30.0, 40.0, size=n_cells)  # beyond both ends of the zone bins
    temperature = rng.uniform(-40.0, 50.0, size=(n_days, n_cells))
    return weights, zone, temperature


def _dense_person_days(
    weights: pd.DataFrame,
    zone: np.ndarray,
    temperature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Person-days with the section 1 kernel: locations x temperature bins x zones."""
    location_ids, pair_loc = np.unique(
        weights["location_id"].to_numpy(), return_inverse=True
    )
    temp_idx = (
        np.clip(np.digitize(temperature, TEMPERATURE_BINS), 1, len(TEMPERATURE_BINS))
        - 1
    )
    tz_idx = (
        np.clip(np.digitize(zone, TEMPERATURE_ZONE_BINS), 1, len(TEMPERATURE_ZONE_BINS))
        - 1
    )
    out = np.zeros(
        (len(location_ids), len(TEMPERATURE_BINS), len(TEMPERATURE_ZONE_BINS))
    )
    compute_person_days_from_weights(
        pair_loc,
        weights["cell"].to_numpy(),
        weights["population"].to_numpy(),
        temp_idx,
        tz_idx,
        out,
    )
    return location_ids, out


def test_history_person_days_matches_dense_kernel() -> None:
    weights, zone, temperature = _random_case()
    cells = used_cells(weights)

    result = history_person_days(weights, zone, temperature[:, cells], 2005)

    location_ids, dense = _dense_person_days(weights, zone, temperature)
    expected_rows = {}
    for i, location_id in enumerate(location_ids):
        for z, zone_value in enumerate(TEMPERATURE_ZONE_BINS):
            if dense[i, :, z].any():
                expected_rows[(int(location_id), 2005, int(zone_value))] = dense[
                    i, :, z
                ]
    assert len(expected_rows) > len(location_ids)  # several zones per location
    actual_rows = {}
    for key, row in zip(result.index, result.to_numpy(), strict=True):
        actual_rows[key] = row
    assert set(actual_rows) == set(expected_rows)
    for key, row in expected_rows.items():
        np.testing.assert_allclose(actual_rows[key], row, rtol=1e-12, atol=0)


def test_history_person_days_has_production_layout() -> None:
    weights, zone, temperature = _random_case()
    cells = used_cells(weights)

    result = history_person_days(weights, zone, temperature[:, cells], 2005)

    assert result.index.names == ["location_id", "year", "temperature_zone"]
    for level in range(3):
        assert result.index.get_level_values(level).dtype == np.int64
    assert result.columns.dtype == np.float64
    np.testing.assert_array_equal(result.columns.to_numpy(), TEMPERATURE_COLUMNS)
    # The pilot's and production's labels: -35.0, -34.9, ..., 44.9.
    np.testing.assert_array_equal(TEMPERATURE_COLUMNS[[0, 1, -1]], [-35.0, -34.9, 44.9])
    assert len(TEMPERATURE_COLUMNS) == len(TEMPERATURE_BINS)
    assert result.to_numpy().dtype == np.float64
    assert (result.sum(axis=1) > 0).all()  # no all-zero zone rows
    assert result.index.is_monotonic_increasing


def test_history_person_days_conserves_population_times_days() -> None:
    weights, zone, temperature = _random_case(n_days=365)
    cells = used_cells(weights)

    result = history_person_days(weights, zone, temperature[:, cells], 2005)

    expected = weights["population"].sum() * 365
    assert result.to_numpy().sum() == pytest.approx(expected, rel=1e-12)


def test_history_person_days_rejects_temperature_for_other_cells() -> None:
    weights, zone, temperature = _random_case()

    with pytest.raises(ValueError, match="cells"):
        history_person_days(weights, zone, temperature, 2005)


# --- Paths, job list, CLI wiring ---


def test_history_person_days_path_layout(tmp_path: Path) -> None:
    path = history_person_days_path(tmp_path, "lsae_1285", 1990)
    assert path == tmp_path / "history" / "lsae_1285" / "1990.parquet"


def test_missing_years_skips_written_years(tmp_path: Path) -> None:
    done = history_person_days_path(tmp_path, "lsae_1285", 1951)
    done.parent.mkdir(parents=True)
    done.touch()

    assert missing_years([1950, 1951, 1952], tmp_path, "lsae_1285") == [1950, 1952]


def test_cli_commands_are_registered_and_require_output_dir() -> None:
    from click.testing import CliRunner

    from climate_data.special import RUNNERS, TASK_RUNNERS

    runner = CliRunner()
    for command in (
        RUNNERS["history_person_days"],
        TASK_RUNNERS["history_person_days"],
    ):
        assert runner.invoke(command, ["--help"]).exit_code == 0
    result = runner.invoke(TASK_RUNNERS["history_person_days"], ["--year", "1990"])
    assert result.exit_code != 0
    assert "--output-dir" in result.output


def test_runner_checks_blocks_in_the_population_model_dir_it_is_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from click.testing import CliRunner

    from climate_data.special import RUNNERS
    from climate_data.special import history_person_days as hpd

    seen = []

    def fake_frame_block_keys(population_model_root: str) -> list[str]:
        seen.append(population_model_root)
        return ["B-0000X-0000Y"]  # no weight table exists, so the runner stops

    monkeypatch.setattr(hpd, "frame_block_keys", fake_frame_block_keys)
    pm_dir = tmp_path / "population-model"
    pm_dir.mkdir()
    result = CliRunner().invoke(
        RUNNERS["history_person_days"],
        ["--population-model-dir", str(pm_dir), "-o", str(tmp_path), "--dry-run"],
    )

    assert seen == [str(pm_dir)]
    assert "1 blocks have no weight table" in result.output
