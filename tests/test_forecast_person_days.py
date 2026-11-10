"""Tests for the CLIMATE-45 forecast stage: admin2 person-days from in-memory dailies.

Production wrote each forecast member-year to disk as int16 with a 0.01 scale and binned
what it read back. The forecast stage generates the dailies in memory, so it must round
them the same way. Its temperature zone is built from its own annual means (option B1):
one chain per member-scenario runs the years in order, and each year's zone is the
10-year trailing mean over ERA5 (before 2024) and the chain's annual means (2024 on).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_data.generate import utils as gutils
from climate_data.special.forecast_person_days import (
    DAILY_FILL,
    annual_mean_path,
    annual_mean_rounded,
    forecast_person_days_path,
    forecast_zone,
    missing_chains,
    pack_int16,
    round_daily_storage,
    run_forecast_chain,
)
from climate_data.special.history_person_days import (
    decode_int16,
    history_person_days,
    trailing_zone,
    used_cells,
)

RNG_SEED = 20261110
JULY_ENCODING = {
    "dtype": "int16",
    "_FillValue": -32767,
    "scale_factor": 0.01,
    "add_offset": 0.0,
}
LATITUDE = [0.05, -0.05]
LONGITUDE = [-0.05, 0.05, 0.15]
N_CELLS = len(LATITUDE) * len(LONGITUDE)


def _netcdf_round_trip(values: np.ndarray, path: Path) -> np.ndarray:
    xr.Dataset({"value": (("date", "cell"), values)}).to_netcdf(
        path, encoding={"value": JULY_ENCODING}
    )
    with xr.open_dataset(path) as ds:
        return ds["value"].to_numpy()


# --- Rounding as the files on disk did ---


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_round_daily_storage_matches_a_netcdf_round_trip(
    tmp_path: Path, dtype: type
) -> None:
    rng = np.random.default_rng(RNG_SEED)
    values: np.ndarray = rng.normal(10.0, 15.0, size=(40, 50)).astype(dtype)
    values[0, :5] = np.nan
    values[1, :4] = np.array([12.345, -0.005, 14.995, 30.0049], dtype=dtype)

    expected = _netcdf_round_trip(values, tmp_path / "toy.nc")

    actual = round_daily_storage(values)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


def test_pack_int16_decodes_to_the_rounded_value() -> None:
    rng = np.random.default_rng(RNG_SEED)
    values = rng.normal(10.0, 15.0, size=1000)
    values[:3] = np.nan

    packed = pack_int16(values)

    assert packed.dtype == np.int16
    assert (packed[:3] == DAILY_FILL).all()
    decoded = decode_int16(packed, 0.01, 0.0, DAILY_FILL)
    np.testing.assert_array_equal(decoded, round_daily_storage(values))


def _toy_daily(year: int, seed: int, dims: tuple[str, ...]) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    values = rng.normal(15.0, 10.0, size=(len(dates), len(LATITUDE), len(LONGITUDE)))
    ds = xr.Dataset(
        {"value": (("date", "latitude", "longitude"), values)},
        coords={"date": dates, "latitude": LATITUDE, "longitude": LONGITUDE},
    )
    return ds.transpose(*dims)


def test_annual_mean_rounded_matches_the_stored_annual_file(tmp_path: Path) -> None:
    daily = _toy_daily(2030, RNG_SEED, ("date", "latitude", "longitude"))

    rounded = annual_mean_rounded(daily)

    stored = gutils.annual_mean(daily)
    stored.to_netcdf(tmp_path / "annual.nc", encoding={"value": JULY_ENCODING})
    with xr.open_dataset(tmp_path / "annual.nc") as ds:
        expected = ds["value"].sel(year=2030).to_numpy().reshape(-1)
    np.testing.assert_array_equal(rounded, expected)


# --- The zone ---


def _toy_era5_annual(n_cells: int = N_CELLS) -> xr.DataArray:
    rng = np.random.default_rng(RNG_SEED)
    years = np.arange(1950, 2024)
    return xr.DataArray(
        rng.normal(15.0, 8.0, size=(len(years), n_cells)),
        dims=("year", "cell"),
        coords={"year": years},
    )


def _toy_member_annual(years: list[int], n_cells: int = N_CELLS) -> xr.DataArray:
    rng = np.random.default_rng(RNG_SEED + 1)
    values = round_daily_storage(rng.normal(17.0, 8.0, size=(len(years), n_cells)))
    return xr.DataArray(values, dims=("year", "cell"), coords={"year": years})


def test_forecast_zone_spans_the_era5_member_seam() -> None:
    era5 = _toy_era5_annual()
    member = _toy_member_annual([2024, 2025, 2026, 2027])

    zone = forecast_zone(era5, member, 2027)

    window = np.concatenate(
        [era5.sel(year=slice(2018, 2023)).to_numpy(), member.to_numpy()]
    )
    expected = round_daily_storage(window.mean(axis=0))
    np.testing.assert_allclose(zone, expected, rtol=0, atol=1e-12)


def test_forecast_zone_equals_trailing_zone_on_the_compiled_series() -> None:
    era5 = _toy_era5_annual()
    member = _toy_member_annual(list(range(2024, 2041)))
    compiled = xr.concat([era5, member], dim="year")

    for year in (2024, 2033, 2040):
        expected = trailing_zone(xr.Dataset({"value": compiled}), year)
        np.testing.assert_array_equal(forecast_zone(era5, member, year), expected)


def test_forecast_zone_on_a_subset_of_cells_equals_the_full_grid() -> None:
    era5 = _toy_era5_annual(n_cells=40)
    member = _toy_member_annual(list(range(2024, 2036)), n_cells=40)
    cells = np.array([3, 7, 8, 21, 39])

    full = forecast_zone(era5, member, 2035)
    subset = forecast_zone(era5[:, cells], member[:, cells], 2035)

    np.testing.assert_array_equal(subset, full[cells])


def test_forecast_zone_fails_when_a_member_year_is_missing() -> None:
    era5 = _toy_era5_annual()
    member = _toy_member_annual([2024, 2026])

    with pytest.raises(ValueError, match="contiguous"):
        forecast_zone(era5, member, 2026)


# --- The chain ---


def _toy_weights(year: int) -> pd.DataFrame:
    rng = np.random.default_rng(year)
    return pd.DataFrame(
        {
            "location_id": [11, 11, 12, 13],
            "cell": [0, 4, 4, 5],
            "population": rng.gamma(2.0, 100.0, size=4),
        }
    )


class _Chain:
    """A chain on a 2 x 3 toy grid, with a generator stub that records its calls."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[int] = []
        # Rows of a (year, latitude, longitude) array flatten to cell ids 0-5.
        self.era5 = _toy_era5_annual().rename(cell="flat")

    def generate(self, year: int) -> xr.Dataset:
        self.calls.append(year)
        # Not date-first, so the chain has to transpose.
        return _toy_daily(year, year, ("latitude", "longitude", "date"))

    def output_path(self, year: int) -> Path:
        return self.root / "out" / f"{year}.parquet"

    def annual_path(self, year: int) -> Path:
        return self.root / "annual" / f"{year}.nc"

    def run(
        self,
        years: list[int],
        generate: Callable[[int], xr.Dataset] | None = None,
    ) -> list[int]:
        return run_forecast_chain(
            years,
            self.era5,
            generate or self.generate,
            _toy_weights,
            self.output_path,
            self.annual_path,
        )


def test_chain_writes_each_year_with_the_member_zone(tmp_path: Path) -> None:
    chain = _Chain(tmp_path)

    done = chain.run([2024, 2025, 2026])

    assert done == [2024, 2025, 2026]
    assert chain.calls == [2024, 2025, 2026]
    member_years = []
    member_values = []
    for year in (2024, 2025, 2026):
        assert chain.output_path(year).exists()
        with xr.open_dataset(chain.annual_path(year)) as ds:
            stored = ds["value"].to_numpy().reshape(-1)
        daily = _toy_daily(year, year, ("date", "latitude", "longitude"))
        np.testing.assert_array_equal(stored, annual_mean_rounded(daily))
        member_years.append(year)
        member_values.append(stored)

    member = xr.DataArray(
        np.stack(member_values), dims=("year", "flat"), coords={"year": member_years}
    )
    zone = forecast_zone(chain.era5, member, 2026)
    daily = _toy_daily(2026, 2026, ("date", "latitude", "longitude"))
    weights = _toy_weights(2026)
    temperature = round_daily_storage(
        daily["value"].to_numpy().reshape(len(daily["date"]), -1)
    )
    expected = history_person_days(
        weights, zone, temperature[:, used_cells(weights)], 2026
    )
    actual = pd.read_parquet(chain.output_path(2026))
    pd.testing.assert_frame_equal(actual, expected)


def test_chain_resumes_from_annual_means_on_disk(tmp_path: Path) -> None:
    _Chain(tmp_path).run([2024, 2025, 2026])
    full = pd.read_parquet(_Chain(tmp_path).output_path(2026))
    _Chain(tmp_path).output_path(2026).unlink()

    def refuse(year: int) -> xr.Dataset:
        if year != 2026:  # noqa: PLR2004
            msg = f"regenerated finished year {year}"
            raise AssertionError(msg)
        return _toy_daily(year, year, ("latitude", "longitude", "date"))

    resumed = _Chain(tmp_path)
    done = resumed.run([2024, 2025, 2026], generate=refuse)

    assert done == [2026]
    pd.testing.assert_frame_equal(pd.read_parquet(resumed.output_path(2026)), full)


def test_chain_decodes_only_the_zone_window_years(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from climate_data.special import history_person_days as hpd

    years = list(range(2024, 2037))  # 13 years; the window is 10
    decoded = []

    def counting_decode(
        raw: np.ndarray, scale_factor: float, add_offset: float, fill_value: int
    ) -> np.ndarray:
        decoded.append(raw.size)
        return decode_int16(raw, scale_factor, add_offset, fill_value)

    monkeypatch.setattr(hpd, "decode_int16", counting_decode)
    chain = _Chain(tmp_path)
    chain.run(years)

    expected_decodes = 0
    for i in range(1, len(years) + 1):
        expected_decodes += min(i, hpd.ZONE_WINDOW)
    assert len(decoded) == expected_decodes

    # The last year still bins against the zone from the whole member series.
    member_values = []
    for year in years:
        with xr.open_dataset(chain.annual_path(year)) as ds:
            member_values.append(ds["value"].to_numpy().reshape(-1))
    member = xr.DataArray(
        np.stack(member_values), dims=("year", "flat"), coords={"year": years}
    )
    last = years[-1]
    zone = forecast_zone(chain.era5, member, last)
    daily = _toy_daily(last, last, ("date", "latitude", "longitude"))
    weights = _toy_weights(last)
    temperature = round_daily_storage(
        daily["value"].to_numpy().reshape(len(daily["date"]), -1)
    )
    expected = history_person_days(
        weights, zone, temperature[:, used_cells(weights)], last
    )
    pd.testing.assert_frame_equal(pd.read_parquet(chain.output_path(last)), expected)


def test_chain_starting_late_needs_the_earlier_annual_means(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="annual mean for 2024"):
        _Chain(tmp_path).run([2026, 2027])


def test_chain_rejects_years_out_of_order(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _Chain(tmp_path).run([2024, 2026])


# --- Paths, job list, CLI wiring ---


def test_paths(tmp_path: Path) -> None:
    assert forecast_person_days_path(tmp_path, "lsae_1285", "ssp245", "M_r1", 2050) == (
        tmp_path / "forecast" / "lsae_1285" / "ssp245" / "M_r1" / "2050.parquet"
    )
    assert annual_mean_path(tmp_path, "ssp245", "M_r1", 2050) == (
        tmp_path / "annual_mean" / "ssp245" / "M_r1" / "2050.nc"
    )


def test_missing_chains_skips_finished_chains(tmp_path: Path) -> None:
    for year in (2024, 2025):
        path = forecast_person_days_path(tmp_path, "lsae_1285", "ssp245", "A", year)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    path = forecast_person_days_path(tmp_path, "lsae_1285", "ssp245", "B", 2024)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

    chains = missing_chains(
        [("ssp245", "A"), ("ssp245", "B"), ("ssp126", "A")],
        [2024, 2025],
        tmp_path,
        "lsae_1285",
    )

    assert chains == [("ssp245", "B"), ("ssp126", "A")]


def test_cli_commands_are_registered_and_require_output_dir() -> None:
    from click.testing import CliRunner

    from climate_data.special import RUNNERS, TASK_RUNNERS

    runner = CliRunner()
    for command in (
        RUNNERS["forecast_person_days"],
        TASK_RUNNERS["forecast_person_days"],
    ):
        assert runner.invoke(command, ["--help"]).exit_code == 0
    result = runner.invoke(
        TASK_RUNNERS["forecast_person_days"],
        ["--cmip6-experiment", "ssp245", "--gcm-member", "CNRM-CM6-1_r4i1p1f2"],
    )
    assert result.exit_code != 0
    assert "--output-dir" in result.output


def test_runner_checks_blocks_in_the_population_model_dir_it_is_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from click.testing import CliRunner

    from climate_data.special import RUNNERS
    from climate_data.special import forecast_person_days as fpd

    seen = []

    def fake_frame_block_keys(population_model_root: str) -> list[str]:
        seen.append(population_model_root)
        return ["B-0000X-0000Y"]  # no weight table exists, so the runner stops

    monkeypatch.setattr(fpd, "frame_block_keys", fake_frame_block_keys)
    pm_dir = tmp_path / "population-model"
    pm_dir.mkdir()
    result = CliRunner().invoke(
        RUNNERS["forecast_person_days"],
        ["--population-model-dir", str(pm_dir), "-o", str(tmp_path), "--dry-run"],
    )

    assert seen == [str(pm_dir)]
    assert "1 blocks have no weight table" in result.output
