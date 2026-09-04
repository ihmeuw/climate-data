"""Tests for the anomaly schemes in generate/scenario_daily.py."""

from pathlib import Path

import click
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_data import constants as cdc
from climate_data.generate import scenario_daily, utils
from climate_data.generate.scenario_daily import (
    ANOMALY_TYPES,
    compute_anomaly,
    load_and_shift_longitude_and_correct_time,
    load_variable,
    variables_for_anomaly_scheme,
)

DRY_RATE = 0.0
WET_RATE = 5.0  # mm/day, unambiguously over the wet-day threshold
# The bare literal in scenario_annual.TRANSFORM_MAP's precipitation_days entry.
# Not cdc.DRY_DAY_THRESHOLD_MM, which is a different threshold for a different job.
PRECIP_DAY_THRESHOLD_MM = 0.1
DAYS_IN_COMMON_YEAR = 365
DAYS_IN_LEAP_YEAR = 366


def _daily_ds(start: str, end: str, seed: int) -> xr.Dataset:
    dates = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.5, 8.0, size=(dates.size, 2, 2))
    return xr.Dataset(
        {"value": (("date", "latitude", "longitude"), values)},
        coords={"date": dates, "latitude": [0.0, 1.0], "longitude": [10.0, 11.0]},
    )


def _write_member(
    path: Path, year: int, calendar: str, end_year: int | None = None
) -> xr.Dataset:
    """Write a synthetic one-variable CMIP6 extract on ``calendar``; return what was written.

    Days alternate bone dry and clearly wet, so any blend of two adjacent days lands on a
    value that is neither -- which is exactly what the leap-year calendar defect produced.
    Longitudes are chosen to survive ``load_and_shift_longitude``'s recentring in place, so
    a reordering cannot be mistaken for a passing comparison.
    """
    time = xr.date_range(
        f"{year}-01-01",
        f"{end_year or year}-12-31",
        freq="D",
        calendar=calendar,
        use_cftime=True,
    )
    daily = np.where(np.arange(time.size) % 2 == 0, DRY_RATE, WET_RATE)
    values = np.broadcast_to(daily[:, None, None], (time.size, 2, 2)).copy()
    ds = xr.Dataset(
        {"pr": (("time", "lat", "lon"), values)},
        coords={"time": time, "lat": [0.0, 1.0], "lon": [10.0, 20.0]},
    )
    ds.to_netcdf(path)
    return ds


@pytest.fixture(scope="module")
def reference() -> xr.Dataset:
    return _daily_ds("2019-01-01", "2023-12-31", seed=42)


@pytest.fixture(scope="module")
def target() -> xr.Dataset:
    return _daily_ds("2030-01-01", "2030-12-31", seed=7)


def test_monthly_multiplicative_unchanged(
    reference: xr.Dataset, target: xr.Dataset
) -> None:
    anomaly = compute_anomaly(
        reference, target, "multiplicative", debias_method="none", dry_day_rule="none"
    )
    monthly_ref = reference.groupby("date.month").mean("date")
    expected = (target["value"].isel(date=0) + 1) / (
        monthly_ref["value"].sel(month=1) + 1
    )
    np.testing.assert_allclose(anomaly["value"].isel(date=0).values, expected.values)


def test_yearly_is_daily_over_window_mean(
    reference: xr.Dataset, target: xr.Dataset
) -> None:
    anomaly = compute_anomaly(
        reference,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_YEARLY,
    )
    expected = target["value"] / reference["value"].mean("date")
    np.testing.assert_allclose(anomaly["value"].values, expected.values)


def test_yearly_rakes_totals_and_keeps_daily_shape(
    reference: xr.Dataset, target: xr.Dataset
) -> None:
    anomaly = compute_anomaly(
        reference,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_YEARLY,
    )
    level = 1234.5  # stand-in for the ERA5 annual-mean daily rate
    scenario = level * anomaly["value"]
    # annual total == level * n_days * (target-year mean / reference mean)
    expected_total = (
        level
        * target.sizes["date"]
        * target["value"].mean("date")
        / reference["value"].mean("date")
    )
    np.testing.assert_allclose(scenario.sum("date").values, expected_total.values)
    # each day's share of the year matches the raw GCM's own share
    np.testing.assert_allclose(
        (scenario / scenario.sum("date")).values,
        (target["value"] / target["value"].sum("date")).values,
    )


def test_yearly_delta_divides_by_jensen_factor(
    reference: xr.Dataset, target: xr.Dataset
) -> None:
    plain = compute_anomaly(
        reference,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_YEARLY,
    )
    delta = compute_anomaly(
        reference,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_YEARLY_DELTA,
    )
    yearly_means = reference.groupby("date.year").mean("date")
    n_years = yearly_means.sizes["year"]
    reference_mean = reference["value"].mean("date")
    factor = (
        1.0 + (yearly_means["value"].var("year", ddof=1) / n_years) / reference_mean**2
    )
    assert (factor > 1.0).all()
    np.testing.assert_allclose(delta["value"].values, (plain["value"] / factor).values)


@pytest.mark.parametrize(
    "scheme", [cdc.ANOMALY_SCHEME_YEARLY, cdc.ANOMALY_SCHEME_YEARLY_DELTA]
)
def test_yearly_zero_reference_forecasts_zero(
    reference: xr.Dataset, target: xr.Dataset, scheme: str
) -> None:
    ref0 = reference.copy(deep=True)
    ref0["value"].loc[{"latitude": 0.0, "longitude": 10.0}] = 0.0
    anomaly = compute_anomaly(
        ref0,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=scheme,
    )
    dry = anomaly["value"].sel(latitude=0.0, longitude=10.0)
    assert (dry == 0.0).all()
    wet = anomaly["value"].sel(latitude=1.0, longitude=11.0)
    assert np.isfinite(wet).all()
    assert (wet > 0.0).all()


def test_yearly_rejects_additive(reference: xr.Dataset, target: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="multiplicative"):
        compute_anomaly(
            reference,
            target,
            "additive",
            debias_method="none",
            dry_day_rule="none",
            anomaly_scheme=cdc.ANOMALY_SCHEME_YEARLY,
        )


def test_parse_reference_years() -> None:
    assert utils.parse_reference_years("2019-2023") == slice("2019-01-01", "2023-12-31")


@pytest.mark.parametrize(
    "bad", ["2023-2019", "2019", "abcd-efgh", "2019:2023", "19-2023", "0-99999"]
)
def test_parse_reference_years_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match="reference-years"):
        utils.parse_reference_years(bad)


def test_annual_mean_from_monthly_is_day_weighted() -> None:
    values = np.arange(1.0, 13.0)
    monthly = xr.Dataset(
        {"value": (("month",), values)},
        coords={"month": np.arange(1, 13)},
    )
    got = utils.annual_mean_from_monthly(monthly)["value"].item()
    expected = float(np.sum(values * np.array(cdc.DAYS_IN_MONTH)) / 365.0)
    assert got == pytest.approx(expected)


def test_annual_mean_from_monthly_binds_weights_by_label() -> None:
    values = np.arange(1.0, 13.0)
    monthly = xr.Dataset(
        {"value": (("month",), values)},
        coords={"month": np.arange(1, 13)},
    )
    shuffled = monthly.sel(month=[1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9])
    a = utils.annual_mean_from_monthly(monthly)["value"].item()
    b = utils.annual_mean_from_monthly(shuffled)["value"].item()
    assert a == pytest.approx(b)


def test_annual_mean_from_monthly_rejects_partial_months() -> None:
    monthly = xr.Dataset(
        {"value": (("month",), np.ones(11))},
        coords={"month": np.arange(1, 12)},
    )
    with pytest.raises(ValueError, match="month coordinate"):
        utils.annual_mean_from_monthly(monthly)


def test_unknown_anomaly_scheme_is_rejected(
    reference: xr.Dataset, target: xr.Dataset
) -> None:
    with pytest.raises(ValueError, match="Unknown anomaly scheme"):
        compute_anomaly(
            reference,
            target,
            "multiplicative",
            debias_method="none",
            dry_day_rule="none",
            anomaly_scheme="yearly_delta",
        )


def test_yearly_delta_rejects_a_single_year_reference(target: xr.Dataset) -> None:
    one_year = _daily_ds("2023-01-01", "2023-12-31", seed=3)
    with pytest.raises(ValueError, match="at least two reference years"):
        compute_anomaly(
            one_year,
            target,
            "multiplicative",
            debias_method="none",
            dry_day_rule="none",
            anomaly_scheme=cdc.ANOMALY_SCHEME_YEARLY_DELTA,
        )
    # the plain yearly scheme has no variance estimate and stays valid
    compute_anomaly(
        one_year,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_YEARLY,
    )


def test_zero_reference_guard_reports_its_count(
    reference: xr.Dataset, target: xr.Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    ref0 = reference.copy(deep=True)
    ref0["value"].loc[{"latitude": 0.0, "longitude": 10.0}] = 0.0
    compute_anomaly(
        ref0,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_YEARLY,
    )
    assert "Zero-reference guard: 1 cells" in capsys.readouterr().out


def test_monthly_ratio_is_the_per_month_ratio(
    reference: xr.Dataset, target: xr.Dataset
) -> None:
    anomaly = compute_anomaly(
        reference,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_MONTHLY_RATIO,
    )
    monthly_ref = reference.groupby("date.month").mean("date")
    expected = target["value"].isel(date=0) / monthly_ref["value"].sel(month=1)
    np.testing.assert_allclose(anomaly["value"].isel(date=0).values, expected.values)


def test_monthly_ratio_equals_yearly_for_a_seasonless_reference(
    target: xr.Dataset,
) -> None:
    dates = pd.date_range("2019-01-01", "2023-12-31", freq="D")
    flat = xr.Dataset(
        {
            "value": (
                ("date", "latitude", "longitude"),
                np.full((dates.size, 2, 2), 3.0),
            )
        },
        coords={"date": dates, "latitude": [0.0, 1.0], "longitude": [10.0, 11.0]},
    )
    a = compute_anomaly(
        flat,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_MONTHLY_RATIO,
    )
    b = compute_anomaly(
        flat,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_YEARLY,
    )
    np.testing.assert_allclose(a["value"].values, b["value"].values)


def test_monthly_ratio_zero_month_forecasts_zero_for_that_month_only(
    reference: xr.Dataset, target: xr.Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    ref0 = reference.copy(deep=True)
    ref_months = pd.DatetimeIndex(ref0["date"].to_numpy()).month
    ref0["value"].data[ref_months == 1, 0, 0] = 0.0
    anomaly = compute_anomaly(
        ref0,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_MONTHLY_RATIO,
    )
    months = pd.DatetimeIndex(anomaly["date"].to_numpy()).month
    cell = anomaly["value"].to_numpy()[:, 0, 0]
    assert (cell[months == 1] == 0.0).all()
    assert (cell[months == 2] > 0.0).all()  # noqa: PLR2004
    assert "Zero-reference guard: 1 month-cells" in capsys.readouterr().out


def test_monthly_delta_divides_by_the_per_month_jensen_factor(
    reference: xr.Dataset, target: xr.Dataset
) -> None:
    plain = compute_anomaly(
        reference,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_MONTHLY_RATIO,
    )
    delta = compute_anomaly(
        reference,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_MONTHLY_DELTA,
    )
    per_month = reference["value"].resample(date="1MS").mean()
    var_of_mean = per_month.groupby("date.month").var("date", ddof=1) / 5
    mean_m = reference["value"].groupby("date.month").mean("date")
    factor = 1.0 + var_of_mean / mean_m**2
    assert (factor > 1.0).all()
    months = pd.DatetimeIndex(plain["date"].to_numpy()).month
    daily_factor = factor.sel(month=xr.DataArray(months, dims="date")).to_numpy()
    np.testing.assert_allclose(
        delta["value"].to_numpy(), plain["value"].to_numpy() / daily_factor
    )


def test_monthly_delta_rejects_a_single_year_reference(target: xr.Dataset) -> None:
    one_year = _daily_ds("2023-01-01", "2023-12-31", seed=3)
    with pytest.raises(ValueError, match="at least two reference years"):
        compute_anomaly(
            one_year,
            target,
            "multiplicative",
            debias_method="none",
            dry_day_rule="none",
            anomaly_scheme=cdc.ANOMALY_SCHEME_MONTHLY_DELTA,
        )
    # the plain ratio needs no variance estimate and stays valid
    compute_anomaly(
        one_year,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_MONTHLY_RATIO,
    )


def test_variables_filter_is_a_passthrough_for_monthly() -> None:
    selected = ["mean_temperature", "total_precipitation"]
    got = variables_for_anomaly_scheme(
        selected, cdc.ANOMALY_SCHEME_MONTHLY, ANOMALY_TYPES
    )
    assert got == selected


def test_variables_filter_drops_additive_under_yearly() -> None:
    got = variables_for_anomaly_scheme(
        ["mean_temperature", "total_precipitation", "wind_speed"],
        cdc.ANOMALY_SCHEME_YEARLY,
        ANOMALY_TYPES,
    )
    assert got == ["total_precipitation", "wind_speed"]


def test_variables_filter_errors_when_nothing_is_runnable() -> None:
    with pytest.raises(click.UsageError, match="nothing to run"):
        variables_for_anomaly_scheme(
            ["mean_temperature"], cdc.ANOMALY_SCHEME_YEARLY, ANOMALY_TYPES
        )


def test_leap_year_does_not_blend_adjacent_days_for_a_noleap_member(
    tmp_path: Path,
) -> None:
    """Every source day must survive at its own date, unmixed with its neighbour.

    A `noleap` member has 365 days where a leap year has 366. The old conversion resampled
    values onto the longer axis by linear interpolation, so each target day became a blend
    of two source days: a dry day beside a wet one picked up a share of the wet day's rain
    and crossed the 0.1 mm wet-day threshold. `precipitation_days` inflated by ~13.5 d per
    noleap member, and by ~4.5 d in the shipped 100-draw ensemble, in all 19 leap years
    2024-2096. (CLIMATE-35)
    """
    path = tmp_path / "pr_ssp245_SYNTH-NOLEAP_r1i1p1f1.nc"
    source = _write_member(path, 2024, "noleap")

    got = load_and_shift_longitude_and_correct_time(path, "2024")

    assert got.sizes["date"] == DAYS_IN_LEAP_YEAR
    # Drop the one day the Gregorian year has and the noleap source does not; what is left
    # must be the source, value for value, in order.
    without_leap_day = got["pr"].drop_sel(date="2024-02-29")
    np.testing.assert_array_equal(without_leap_day.to_numpy(), source["pr"].to_numpy())


def test_the_leap_year_wet_day_count_no_longer_inflates(tmp_path: Path) -> None:
    """The measure the defect actually damaged: a count of days over 0.1 mm."""
    path = tmp_path / "pr_ssp245_SYNTH-NOLEAP_r1i1p1f1.nc"
    source = _write_member(path, 2024, "noleap")

    got = load_and_shift_longitude_and_correct_time(path, "2024")

    source_wet = int((source["pr"].isel(lat=0, lon=0) > PRECIP_DAY_THRESHOLD_MM).sum())
    cell = got["pr"].isel(latitude=0, longitude=0)
    got_wet = int((cell > PRECIP_DAY_THRESHOLD_MM).sum())
    # The inserted 29 February is the only day that can add to the count.
    leap_day_is_wet = cell.sel(date="2024-02-29").item() > PRECIP_DAY_THRESHOLD_MM
    assert got_wet == source_wet + int(leap_day_is_wet)


def test_february_29_is_filled_from_february_28(tmp_path: Path) -> None:
    """29 February is an explicit missing day with an explicit fill, not a smear.

    Nearest-neighbour interpolation breaks the tie toward the earlier day, so the inserted
    day copies 28 February rather than 1 March.
    """
    path = tmp_path / "pr_ssp245_SYNTH-NOLEAP_r1i1p1f1.nc"
    _write_member(path, 2024, "noleap")

    got = load_and_shift_longitude_and_correct_time(path, "2024")

    cell = got["pr"].isel(latitude=0, longitude=0)
    assert cell.sel(date="2024-02-29").item() == cell.sel(date="2024-02-28").item()


def test_a_common_year_is_untouched_for_a_noleap_member(tmp_path: Path) -> None:
    """Source and target year lengths already agree, so the conversion is a no-op."""
    path = tmp_path / "pr_ssp245_SYNTH-NOLEAP_r1i1p1f1.nc"
    source = _write_member(path, 2025, "noleap")

    got = load_and_shift_longitude_and_correct_time(path, "2025")

    assert got.sizes["date"] == DAYS_IN_COMMON_YEAR
    np.testing.assert_array_equal(got["pr"].to_numpy(), source["pr"].to_numpy())


def test_a_gregorian_member_is_bit_exact_in_a_leap_year(tmp_path: Path) -> None:
    """Most members are already on a real calendar and must pass through untouched."""
    path = tmp_path / "pr_ssp245_SYNTH-STANDARD_r1i1p1f1.nc"
    source = _write_member(path, 2024, "standard")

    got = load_and_shift_longitude_and_correct_time(path, "2024")

    assert got.sizes["date"] == DAYS_IN_LEAP_YEAR
    np.testing.assert_array_equal(got["pr"].to_numpy(), source["pr"].to_numpy())


def test_a_julian_member_drops_its_phantom_leap_day_in_2100(tmp_path: Path) -> None:
    """Julian makes 2100 a leap year; Gregorian does not, so a day has to go.

    IITM-ESM is the one julian member in the extracts. Its data stops at 2099 so it reaches
    2100 through `load_variable`'s fallback rather than here, but the conversion must be
    right for the case regardless.
    """
    path = tmp_path / "pr_ssp245_SYNTH-JULIAN_r1i1p1f1.nc"
    source = _write_member(path, 2100, "julian")
    assert source.sizes["time"] == DAYS_IN_LEAP_YEAR

    got = load_and_shift_longitude_and_correct_time(path, "2100")

    assert got.sizes["date"] == DAYS_IN_COMMON_YEAR


def test_a_member_that_stops_in_2099_is_relabelled_onto_2100(tmp_path: Path) -> None:
    """CAMS-CSM1-0 and IITM-ESM stop at 2099, so 2100 reuses 2099 under 2100's dates."""
    path = tmp_path / "pr_ssp245_SYNTH-SHORT_r1i1p1f1.nc"
    _write_member(path, 2098, "noleap", end_year=2099)

    got = load_variable(path, 2100)

    dates = pd.DatetimeIndex(got["date"].to_numpy())
    assert dates.size == DAYS_IN_COMMON_YEAR
    assert dates[0] == pd.Timestamp("2100-01-01")
    assert dates[-1] == pd.Timestamp("2100-12-31")


def test_relabelling_2099_as_2100_refuses_a_mismatched_day_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2099 that is not 365 days long must raise, not silently slide into 2101.

    The relabel used to add `date.size` days to every stamp -- using the axis COUNT as a
    calendar DURATION. The two agree only while 2099 is a complete 365-day run. Given 366
    days it would have shifted the year onto 2100-01-02..2101-01-01, and `annual_sum`'s
    `groupby("date.year")` would then have filed a day under 2101, with nothing raising.
    """
    path = tmp_path / "pr_ssp245_SYNTH-SHORT_r1i1p1f1.nc"
    _write_member(path, 2098, "noleap", end_year=2099)

    def _too_many_days(member_path: str | Path, year: str) -> xr.Dataset:
        if year != "2099":
            msg = "No data in slice"
            raise KeyError(msg)
        dates = pd.date_range("2099-01-01", periods=DAYS_IN_LEAP_YEAR)
        values = np.zeros((dates.size, 2, 2))
        return xr.Dataset(
            {"pr": (("date", "latitude", "longitude"), values)},
            coords={"date": dates, "latitude": [0.0, 1.0], "longitude": [10.0, 20.0]},
        )

    monkeypatch.setattr(
        scenario_daily, "load_and_shift_longitude_and_correct_time", _too_many_days
    )

    with pytest.raises(ValueError, match="conflicting sizes"):
        load_variable(path, 2100)
