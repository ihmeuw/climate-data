"""Tests for the anomaly schemes in generate/scenario_daily.py."""

import click
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_data import constants as cdc
from climate_data.generate import utils
from climate_data.generate.scenario_daily import (
    ANOMALY_TYPES,
    compute_anomaly,
    variables_for_anomaly_scheme,
)


def _daily_ds(start: str, end: str, seed: int) -> xr.Dataset:
    dates = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.5, 8.0, size=(dates.size, 2, 2))
    return xr.Dataset(
        {"value": (("date", "latitude", "longitude"), values)},
        coords={"date": dates, "latitude": [0.0, 1.0], "longitude": [10.0, 11.0]},
    )


@pytest.fixture(scope="module")
def reference() -> xr.Dataset:
    return _daily_ds("2019-01-01", "2023-12-31", seed=42)


@pytest.fixture(scope="module")
def target() -> xr.Dataset:
    return _daily_ds("2030-01-01", "2030-12-31", seed=7)


def test_monthly_multiplicative_unchanged(
    reference: xr.Dataset, target: xr.Dataset
) -> None:
    anomaly = compute_anomaly(reference, target, "multiplicative")
    monthly_ref = reference.groupby("date.month").mean("date")
    expected = (target["value"].isel(date=0) + 1) / (
        monthly_ref["value"].sel(month=1) + 1
    )
    np.testing.assert_allclose(anomaly["value"].isel(date=0).values, expected.values)


def test_yearly_is_daily_over_window_mean(
    reference: xr.Dataset, target: xr.Dataset
) -> None:
    anomaly = compute_anomaly(
        reference, target, "multiplicative", cdc.ANOMALY_SCHEME_YEARLY
    )
    expected = target["value"] / reference["value"].mean("date")
    np.testing.assert_allclose(anomaly["value"].values, expected.values)


def test_yearly_rakes_totals_and_keeps_daily_shape(
    reference: xr.Dataset, target: xr.Dataset
) -> None:
    anomaly = compute_anomaly(
        reference, target, "multiplicative", cdc.ANOMALY_SCHEME_YEARLY
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
        reference, target, "multiplicative", cdc.ANOMALY_SCHEME_YEARLY
    )
    delta = compute_anomaly(
        reference, target, "multiplicative", cdc.ANOMALY_SCHEME_YEARLY_DELTA
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
    anomaly = compute_anomaly(ref0, target, "multiplicative", scheme)
    dry = anomaly["value"].sel(latitude=0.0, longitude=10.0)
    assert (dry == 0.0).all()
    wet = anomaly["value"].sel(latitude=1.0, longitude=11.0)
    assert np.isfinite(wet).all()
    assert (wet > 0.0).all()


def test_yearly_rejects_additive(reference: xr.Dataset, target: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="multiplicative"):
        compute_anomaly(reference, target, "additive", cdc.ANOMALY_SCHEME_YEARLY)


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
        compute_anomaly(reference, target, "multiplicative", "yearly_delta")


def test_yearly_delta_rejects_a_single_year_reference(target: xr.Dataset) -> None:
    one_year = _daily_ds("2023-01-01", "2023-12-31", seed=3)
    with pytest.raises(ValueError, match="at least two reference years"):
        compute_anomaly(
            one_year, target, "multiplicative", cdc.ANOMALY_SCHEME_YEARLY_DELTA
        )
    # the plain yearly scheme has no variance estimate and stays valid
    compute_anomaly(one_year, target, "multiplicative", cdc.ANOMALY_SCHEME_YEARLY)


def test_zero_reference_guard_reports_its_count(
    reference: xr.Dataset, target: xr.Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    ref0 = reference.copy(deep=True)
    ref0["value"].loc[{"latitude": 0.0, "longitude": 10.0}] = 0.0
    compute_anomaly(ref0, target, "multiplicative", cdc.ANOMALY_SCHEME_YEARLY)
    assert "Zero-reference guard: 1 cells" in capsys.readouterr().out


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
