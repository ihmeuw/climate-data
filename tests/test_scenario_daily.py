"""Tests for the anomaly schemes in generate/scenario_daily.py."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_data import constants as cdc
from climate_data.generate import utils
from climate_data.generate.scenario_daily import compute_anomaly


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


def test_yearly_rejects_additive(reference: xr.Dataset, target: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="multiplicative"):
        compute_anomaly(reference, target, "additive", cdc.ANOMALY_SCHEME_YEARLY)


def test_parse_reference_years() -> None:
    assert utils.parse_reference_years("2019-2023") == slice("2019-01-01", "2023-12-31")


@pytest.mark.parametrize("bad", ["2023-2019", "2019", "abcd-efgh", "2019:2023"])
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
