"""Tests for the annual runner's anomaly-scheme handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from climate_data import constants as cdc
from climate_data.generate import scenario_annual

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_forecast_jobs_filter_is_a_passthrough_for_monthly() -> None:
    jobs = [("mean_temperature", "ssp126", "2030", "m1")]
    got = scenario_annual.forecast_jobs_for_anomaly_scheme(
        jobs, cdc.ANOMALY_SCHEME_MONTHLY
    )
    assert got == jobs


def test_forecast_jobs_filter_keeps_historical_additive_jobs() -> None:
    jobs = [
        ("mean_temperature", "historical", "2020", "m1"),
        ("mean_temperature", "ssp126", "2030", "m1"),
        ("total_precipitation", "ssp126", "2030", "m1"),
    ]
    got = scenario_annual.forecast_jobs_for_anomaly_scheme(
        jobs, cdc.ANOMALY_SCHEME_YEARLY
    )
    # the additive variable stays runnable for historical, is dropped for forecasts
    assert got == [jobs[0], jobs[2]]


def test_annual_main_threads_the_scheme_into_the_daily_builds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression for the review blocker: the flags must reach the daily main."""
    captured: dict[str, object] = {}
    dates = pd.date_range("2030-01-01", "2030-12-31", freq="D")

    def fake_daily_main(**kwargs: object) -> xr.Dataset:
        captured.update(kwargs)
        return xr.Dataset(
            {"value": (("date", "latitude", "longitude"), np.ones((dates.size, 1, 1)))},
            coords={"date": dates, "latitude": [0.0], "longitude": [0.0]},
        )

    saved: dict[str, object] = {}

    class FakeClimateData:
        def __init__(self, root: str | Path) -> None:
            pass

        def save_raw_annual_results(self, ds: xr.Dataset, **kwargs: object) -> None:
            saved["ds"] = ds

    monkeypatch.setattr(
        scenario_annual, "generate_scenario_daily_main", fake_daily_main
    )
    monkeypatch.setattr(scenario_annual, "ClimateData", FakeClimateData)

    scenario_annual.generate_scenario_annual_main(
        "total_precipitation",
        "ssp126",
        "2030",
        "m1",
        tmp_path,
        anomaly_scheme=cdc.ANOMALY_SCHEME_YEARLY_DELTA,
        reference_years="2015-2020",
    )

    assert captured["anomaly_scheme"] == cdc.ANOMALY_SCHEME_YEARLY_DELTA
    assert captured["reference_years"] == "2015-2020"
    ds = saved["ds"]
    assert isinstance(ds, xr.Dataset)
    assert ds.attrs["anomaly_scheme"] == cdc.ANOMALY_SCHEME_YEARLY_DELTA
    assert ds.attrs["reference_years"] == "2015-2020"
