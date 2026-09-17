"""Tests for the annual runner's anomaly-scheme handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_data import constants as cdc
from climate_data.generate import scenario_annual

if TYPE_CHECKING:
    from pathlib import Path


def test_forecast_jobs_filter_is_a_passthrough_for_monthly_unbounded() -> None:
    """`monthly` constrains nothing -- except value-bounded variables, tested below."""
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


def test_forecast_jobs_filter_drops_bounded_forecasts_under_the_default_scheme() -> (
    None
):
    """Regression for review finding 1: these tasks used to be queued and then die.

    The annual stage calls `generate_scenario_daily_main` in memory, so a bounded forecast
    job reaches `check_bounded_variable_scheme` and raises in the worker -- with
    `max_attempts=1`, after the whole fan-out has been scheduled.
    """
    jobs = [
        ("relative_humidity", "historical", "2020", "m1"),
        ("relative_humidity", "ssp245", "2030", "m1"),
        ("mean_temperature", "ssp245", "2030", "m1"),
    ]
    got = scenario_annual.forecast_jobs_for_anomaly_scheme(
        jobs, cdc.ANOMALY_SCHEME_MONTHLY
    )
    # historical reads the daily results off disk, so the scheme does not constrain it
    assert got == [jobs[0], jobs[2]]


def test_forecast_jobs_filter_keeps_bounded_forecasts_under_the_plain_ratio() -> None:
    jobs = [
        ("relative_humidity", "historical", "2020", "m1"),
        ("relative_humidity", "ssp245", "2030", "m1"),
    ]
    got = scenario_annual.forecast_jobs_for_anomaly_scheme(
        jobs, cdc.ANOMALY_SCHEME_MONTHLY_RATIO
    )
    assert got == jobs


def test_forecast_jobs_filter_errors_when_nothing_is_runnable() -> None:
    """An empty fan-out looks like success, so asking for the impossible is an error."""
    jobs = [("relative_humidity", "ssp245", "2030", "m1")]
    with pytest.raises(click.UsageError, match="nothing to submit"):
        scenario_annual.forecast_jobs_for_anomaly_scheme(
            jobs, cdc.ANOMALY_SCHEME_MONTHLY
        )


def test_nothing_runnable_error_names_the_scheme_to_use() -> None:
    jobs = [("relative_humidity", "ssp245", "2030", "m1")]
    with pytest.raises(click.UsageError, match=cdc.ANOMALY_SCHEME_MONTHLY_RATIO):
        scenario_annual.forecast_jobs_for_anomaly_scheme(
            jobs, cdc.ANOMALY_SCHEME_MONTHLY
        )


def test_bounded_source_variables_resolves_through_the_transform_map() -> None:
    """`cdc.VALUE_BOUNDS` is keyed on the daily variable, not the annual target."""
    assert scenario_annual.bounded_source_variables(
        "relative_humidity", cdc.ANOMALY_SCHEME_MONTHLY
    ) == ["relative_humidity"]


def test_bounded_source_variables_is_empty_under_a_bounded_scheme() -> None:
    assert (
        scenario_annual.bounded_source_variables(
            "relative_humidity", cdc.ANOMALY_SCHEME_MONTHLY_RATIO
        )
        == []
    )


def test_bounded_source_variables_is_empty_for_an_unbounded_target() -> None:
    assert (
        scenario_annual.bounded_source_variables(
            "mean_temperature", cdc.ANOMALY_SCHEME_MONTHLY
        )
        == []
    )


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
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_YEARLY_DELTA,
        reference_years="2015-2020",
    )

    assert captured["anomaly_scheme"] == cdc.ANOMALY_SCHEME_YEARLY_DELTA
    assert captured["reference_years"] == "2015-2020"
    ds = saved["ds"]
    assert isinstance(ds, xr.Dataset)
    assert ds.attrs["anomaly_scheme"] == cdc.ANOMALY_SCHEME_YEARLY_DELTA
    assert ds.attrs["reference_years"] == "2015-2020"
