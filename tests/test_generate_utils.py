"""Tests for the daily collapse of cumulative ERA5-Land variables (CLIMATE-29)."""

import datetime
import typing

import numpy as np
import xarray as xr

from climate_data.generate import utils

# Day 0's completed total, its 23-hour partial, and the same for day 1. The partials sit
# just under the totals because some rain always falls in the final hour -- that gap is
# what a bucket ending at hour 23 loses.
YESTERDAY_TOTAL = 9.0
DAY0_TOTAL = 2.0
DAY0_PARTIAL = 1.5
DAY1_TOTAL = 6.0
DAY1_PARTIAL = 5.0

DAY0 = datetime.date(2020, 1, 1)
DAY1 = datetime.date(2020, 1, 2)
HOURS_PER_DAY = 24


def _minimal_hourly_precip() -> xr.Dataset:
    """Build the smallest cube following the ERA5-Land accumulation convention.

    One pixel, two days. ERA5-Land ``total_precipitation`` accumulates since 00Z, so
    within a day the value rises; the day's closing total is the step-24 sample, which
    CDS stamps 00:00 of the *next* day. Hour 00 of a day therefore holds the *previous*
    day's completed total.

    The cube runs 00:00..23:00 for both days plus one trailing 00:00 sample -- the
    look-ahead that closes day 1, and which in production lives in the next month's file.
    """
    totals = (DAY0_TOTAL, DAY1_TOTAL)
    partials = (DAY0_PARTIAL, DAY1_PARTIAL)
    n_time = len(totals) * HOURS_PER_DAY + 1
    hourly = np.zeros((n_time, 1, 1), dtype="float64")

    for day in range(len(totals)):
        # Hour 00 holds the previous day's completed total.
        previous = YESTERDAY_TOTAL if day == 0 else totals[day - 1]
        hourly[day * HOURS_PER_DAY] = previous
        # Hours 01..23 rise monotonically to the 23-hour partial.
        for hour in range(1, HOURS_PER_DAY):
            hourly[day * HOURS_PER_DAY + hour] = (
                partials[day] * hour / (HOURS_PER_DAY - 1)
            )
    # The look-ahead sample: day 1's closing total.
    hourly[-1] = totals[-1]

    time = xr.date_range(str(DAY0), periods=n_time, freq="h", use_cftime=False)
    return xr.Dataset(
        {"value": (("time", "latitude", "longitude"), hourly)},
        coords={"time": time, "latitude": [0.0], "longitude": [0.0]},
    )


def _daily_value(collapsed: xr.Dataset, date: datetime.date) -> float:
    """Pull the single pixel's collapsed value for one date.

    The two collapses label the ``date`` axis differently: ``groupby("time.date")``
    yields object-dtype ``datetime.date``, while ``resample`` yields ``datetime64``.
    Match whichever this dataset carries.
    """
    key: typing.Any = date
    if np.issubdtype(collapsed.date.dtype, np.datetime64):
        key = np.datetime64(date)
    return float(collapsed.value.sel(date=key).isel(latitude=0, longitude=0))


def test_daily_max_returns_previous_day_total() -> None:
    """Document why ``daily_accumulation_last`` exists: ``daily_max`` returns yesterday's
    rain.

    ``groupby("time.date").max()`` buckets hours 00..23, which holds the previous day's
    completed total (hour 00) and excludes today's (stamped 00:00 tomorrow). Whenever
    yesterday was wetter than today's partial, the bucket maximum *is* yesterday's total.
    This is the 1.5-1.6x wet bias in the 1950-2023 historical product.
    """
    collapsed = utils.daily_max(_minimal_hourly_precip())

    assert _daily_value(collapsed, DAY0) == YESTERDAY_TOTAL
    assert _daily_value(collapsed, DAY0) != DAY0_TOTAL


def test_daily_accumulation_last_returns_the_closing_sample() -> None:
    """Treating the timestamps as hour-ending intervals recovers the true daily total.

    ``resample(closed="right", label="left")`` bins on ``(D 00:00, D+1 00:00]`` labelled
    ``D``, i.e. one whole accumulation window per day. ``last`` rather than ``max``
    because int16 packing can make the window tick down in its final step, in which case
    the maximum is an earlier sample -- exact for 100% of real land day-pixels measured,
    against 99.9864% for max.
    """
    collapsed = utils.daily_accumulation_last(_minimal_hourly_precip())

    assert _daily_value(collapsed, DAY0) == DAY0_TOTAL
    assert _daily_value(collapsed, DAY1) == DAY1_TOTAL


def test_daily_accumulation_last_nans_a_day_missing_its_closing_sample() -> None:
    """An absent closing sample must not silently become the 23-hour partial.

    ``Resample.last`` defaults to ``skipna=True``, which steps back past a missing
    closing sample to hour 23 -- reintroducing, one pixel at a time, exactly the
    incomplete window this collapse exists to eliminate. ``skipna=False`` yields NaN
    instead.

    NaN is the honest answer, not a loud one: ``generate_historical_daily_main`` fills
    ERA5-Land NaNs from the interpolated single-level field, which is how ocean pixels
    are supplied, so this pixel ends up carrying a complete 0.25 degree value rather
    than a truncated 0.1 degree one. Neither setting reaches ``validate_output``.
    """
    ds = _minimal_hourly_precip()
    # Drop the sample that closes day 0: 00:00 of day 1, at hour 24 of the cube.
    ds["value"][HOURS_PER_DAY] = np.nan

    collapsed = utils.daily_accumulation_last(ds)

    assert np.isnan(_daily_value(collapsed, DAY0))
    assert _daily_value(collapsed, DAY0) != DAY0_PARTIAL
    # Day 1 still closes on its own look-ahead sample.
    assert _daily_value(collapsed, DAY1) == DAY1_TOTAL
