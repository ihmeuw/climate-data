"""Tests for the month-boundary handling in historical daily generation (CLIMATE-29)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from climate_data import constants as cdc
from climate_data.data import ClimateData
from climate_data.generate import historical_daily as hd
from climate_data.generate import utils

# February 2020: a real, short, leap month. Using a whole month matters -- the collapse
# puts a bin at each end that belongs to the neighbouring month, and a truncated
# stand-in would place the trailing one inside the target month instead.
YEAR = "2020"
MONTH = 2
DAYS_IN_MONTH = 29
HOURS_PER_DAY = 24
PREVIOUS_MONTH_TOTAL = 99.0
PARTIAL_SHORTFALL = 0.5


def _daily_total(day: int) -> float:
    """A distinct total per day, so any misalignment shows up as an off-by-one."""
    return float(day + 1)


def _month_with_lookahead() -> xr.Dataset:
    """Hourly samples for the whole month plus the one sample that closes its last day.

    Follows the ERA5 accumulation convention: hours 01..23 of a day rise toward its
    total, hour 00 holds the *previous* day's completed total, and the day's own total is
    stamped 00:00 of the next day.
    """
    n_time = DAYS_IN_MONTH * HOURS_PER_DAY + 1
    hourly = np.zeros((n_time, 1, 1), dtype="float64")
    for day in range(DAYS_IN_MONTH):
        previous = PREVIOUS_MONTH_TOTAL if day == 0 else _daily_total(day - 1)
        hourly[day * HOURS_PER_DAY] = previous
        partial = _daily_total(day) - PARTIAL_SHORTFALL
        for hour in range(1, HOURS_PER_DAY):
            hourly[day * HOURS_PER_DAY + hour] = partial * hour / (HOURS_PER_DAY - 1)
    # The look-ahead: the final day's total, stamped 00:00 of the next month.
    hourly[-1] = _daily_total(DAYS_IN_MONTH - 1)

    time = xr.date_range(
        f"{YEAR}-{MONTH:02d}-01", periods=n_time, freq="h", use_cftime=False
    )
    return xr.Dataset(
        {"value": (("time", "latitude", "longitude"), hourly)},
        coords={"time": time, "latitude": [0.0], "longitude": [0.0]},
    )


def test_trim_keeps_only_the_target_month_and_closes_its_last_day() -> None:
    """The collapse puts a bin outside the month at each end; the trim removes both.

    Binning on ``(D 00:00, D+1 00:00]`` labels the leading bin with the *previous*
    month's last day, since it holds that day's closing sample, and appends a trailing
    empty bin labelled with the next month. Left in place, every month contributes a
    duplicate date and a NaN at its seams, and ``validate_output`` rejects the year on
    both day count and NaNs. Meanwhile the final day is closed by the look-ahead sample
    rather than left at its 23-hour partial.

    Covers the transformation only -- reading the look-ahead out of the next month's file
    is exercised by the sandbox validation run, not here.
    """
    collapsed = utils.daily_accumulation_last(_month_with_lookahead())
    trimmed = hd.trim_to_month(collapsed, YEAR, MONTH)

    # One bin for the previous month's last day, one empty bin for the next month.
    assert collapsed.sizes["date"] == DAYS_IN_MONTH + 2
    assert trimmed.sizes["date"] == DAYS_IN_MONTH

    dates = trimmed.date.to_index()
    assert set(dates.month) == {MONTH}
    assert set(dates.day) == set(range(1, DAYS_IN_MONTH + 1))

    expected = []
    for day in range(DAYS_IN_MONTH):
        expected.append(_daily_total(day))
    values = trimmed.value.isel(latitude=0, longitude=0).to_numpy()
    assert values == pytest.approx(expected)


def test_missing_lookahead_raises_and_december_looks_into_january(
    tmp_path: Path,
) -> None:
    """December needs the next *year's* January, and an absent file must be loud.

    ERA5 extracts stop at 2023, so regenerating 2023 requires a January 2024 file that
    does not exist. Failing loudly is deliberate: the alternative is a silent fallback to
    the 23-hour partial, and a warning buried in a 74-job cluster log is how CLIMATE-23
    went unnoticed.
    """
    cdata = ClimateData(tmp_path, read_only=True)

    with pytest.raises(FileNotFoundError, match="2024_01"):
        hd.load_variable_with_lookahead(
            cdata,
            cdc.ERA5_VARIABLES.total_precipitation,
            "2023",
            "12",
            cdc.ERA5_DATASETS.reanalysis_era5_land,
        )
