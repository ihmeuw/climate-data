"""Tests for the month-boundary handling in historical daily generation (CLIMATE-29)."""

from pathlib import Path

import numpy as np
import pandas as pd
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
JOINED_SAMPLES = 2


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


def test_drop_noncore_coords_allows_concat_across_extract_formats() -> None:
    """A look-ahead crossing from an old extract into a new one must still concatenate.

    Extracts pulled through the newer CDS API carry `number` and `expver` coordinates
    that the 1950-2023 extracts lack, and `xr.concat` refuses to join datasets whose
    coordinates differ. This is live at the 2023/2024 seam: closing 31 Dec 2023 reads the
    January 2024 file from the newer GBD-2025 pull.
    """
    time = xr.date_range("2023-12-31T23:00", periods=1, freq="h", use_cftime=False)
    old = xr.Dataset({"value": (("time",), np.zeros(1))}, coords={"time": time})
    # `expver` varies along time in the real extracts (size 744), `number` is scalar.
    new = xr.Dataset(
        {"value": (("time",), np.zeros(1))},
        coords={
            "time": time + np.timedelta64(1, "h"),
            "expver": (("time",), np.array(["0001"])),
            "number": 0,
        },
    )

    with pytest.raises(ValueError, match="expver"):
        xr.concat([old, new], dim="time")

    joined = xr.concat(
        [hd.drop_noncore_coords(old), hd.drop_noncore_coords(new)], dim="time"
    )

    assert set(joined.coords) == {"time"}
    assert joined.sizes["time"] == JOINED_SAMPLES


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


SINGLE_LEVELS = cdc.ERA5_DATASETS.reanalysis_era5_single_levels
PRECIP = cdc.ERA5_VARIABLES.total_precipitation
# Longitudes as the extracts store them, 0..360, before `load_and_shift_longitude`.
STORED_LONGITUDES = (0.0, 0.25)
STORED_LATITUDES = (1.0, 0.75)


def _write_single_levels_month(
    cdata: ClimateData,
    year: str,
    month: str,
    first_stamp: str,
    n_hours: int = 3,
    longitudes: tuple[float, ...] = STORED_LONGITUDES,
) -> None:
    """Write the smallest file `load_variable` will accept for the single-level dataset.

    Only the single-level branch is usable at this size: the ERA5-Land branch overwrites
    its coordinates from `cdc.ERA5_LAND_*`, which would demand the full 1801x3600 grid.
    """
    time = xr.date_range(first_stamp, periods=n_hours, freq="h", use_cftime=False)
    shape = (n_hours, len(STORED_LATITUDES), len(longitudes))
    ds = xr.Dataset(
        {"tp": (("time", "latitude", "longitude"), np.zeros(shape))},
        coords={
            "time": time,
            "latitude": list(STORED_LATITUDES),
            "longitude": list(longitudes),
        },
    )
    ds.to_netcdf(cdata.extracted_era5_path(SINGLE_LEVELS, PRECIP, year, month))


def test_lookahead_must_open_on_midnight(tmp_path: Path) -> None:
    """A look-ahead whose first sample is not midnight does not close the final day.

    `.isel(time=[0])` takes whatever the next month opens with and treats it as the
    closing sample of this month's last day. That assumption is not free: ERA5-Land
    `1950_01` opens at 01:00, so a file of that shape exists in the archive. Silently
    accepting it leaves the final day on its 23-hour partial -- one bad day per month,
    invisible in the output.
    """
    cdata = ClimateData(tmp_path)
    _write_single_levels_month(cdata, "2020", "01", "2020-01-31T21:00")
    _write_single_levels_month(cdata, "2020", "02", "2020-02-01T01:00")

    with pytest.raises(ValueError, match="not midnight"):
        hd.load_variable_with_lookahead(cdata, PRECIP, "2020", "01", SINGLE_LEVELS)


def test_lookahead_on_a_different_grid_raises(tmp_path: Path) -> None:
    """A look-ahead on another grid must fail, not NaN-pad onto a union grid.

    `xr.concat` defaults to `join="outer"`, which would quietly widen both months to the
    union of their coordinates and fill the difference with NaN -- and NaNs in the land
    field are then filled from the single-level field, so the damage would not surface in
    `validate_output`. `join="exact"` makes the mismatch loud.
    """
    cdata = ClimateData(tmp_path)
    _write_single_levels_month(cdata, "2020", "01", "2020-01-31T21:00")
    _write_single_levels_month(
        cdata, "2020", "02", "2020-02-01T00:00", longitudes=(0.0, 0.5)
    )

    with pytest.raises(ValueError, match="align|exact|index"):
        hd.load_variable_with_lookahead(cdata, PRECIP, "2020", "01", SINGLE_LEVELS)


def test_lookahead_on_a_matching_grid_joins(tmp_path: Path) -> None:
    """The guards must not reject the ordinary case they are wrapped around."""
    cdata = ClimateData(tmp_path)
    _write_single_levels_month(cdata, "2020", "01", "2020-01-31T21:00")
    _write_single_levels_month(cdata, "2020", "02", "2020-02-01T00:00")

    joined = hd.load_variable_with_lookahead(cdata, PRECIP, "2020", "01", SINGLE_LEVELS)

    assert joined.sizes["time"] == 3 + 1
    assert pd.Timestamp(joined.time.to_index()[-1]) == pd.Timestamp("2020-02-01T00:00")
