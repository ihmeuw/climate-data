"""Tests for the CMIP6 extract encoding and output paths (CLIMATE-29)."""

from pathlib import Path

import pytest

from climate_data import constants as cdc
from climate_data.data import ClimateData
from climate_data.extract import cmip6

# A daily rainfall well above anything a GCM produces, but not absurd: the wettest
# observed daily totals on Earth are ~1800 mm. The encoding must represent this without
# overflowing int16.
EXTREME_RAINFALL_MM_PER_DAY = 500.0
SECONDS_PER_DAY = 86400
INT16_MAX = 32767


def test_pr_encoding_represents_realistic_extremes() -> None:
    """The declared `pr` encoding must not overflow on plausible rainfall.

    CMIP6 `pr` is a flux in kg m-2 s-1. At the original `scale_factor` of 1e-9 the int16
    ceiling was 32767e-9 kg m-2 s-1, i.e. only 2.83 mm/day, so anything wetter wrapped
    modulo 65536 and decoded as garbage -- including negative precipitation. All 295
    extracted files carry that encoding: 26.4% of sampled cells corrupted, 12.4%
    negative.
    """
    pr = cdc.CMIP6_VARIABLES.get("pr")
    flux = EXTREME_RAINFALL_MM_PER_DAY / SECONDS_PER_DAY
    stored = (flux - pr.encoding_offset) / pr.encoding_scale

    assert abs(stored) <= INT16_MAX


def test_extracted_cmip6_path_argument_order(tmp_path: Path) -> None:
    """Pin the (variable, experiment, gcm_member) order the consumer relies on.

    `extract_cmip6_main` passed (experiment, variable, member) into this signature, so a
    re-extract would have written `ssp126_pr_<member>.nc` while `scenario_daily` looks up
    `pr_ssp126_<member>.nc` -- new files invisible to the pipeline that consumes them.
    """
    cdata = ClimateData(tmp_path, read_only=True)

    path = cdata.extracted_cmip6_path("pr", "ssp126", "ACCESS-CM2_r1i1p1f1")

    assert path.name == "pr_ssp126_ACCESS-CM2_r1i1p1f1.nc"


def test_encoding_overflow_guard_rejects_unrepresentable_values() -> None:
    """A value the encoding cannot hold must raise rather than silently wrap."""
    unrepresentable = (INT16_MAX + 1) * 1e-6

    with pytest.raises(ValueError, match="cannot be represented"):
        cmip6.check_encoding_covers(
            data_min=0.0,
            data_max=unrepresentable,
            offset=0.0,
            scale=1e-6,
            variable="pr",
        )
