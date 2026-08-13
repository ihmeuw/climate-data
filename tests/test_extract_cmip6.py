"""Tests for the CMIP6 extract encoding and output paths (CLIMATE-29)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

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


SOURCE = "ACCESS-CM2"
EXPERIMENT = "ssp126"
MEMBER = "r1i1p1f1"
# A flux of 1 kg m-2 s-1 is 86400 mm/day -- far past the 2831 mm/day ceiling, so the
# encoding guard rejects it before anything is written.
UNREPRESENTABLE_FLUX = 1.0
EXISTING_CONTENT = b"the previous extract"


def _metadata_for_one_member(zstore: str) -> pd.DataFrame:
    """The single metadata row `extract_cmip6_main` selects on."""
    return pd.DataFrame(
        {
            "source_id": [SOURCE],
            "experiment_id": [EXPERIMENT],
            "variable_id": ["pr"],
            "table_id": ["day"],
            "member_id": [MEMBER],
            "zstore": [zstore],
        }
    )


def test_failed_guard_leaves_the_existing_extract_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected extract must not delete the file it failed to replace.

    The encoding guard runs *before* `shell_tools.touch`, so when it raises nothing has
    been written and the file on disk is still the previous extract. The failure handler
    used to unlink unconditionally, which turns "this GCM is too wet to encode" into
    "the old file is gone and no new one exists" -- on shared storage, mid-way through
    re-extracting the 295 corrupt `pr` files with `--overwrite`.
    """
    cdata = ClimateData(tmp_path)
    _metadata_for_one_member("gs://irrelevant").to_parquet(
        cdata.extracted_cmip6 / "cmip6-metadata.parquet"
    )
    out_path = cdata.extracted_cmip6_path("pr", EXPERIMENT, MEMBER)
    out_path.write_bytes(EXISTING_CONTENT)

    def _too_wet_to_encode(_zarr_path: str) -> xr.Dataset:
        return xr.Dataset({"pr": (("time",), np.array([UNREPRESENTABLE_FLUX]))})

    monkeypatch.setattr(cmip6, "load_cmip_data", _too_wet_to_encode)

    with pytest.raises(ValueError, match="cannot be represented"):
        cmip6.extract_cmip6_main(
            cmip6_source=SOURCE,
            cmip6_experiment=EXPERIMENT,
            cmip6_variable="pr",
            output_dir=tmp_path,
            overwrite=True,
        )

    assert out_path.exists()
    assert out_path.read_bytes() == EXISTING_CONTENT
