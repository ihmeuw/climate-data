"""Tests for the CMIP6 extract encoding and output paths (CLIMATE-29)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_data import constants as cdc
from climate_data.data import ClimateData, gcm_member_id
from climate_data.extract import cmip6
from climate_data.extract.cmip6 import select_members

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
    re-extract would have written `ssp126_pr_<gcm_member>.nc` while `scenario_daily` looks
    up `pr_ssp126_<gcm_member>.nc` -- new files invisible to the pipeline that consumes
    them. Note `gcm_member` is `<source>_<variant>`, not the bare `member_id`; keying on
    the variant alone was a separate bug, covered below.
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
    # Must be the path the extract actually targets, or the unlink under test would miss
    # it and this would pass vacuously.
    out_path = cdata.extracted_cmip6_path(
        "pr", EXPERIMENT, gcm_member_id(SOURCE, MEMBER)
    )
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


# A representable flux: 1e-5 kg m-2 s-1 is 0.864 mm/day, well inside the 1e-6 encoding.
REPRESENTABLE_FLUX = 1e-5
OTHER_SOURCE = "CESM2-WACCM"


def _one_member_of_pr(_zarr_path: str) -> xr.Dataset:
    """A minimal `pr` dataset the encoding guard accepts and `to_netcdf` can write."""
    time = xr.date_range("2015-01-01", periods=2, freq="D", use_cftime=False)
    return xr.Dataset(
        {"pr": (("time", "lat", "lon"), np.full((2, 2, 2), REPRESENTABLE_FLUX))},
        coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )


def test_extract_writes_the_filename_the_generate_stage_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extract's output name must be the one `get_gcms` hands to `scenario_daily`.

    `extract_cmip6_main` iterates the metadata indexed by `member_id`, so it used to key
    the output on the bare variant -- writing `pr_ssp126_r1i1p1f1.nc`. But `get_gcms`
    returns `<source>_<variant>`, and `scenario_daily` feeds that straight into
    `extracted_cmip6_path`, so it looked up `pr_ssp126_<source>_<variant>.nc`: a name
    nothing had written. Every re-extracted file would have been invisible to the stage
    that consumes it.

    Asserted against `get_gcms`'s real output rather than a hard-coded string, so the
    writer and the reader cannot drift apart again.
    """
    cdata = ClimateData(tmp_path)
    _metadata_for_one_member("gs://irrelevant").to_parquet(
        cdata.extracted_cmip6 / "cmip6-metadata.parquet"
    )
    # What `scenario_daily` will ask for: `get_gcms` reads the inclusion metadata, whose
    # index is (model, variant), and joins them.
    inclusion = pd.DataFrame(
        {"pr": [True]},
        index=pd.MultiIndex.from_tuples([(SOURCE, MEMBER)], names=["model", "variant"]),
    )
    inclusion.to_parquet(cdata.results_metadata / "scenario_inclusion_metadata.parquet")
    monkeypatch.setattr(cmip6, "load_cmip_data", _one_member_of_pr)

    cmip6.extract_cmip6_main(
        cmip6_source=SOURCE,
        cmip6_experiment=EXPERIMENT,
        cmip6_variable="pr",
        output_dir=tmp_path,
        overwrite=True,
    )

    written = sorted(p.name for p in cdata.extracted_cmip6.glob("pr_*.nc"))
    expected = cdata.extracted_cmip6_path(
        "pr", EXPERIMENT, cdata.get_gcms(["pr"])[0]
    ).name

    assert written == [expected]
    assert SOURCE in expected


def _metadata_for_three_members() -> pd.DataFrame:
    """Two variants of our source plus a same-variant member of a different source."""
    rows = []
    for source, member in (
        (SOURCE, MEMBER),
        (SOURCE, "r2i1p1f1"),
        (OTHER_SOURCE, MEMBER),
    ):
        rows.append(
            {
                "source_id": source,
                "experiment_id": EXPERIMENT,
                "variable_id": "pr",
                "table_id": "day",
                "member_id": member,
                "zstore": f"gs://{source}/{member}",
            }
        )
    return pd.DataFrame(rows)


def test_select_members_narrows_to_one_member() -> None:
    """`--gcm-member` must isolate a single member so each can be its own job.

    The runner fans out one job per member because member counts are wildly uneven
    (MIROC6 has 50 `pr` members where most sources have one) and because
    `extract_cmip6_main` re-raises, so a member the encoding guard rejects would abandon
    every member behind it in the same job.
    """
    meta = _metadata_for_three_members()

    both = select_members(meta, SOURCE, EXPERIMENT, "pr")
    one = select_members(meta, SOURCE, EXPERIMENT, "pr", gcm_member_id(SOURCE, MEMBER))

    # The other source's identically-named variant must not leak into either result.
    assert sorted(both) == [MEMBER, "r2i1p1f1"]
    assert list(one) == [MEMBER]
    assert one[MEMBER] == f"gs://{SOURCE}/{MEMBER}"


def test_select_members_raises_on_an_unknown_member() -> None:
    """A member that does not exist must fail loudly, not extract nothing.

    The runner and the task enumerate the same space, so a mismatch between them would
    otherwise produce jobs that silently write no file and report success.
    """
    meta = _metadata_for_three_members()

    with pytest.raises(ValueError, match="No CMIP6 member"):
        select_members(meta, SOURCE, EXPERIMENT, "pr", "ACCESS-CM2_r99i1p1f1")


def test_two_sources_sharing_a_member_id_get_distinct_paths() -> None:
    """A CMIP6 `member_id` is unique only within a source, so the key needs both.

    `r1i1p1f1` is shared by 24 of the extracted sources in ssp126 and 30 in ssp585.
    Keyed on the variant alone, 295 extractions collapsed onto 171 filenames -- 124 of
    them overwriting each other, and doing it concurrently at `concurrency_limit=50`.
    """
    cdata = ClimateData("/nonexistent", read_only=True)

    a = cdata.extracted_cmip6_path("pr", EXPERIMENT, gcm_member_id(SOURCE, MEMBER))
    b = cdata.extracted_cmip6_path(
        "pr", EXPERIMENT, gcm_member_id(OTHER_SOURCE, MEMBER)
    )

    assert a != b
    assert a.name == f"pr_{EXPERIMENT}_{SOURCE}_{MEMBER}.nc"
