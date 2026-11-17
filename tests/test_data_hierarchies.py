"""Which files `PopulationModelData` reads for each hierarchy name (CLIMATE-36).

`lsae_1578` replaces `lsae_1285` (Tamarugal re-parented in Chile). `lsae_1285` stays
loadable so earlier results remain reproducible. No cluster I/O: the readers are
patched to record the path they were asked for.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest

from climate_data import constants as cdc
from climate_data.data import PopulationModelData

ROOT = Path("unused-root")
GBD_INPUTS = ROOT / "admin-inputs" / "raking" / "gbd-inputs"


def _record_reads(monkeypatch: pytest.MonkeyPatch, module: Any) -> list[Path]:
    """Patch `module.read_parquet` to record its path and return an empty frame."""
    reads: list[Path] = []

    def fake_read_parquet(path: Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
        reads.append(Path(path))
        return pd.DataFrame({"location_id": [], "parent_id": []})

    monkeypatch.setattr(module, "read_parquet", fake_read_parquet)
    return reads


@pytest.mark.parametrize("hierarchy", ["lsae_1285", "lsae_1578"])
def test_lsae_hierarchy_is_a_pixel_hierarchy(hierarchy: str) -> None:
    assert cdc.HIERARCHY_MAP[hierarchy] == [hierarchy]


@pytest.mark.parametrize("hierarchy", ["lsae_1285", "lsae_1578"])
def test_lsae_raking_shapes_read_the_a2_file(
    monkeypatch: pytest.MonkeyPatch, hierarchy: str
) -> None:
    reads = _record_reads(monkeypatch, gpd)

    PopulationModelData(str(ROOT)).load_raking_shapes(hierarchy)

    assert reads == [GBD_INPUTS / f"shapes_{hierarchy}_a2.parquet"]


@pytest.mark.parametrize("hierarchy", ["lsae_1285", "lsae_1578"])
def test_lsae_subset_hierarchy_reads_its_own_file(
    monkeypatch: pytest.MonkeyPatch, hierarchy: str
) -> None:
    reads = _record_reads(monkeypatch, pd)

    PopulationModelData(str(ROOT)).load_subset_hierarchy(hierarchy)

    assert reads == [GBD_INPUTS / f"hierarchy_{hierarchy}.parquet"]


def test_unknown_hierarchy_still_raises() -> None:
    pm_data = PopulationModelData(str(ROOT))
    with pytest.raises(ValueError, match="Unknown pixel hierarchy"):
        pm_data.load_raking_shapes("lsae_9999")
    with pytest.raises(ValueError, match="Unknown admin hierarchy"):
        pm_data.load_subset_hierarchy("lsae_9999")
