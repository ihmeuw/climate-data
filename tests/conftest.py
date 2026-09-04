"""Test-suite-wide fixtures.

Keeps the suite hermetic against the project's shared-storage roots. See
`_isolate_shared_storage_roots` for why this is not optional.
"""

from collections.abc import Iterator

import pytest

from climate_data import constants as cdc


@pytest.fixture(autouse=True, scope="session")
def _isolate_shared_storage_roots(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Point the module-level storage roots at a temporary directory.

    Without this, six tests that construct `ClimateData(tmp_path)` still write to
    production storage. `ClimateData.__init__` calls `_create_model_root()` unless
    `read_only=True`, and that mkdirs `self.raw_daily_results` -- which returns
    `cdc.AGGREGATE_ROOT / "erf-scratch"`, ignoring the root it was handed. On the
    cluster the directory already exists so the tests pass; on a GitHub runner they
    fail with `FileNotFoundError` on `/mnt/team/rapidresponse/...`.

    Both roots are patched, not just the one that bites today, so the suite is
    hermetic against the shared roots generally rather than against one path.

    Note this only works for attributes read at *call* time, as
    `ClimateData.raw_daily_results` does. Constructor defaults such as
    `root: str | Path = cdc.MODEL_ROOT` bind at import time and are unaffected --
    pass roots explicitly rather than relying on this fixture to redirect them.

    The replacement roots are created, not just named: `mkdir` defaults to
    `parents=False`, so pointing `AGGREGATE_ROOT` at a directory that does not exist
    yet reproduces the same `FileNotFoundError` one level down.
    """
    tmp_root = tmp_path_factory.mktemp("shared-storage-roots")
    model_root = tmp_root / "model"
    aggregate_root = tmp_root / "aggregate"
    model_root.mkdir(parents=True)
    aggregate_root.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cdc, "MODEL_ROOT", model_root)
        mp.setattr(cdc, "AGGREGATE_ROOT", aggregate_root)
        yield
