"""Tests for the `atomic` option of `save_parquet` and `save_xarray` (CLIMATE-45).

With `atomic=True` a file appears at the target only once it is complete, so a job
killed mid-write cannot leave a partial file that a later run counts as done. The
default (`atomic=False`) keeps today's behavior: touch the target, then write to it.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_data import data as cdd

ENCODING: dict[str, Any] = {"scale_factor": 0.01, "add_offset": 0.0}


class WriteFailedError(RuntimeError):
    """Raised by the stub writers below, part way through a write."""


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"location_id": [1, 2, 3], "value": [0.5, 1.5, 2.5]})


def _dataset() -> xr.Dataset:
    values = np.array([[1.25, 2.5], [3.75, 5.0]])
    return xr.Dataset(
        {"value": (("latitude", "longitude"), values)},
        coords={"latitude": [0.05, 0.15], "longitude": [10.05, 10.15]},
    )


def _failing_to_parquet(self: pd.DataFrame, path: Any, *_: Any, **__: Any) -> None:
    Path(path).write_bytes(b"PAR1 partial")
    raise WriteFailedError


def _failing_to_netcdf(self: xr.Dataset, path: Any, *_: Any, **__: Any) -> None:
    Path(path).write_bytes(b"CDF partial")
    raise WriteFailedError


def _names(directory: Path) -> list[str]:
    names = []
    for path in directory.iterdir():
        names.append(path.name)
    return sorted(names)


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_atomic_parquet_writes_the_same_file_and_mode_as_a_plain_write(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain.parquet"
    atomic = tmp_path / "atomic.parquet"
    cdd.save_parquet(_frame(), plain)
    cdd.save_parquet(_frame(), atomic, atomic=True)

    pd.testing.assert_frame_equal(pd.read_parquet(atomic), pd.read_parquet(plain))
    assert _mode(atomic) == _mode(plain)
    assert _names(tmp_path) == [
        "atomic.parquet",
        "plain.parquet",
    ]


def test_atomic_xarray_writes_the_same_file_and_mode_as_a_plain_write(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain.nc"
    atomic = tmp_path / "atomic.nc"
    cdd.save_xarray(_dataset(), plain, ENCODING)
    cdd.save_xarray(_dataset(), atomic, ENCODING, atomic=True)

    with xr.open_dataset(plain) as p, xr.open_dataset(atomic) as a:
        xr.testing.assert_identical(a.load(), p.load())
        assert a["value"].encoding["dtype"] == p["value"].encoding["dtype"]
    assert _mode(atomic) == _mode(plain)
    assert _names(tmp_path) == ["atomic.nc", "plain.nc"]


def test_atomic_parquet_write_that_fails_leaves_no_file_behind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _failing_to_parquet)
    target = tmp_path / "year.parquet"

    with pytest.raises(WriteFailedError):
        cdd.save_parquet(_frame(), target, atomic=True)

    assert _names(tmp_path) == []


def test_atomic_xarray_write_that_fails_leaves_no_file_behind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xr.Dataset, "to_netcdf", _failing_to_netcdf)
    target = tmp_path / "year.nc"

    with pytest.raises(WriteFailedError):
        cdd.save_xarray(_dataset(), target, ENCODING, atomic=True)

    assert _names(tmp_path) == []


def test_atomic_overwrite_that_fails_keeps_the_old_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "year.parquet"
    cdd.save_parquet(_frame(), target)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _failing_to_parquet)

    with pytest.raises(WriteFailedError):
        cdd.save_parquet(_frame().iloc[:1], target, atomic=True)

    monkeypatch.undo()
    pd.testing.assert_frame_equal(pd.read_parquet(target), _frame())
    assert _names(tmp_path) == ["year.parquet"]


def test_atomic_write_replaces_an_existing_file_and_a_stale_temporary_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "year.parquet"
    cdd.save_parquet(_frame(), target)
    (tmp_path / "year.parquet.tmp").write_bytes(b"left by a killed job")

    cdd.save_parquet(_frame().iloc[:1], target, atomic=True)

    pd.testing.assert_frame_equal(pd.read_parquet(target), _frame().iloc[:1])
    assert _names(tmp_path) == ["year.parquet"]


def test_default_write_is_unchanged_and_leaves_a_partial_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Characterizes today's behavior, which the production stages keep.
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _failing_to_parquet)
    target = tmp_path / "year.parquet"

    with pytest.raises(WriteFailedError):
        cdd.save_parquet(_frame(), target)

    assert target.read_bytes() == b"PAR1 partial"
    assert _names(tmp_path) == ["year.parquet"]
