"""Tests for the CLIMATE-48 rerun compile: one member's person-days in production's layout.

The compile reads the per-year most-detailed files of the history and forecast stages (run
on a GBD hierarchy), sums them up the hierarchy, and writes one file per subset hierarchy
(``gbd_2023`` and ``fhs_2023``) laid out like the July 2026 production files: keys
(``location_id``, ``year_id``, ``temperature_zone``) as the index, every one of the 60 zones
on every location-year, and the 800 bins labelled by floats. These tests use a toy hierarchy
and toy per-year files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_data.data import PopulationModelData
from climate_data.special.compile_country_person_days import person_days_year_path
from climate_data.special.compile_member_person_days import (
    KEY_COLUMNS,
    PRODUCTION_YEARS,
    compile_member_person_days,
    compile_member_person_days_main,
    compiled_member_path,
    subset_to_hierarchy,
)
from climate_data.special.history_person_days import (
    INDEX_NAMES,
    TEMPERATURE_COLUMNS,
    TEMPERATURE_ZONE_BINS,
)

SCENARIO = "ssp245"
MEMBER = "CNRM-CM6-1_r4i1p1f2"
HIERARCHY = "gbd_2023"
N_ZONES = 60


def _hierarchy() -> pd.DataFrame:
    """Global 1; country 10 with subnationals 11 and 12; country 20 with no subnationals."""
    rows = [
        (1, 1, 0, 0),
        (10, 1, 1, 0),
        (11, 10, 2, 1),
        (12, 10, 2, 1),
        (20, 1, 1, 1),
    ]
    return pd.DataFrame(
        rows, columns=["location_id", "parent_id", "level", "most_detailed"]
    )


def _subset_hierarchy() -> pd.DataFrame:
    """Like FHS: the same countries, without country 10's subnationals."""
    hierarchy = _hierarchy()
    return hierarchy.loc[~hierarchy["location_id"].isin([11, 12])]


def _year_frame(year: int, rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    """A per-year stage file: rows of (location, zone, bin index, person-days)."""
    index = []
    values = np.zeros((len(rows), len(TEMPERATURE_COLUMNS)))
    for i, (location_id, zone, bin_index, person_days) in enumerate(rows):
        index.append((location_id, year, zone))
        values[i, bin_index] = person_days
    return pd.DataFrame(
        values,
        index=pd.MultiIndex.from_tuples(index, names=INDEX_NAMES),
        columns=TEMPERATURE_COLUMNS,
    )


def _write_years(root: Path, years: list[int]) -> None:
    for year in years:
        df = _year_frame(
            year,
            [
                (11, 10, 400, 1.0 * year),
                (12, 10, 400, 2.0),
                (12, 11, 401, 3.0),
                (20, -25, 0, 7.0),
            ],
        )
        path = person_days_year_path(root, HIERARCHY, SCENARIO, MEMBER, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)


def _write_frame(root: Path, year: int, df: pd.DataFrame) -> None:
    """Write ``df`` as the stage file for ``year``, whatever years it holds."""
    path = person_days_year_path(root, HIERARCHY, SCENARIO, MEMBER, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _compile(root: Path, years: list[int]) -> pd.DataFrame:
    def year_path(year: int) -> Path:
        return person_days_year_path(root, HIERARCHY, SCENARIO, MEMBER, year)

    return compile_member_person_days(_hierarchy(), years, year_path)


def _value(
    df: pd.DataFrame, location_id: int, year: int, zone: int, bin_index: int
) -> float:
    row = np.asarray(df.xs((location_id, year, zone)), dtype=np.float64).ravel()
    return float(row[bin_index])


# --- Paths and constants ---


def test_compiled_member_path_layout(tmp_path: Path) -> None:
    path = compiled_member_path(tmp_path, "fhs_2023", SCENARIO, MEMBER)
    assert path == tmp_path / "compiled" / "fhs_2023" / f"{SCENARIO}_{MEMBER}.parquet"


def test_production_years_are_julys() -> None:
    assert PRODUCTION_YEARS[0] == 1990  # noqa: PLR2004
    assert PRODUCTION_YEARS[-1] == 2100  # noqa: PLR2004
    assert len(PRODUCTION_YEARS) == 111  # noqa: PLR2004


# --- The compile ---


def test_compile_sums_up_every_level(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023, 2024])
    result = _compile(tmp_path, [2023, 2024])

    for year in (2023, 2024):
        assert _value(result, 11, year, 10, 400) == 1.0 * year
        assert _value(result, 10, year, 10, 400) == 1.0 * year + 2.0
        assert _value(result, 10, year, 11, 401) == 3.0  # noqa: PLR2004
        assert _value(result, 20, year, -25, 0) == 7.0  # noqa: PLR2004
        assert _value(result, 1, year, 10, 400) == 1.0 * year + 2.0
        assert _value(result, 1, year, -25, 0) == 7.0  # noqa: PLR2004


def test_compile_conserves_person_days(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])
    result = _compile(tmp_path, [2023])

    detailed_total = 2023.0 + 2.0 + 3.0 + 7.0
    assert float(result.loc[1].to_numpy().sum()) == detailed_total
    countries = result.loc[[10, 20]].to_numpy().sum()
    assert float(countries) == detailed_total


def test_compile_has_every_zone_on_every_location_year(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023, 2024])
    result = _compile(tmp_path, [2023, 2024])

    assert len(result) == 5 * 2 * N_ZONES
    zones = result.index.get_level_values("temperature_zone")
    assert sorted(set(zones)) == TEMPERATURE_ZONE_BINS.tolist()
    # A zone with no person-days is a row of zeros.
    assert np.asarray(result.xs((20, 2023, 0)), dtype=np.float64).sum() == 0.0


def test_compile_layout_matches_production(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])
    result = _compile(tmp_path, [2023])

    assert list(result.index.names) == KEY_COLUMNS
    assert isinstance(result.index, pd.MultiIndex)
    for level in range(3):
        assert result.index.levels[level].dtype == np.int64
    assert list(result.columns) == list(TEMPERATURE_COLUMNS)
    assert result.columns.dtype == np.float64
    assert (result.dtypes == np.float64).all()
    assert result.index.is_monotonic_increasing


def test_compile_reads_only_the_requested_years(tmp_path: Path) -> None:
    _write_years(tmp_path, [2022, 2023, 2024])
    result = _compile(tmp_path, [2023])

    assert set(result.index.get_level_values("year_id")) == {2023}


def test_compile_fails_when_a_year_file_is_missing(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])
    with pytest.raises(FileNotFoundError, match="2024"):
        _compile(tmp_path, [2023, 2024])


def test_compile_keeps_a_location_with_no_person_days_in_a_year(tmp_path: Path) -> None:
    # Location 20 has person-days in 2023 but none in 2024 (no population, for example),
    # so the 2024 stage file has no rows for it. It still gets 60 rows of zeros in 2024.
    _write_years(tmp_path, [2023])
    df = _year_frame(2024, [(11, 10, 400, 1.0), (12, 10, 400, 2.0)])
    _write_frame(tmp_path, 2024, df)

    result = _compile(tmp_path, [2023, 2024])

    assert len(result) == 5 * 2 * N_ZONES
    assert len(result.xs((20, 2024), level=["location_id", "year_id"])) == N_ZONES
    assert (
        float(result.xs((20, 2024), level=["location_id", "year_id"]).to_numpy().sum())
        == 0.0
    )


def test_compile_fails_when_a_year_file_holds_another_year(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])
    df = _year_frame(2023, [(11, 10, 400, 1.0)])
    _write_frame(tmp_path, 2024, df)

    with pytest.raises(ValueError, match="2024"):
        _compile(tmp_path, [2023, 2024])


def test_compile_fails_when_a_year_file_is_empty(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])
    df = _year_frame(2024, [])
    _write_frame(tmp_path, 2024, df)

    with pytest.raises(ValueError, match="2024"):
        _compile(tmp_path, [2023, 2024])


def test_compile_fails_when_no_years_are_given(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="years"):
        _compile(tmp_path, [])


def test_subset_keeps_only_the_subset_locations(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])
    result = _compile(tmp_path, [2023])

    subset = subset_to_hierarchy(result, _subset_hierarchy())

    assert sorted(set(subset.index.get_level_values("location_id"))) == [1, 10, 20]
    pd.testing.assert_frame_equal(subset.loc[[10]], result.loc[[10]])


# --- Main and the written files ---


def _population_model(root: Path) -> Path:
    pm_root = root / "population-model"
    pm_data = PopulationModelData(pm_root)
    gbd_inputs = pm_data.raking_data / "gbd-inputs"
    gbd_inputs.mkdir(parents=True)
    _hierarchy().to_parquet(gbd_inputs / "hierarchy_gbd_2023.parquet")
    _subset_hierarchy().to_parquet(gbd_inputs / "hierarchy_fhs_2023.parquet")
    return pm_root


def test_main_writes_both_subsets_and_they_round_trip(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023, 2024])
    pm_root = _population_model(tmp_path)

    written = compile_member_person_days_main(
        SCENARIO,
        MEMBER,
        HIERARCHY,
        tmp_path,
        pm_root,
        tmp_path,
        years=[2023, 2024],
    )

    assert sorted(written) == ["fhs_2023", "gbd_2023"]
    for subset, expected in written.items():
        read_back = pd.read_parquet(
            compiled_member_path(tmp_path, subset, SCENARIO, MEMBER)
        )
        pd.testing.assert_frame_equal(read_back, expected)
        assert read_back.columns.dtype == np.float64
    gbd_locations = set(written["gbd_2023"].index.get_level_values("location_id"))
    fhs_locations = set(written["fhs_2023"].index.get_level_values("location_id"))
    assert gbd_locations == {1, 10, 11, 12, 20}
    assert fhs_locations == {1, 10, 20}


def test_main_skips_when_both_files_exist(tmp_path: Path) -> None:
    for subset in ("gbd_2023", "fhs_2023"):
        path = compiled_member_path(tmp_path, subset, SCENARIO, MEMBER)
        path.parent.mkdir(parents=True)
        path.write_text("not parquet")

    written = compile_member_person_days_main(
        SCENARIO, MEMBER, HIERARCHY, tmp_path, tmp_path, tmp_path, years=[2023]
    )

    assert written == {}
    path = compiled_member_path(tmp_path, "gbd_2023", SCENARIO, MEMBER)
    assert path.read_text() == "not parquet"


def test_main_writes_only_the_missing_file(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])
    pm_root = _population_model(tmp_path)
    existing = compiled_member_path(tmp_path, "gbd_2023", SCENARIO, MEMBER)
    existing.parent.mkdir(parents=True)
    existing.write_text("not parquet")

    written = compile_member_person_days_main(
        SCENARIO, MEMBER, HIERARCHY, tmp_path, pm_root, tmp_path, years=[2023]
    )

    assert sorted(written) == ["fhs_2023"]
    assert existing.read_text() == "not parquet"
    assert compiled_member_path(tmp_path, "fhs_2023", SCENARIO, MEMBER).exists()


def test_main_overwrite_rewrites_existing_files(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])
    pm_root = _population_model(tmp_path)
    existing = compiled_member_path(tmp_path, "gbd_2023", SCENARIO, MEMBER)
    existing.parent.mkdir(parents=True)
    existing.write_text("not parquet")

    written = compile_member_person_days_main(
        SCENARIO,
        MEMBER,
        HIERARCHY,
        tmp_path,
        pm_root,
        tmp_path,
        years=[2023],
        overwrite=True,
    )

    assert sorted(written) == ["fhs_2023", "gbd_2023"]
    pd.testing.assert_frame_equal(pd.read_parquet(existing), written["gbd_2023"])


def test_main_rejects_an_empty_year_list(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])
    pm_root = _population_model(tmp_path)

    with pytest.raises(ValueError, match="years"):
        compile_member_person_days_main(
            SCENARIO, MEMBER, HIERARCHY, tmp_path, pm_root, tmp_path, years=[]
        )


# --- CLI ---


def test_cli_task_is_registered_and_requires_output_dir() -> None:
    from click.testing import CliRunner

    from climate_data.special import TASK_RUNNERS

    runner = CliRunner()
    command = TASK_RUNNERS["compile_member_person_days"]
    assert runner.invoke(command, ["--help"]).exit_code == 0
    result = runner.invoke(
        command,
        ["--cmip6-experiment", "ssp245", "--gcm-member", "CNRM-CM6-1_r4i1p1f2"],
    )
    assert result.exit_code != 0
    assert "--output-dir" in result.output
