"""Tests for the CLIMATE-45 compile stage: one country's person-days, all levels, all years.

The compile reads the per-year admin2 files of the history and forecast stages, keeps one
country's admin2 rows, sums them up the hierarchy to the country, and labels the rows. For
the US it adds Census GEOIDs from USHD's crosswalk. These tests use a toy hierarchy and toy
per-year files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_data.data import PopulationModelData
from climate_data.special.compile_country_person_days import (
    compile_country_person_days,
    compiled_country_path,
    country_admin2_ids,
    country_locations,
    insert_geoid,
    load_us_geoids,
    person_days_year_path,
)
from climate_data.special.forecast_person_days import forecast_person_days_path
from climate_data.special.history_person_days import (
    INDEX_NAMES,
    TEMPERATURE_COLUMNS,
    history_person_days_path,
)

SCENARIO = "ssp245"
MEMBER = "CNRM-CM6-1_r4i1p1f2"
WEIGHTS_HIERARCHY = "lsae_1285"
US = 102


def _hierarchy() -> pd.DataFrame:
    """Global 1; the US 102 with states 523 (counties 1001, 1002) and 524 (county 1003);
    another country 200 with one admin1 300 and its admin2 2001."""
    rows = [
        (1, 1, 0, "1", 0, "Global"),
        (US, 1, 1, "1,102", 0, "United States"),
        (523, US, 2, "1,102,523", 0, "Alabama"),
        (524, US, 2, "1,102,524", 0, "Alaska"),
        (1001, 523, 3, "1,102,523,1001", 1, "Autauga"),
        (1002, 523, 3, "1,102,523,1002", 1, "Baldwin"),
        (1003, 524, 3, "1,102,524,1003", 1, "Anchorage"),
        (200, 1, 1, "1,200", 0, "Elsewhere"),
        (300, 200, 2, "1,200,300", 0, "Elsewhere admin1"),
        (2001, 300, 3, "1,200,300,2001", 1, "Elsewhere admin2"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "location_id",
            "parent_id",
            "level",
            "path_to_top_parent",
            "most_detailed",
            "location_name",
        ],
    )


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
                (1001, 10, 400, 1.0 * year),
                (1002, 10, 400, 2.0),
                (1002, 11, 401, 3.0),
                # Only county 1003 has zone 12.
                (1003, 12, 402, 5.0),
                (2001, 10, 400, 100.0),
            ],
        )
        path = person_days_year_path(root, WEIGHTS_HIERARCHY, SCENARIO, MEMBER, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)


def _compile(
    root: Path, years: list[int], geoids: pd.Series | None = None
) -> pd.DataFrame:
    def year_path(year: int) -> Path:
        return person_days_year_path(root, WEIGHTS_HIERARCHY, SCENARIO, MEMBER, year)

    return compile_country_person_days(_hierarchy(), US, years, year_path, geoids)


# --- Paths and the hierarchy ---


def test_person_days_year_path_switches_to_forecast_in_2024(tmp_path: Path) -> None:
    assert person_days_year_path(
        tmp_path, WEIGHTS_HIERARCHY, SCENARIO, MEMBER, 2023
    ) == history_person_days_path(tmp_path, WEIGHTS_HIERARCHY, 2023)
    assert person_days_year_path(
        tmp_path, WEIGHTS_HIERARCHY, SCENARIO, MEMBER, 2024
    ) == forecast_person_days_path(tmp_path, WEIGHTS_HIERARCHY, SCENARIO, MEMBER, 2024)


def test_compiled_country_path_layout(tmp_path: Path) -> None:
    assert compiled_country_path(tmp_path, "lsae_1578", US, SCENARIO, MEMBER) == (
        tmp_path / "compiled" / "lsae_1578" / "102" / f"{SCENARIO}_{MEMBER}.parquet"
    )


def test_load_subset_hierarchy_allows_lsae_1578(tmp_path: Path) -> None:
    pm_data = PopulationModelData(tmp_path)
    path = pm_data.raking_data / "gbd-inputs" / "hierarchy_lsae_1578.parquet"
    path.parent.mkdir(parents=True)
    _hierarchy().to_parquet(path)

    loaded = pm_data.load_subset_hierarchy("lsae_1578")

    pd.testing.assert_frame_equal(loaded, _hierarchy())


def test_country_locations_are_the_country_and_its_descendants() -> None:
    locations = country_locations(_hierarchy(), US)

    assert sorted(locations["location_id"]) == [102, 523, 524, 1001, 1002, 1003]


def test_country_admin2_ids_are_the_most_detailed_locations() -> None:
    assert country_admin2_ids(_hierarchy(), US) == [1001, 1002, 1003]


def test_country_locations_rejects_an_unknown_country() -> None:
    with pytest.raises(ValueError, match="999"):
        country_locations(_hierarchy(), 999)


# --- The compile ---


def test_compile_sums_counties_to_states_and_the_country(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023, 2024])

    result = _compile(tmp_path, [2023, 2024])

    by_key = result.set_index(["location_id", "year_id", "temperature_zone"])
    bin_400 = TEMPERATURE_COLUMNS[400]
    assert by_key.loc[(523, 2023, 10), bin_400] == 2023.0 + 2.0
    assert by_key.loc[(US, 2024, 10), bin_400] == 2024.0 + 2.0
    assert by_key.loc[(US, 2024, 11), TEMPERATURE_COLUMNS[401]] == 3.0  # noqa: PLR2004
    # Every year: the country equals the sum of its counties.
    counties = result[result["level"] == "county"]
    country = result[result["location_id"] == US]
    np.testing.assert_allclose(
        country.groupby("year_id")[list(TEMPERATURE_COLUMNS)].sum().to_numpy(),
        counties.groupby("year_id")[list(TEMPERATURE_COLUMNS)].sum().to_numpy(),
    )


def test_compile_keeps_only_the_country(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])

    result = _compile(tmp_path, [2023])

    assert set(result["location_id"]) == {102, 523, 524, 1001, 1002, 1003}


def test_compile_carries_a_zone_from_one_county_up(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])

    result = _compile(tmp_path, [2023])

    zone_12 = result[result["temperature_zone"] == 12]  # noqa: PLR2004
    assert sorted(zone_12["location_id"]) == [102, 524, 1003]
    # And no state gets a zone none of its counties has.
    assert 12 not in set(result.loc[result["location_id"] == 523, "temperature_zone"])  # noqa: PLR2004


def test_compile_drops_rows_with_no_person_days(tmp_path: Path) -> None:
    df = _year_frame(2023, [(1001, 10, 400, 1.0), (1002, 13, 400, 0.0)])
    path = person_days_year_path(tmp_path, WEIGHTS_HIERARCHY, SCENARIO, MEMBER, 2023)
    path.parent.mkdir(parents=True)
    df.to_parquet(path)

    result = _compile(tmp_path, [2023])

    assert 13 not in set(result["temperature_zone"])  # noqa: PLR2004
    assert (result[list(TEMPERATURE_COLUMNS)].sum(axis=1) > 0).all()


def test_compile_fails_when_a_year_file_is_missing(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023, 2025])

    with pytest.raises(FileNotFoundError, match="2024"):
        _compile(tmp_path, [2023, 2024, 2025])


def test_compile_layout(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023, 2024])

    result = _compile(tmp_path, [2023, 2024])

    assert list(result.columns[:5]) == [
        "location_id",
        "level",
        "location_name",
        "year_id",
        "temperature_zone",
    ]
    assert list(result.columns[5:]) == list(TEMPERATURE_COLUMNS)
    assert isinstance(result.columns[5], float)
    keys = result[["location_id", "year_id", "temperature_zone"]]
    assert keys.equals(keys.sort_values(list(keys.columns)))
    assert not keys.duplicated().any()
    assert isinstance(result.index, pd.RangeIndex)


def test_compile_labels_us_levels_and_names(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])

    result = _compile(tmp_path, [2023]).drop_duplicates("location_id")

    labels = dict(zip(result["location_id"], result["level"], strict=True))
    assert labels == {
        102: "national",
        523: "state",
        524: "state",
        1001: "county",
        1002: "county",
        1003: "county",
    }
    names = dict(zip(result["location_id"], result["location_name"], strict=True))
    assert names[1003] == "Anchorage"


def test_compile_labels_other_countries_by_admin_level(tmp_path: Path) -> None:
    _write_years(tmp_path, [2023])

    def year_path(year: int) -> Path:
        return person_days_year_path(
            tmp_path, WEIGHTS_HIERARCHY, SCENARIO, MEMBER, year
        )

    result = compile_country_person_days(_hierarchy(), 200, [2023], year_path, None)

    labels = dict(zip(result["location_id"], result["level"], strict=True))
    assert labels == {200: "admin0", 300: "admin1", 2001: "admin2"}


# --- GEOIDs ---


def _crosswalk(path: Path, rows: list[tuple[float, int, int]]) -> Path:
    """USHD's crosswalk: (fips, location_id, current); fips is a float, as on disk."""
    pd.DataFrame(rows, columns=["fips", "location_id", "current"]).to_csv(
        path, index=False
    )
    return path


GOOD_ROWS = [
    (1.0, 523, 1),
    (2.0, 524, 1),
    (1001.0, 1001, 1),
    (1003.0, 1002, 1),
    (2020.0, 1003, 1),
    # A current USHD unit LSAE does not have, with no FIPS: ignored.
    (np.nan, 94061, 1),
    # A retired row for an LSAE county: ignored.
    (9999.0, 1001, 0),
]


def test_load_us_geoids_zero_pads_states_and_counties(tmp_path: Path) -> None:
    crosswalk = _crosswalk(tmp_path / "locations.csv", GOOD_ROWS)

    geoids = load_us_geoids(crosswalk, _hierarchy())

    assert geoids.to_dict() == {
        523: "01",
        524: "02",
        1001: "01001",
        1002: "01003",
        1003: "02020",
    }


def test_load_us_geoids_fails_on_an_unmapped_county(tmp_path: Path) -> None:
    rows = []
    for row in GOOD_ROWS:
        if row[1] != 1003:  # noqa: PLR2004
            rows.append(row)
    crosswalk = _crosswalk(tmp_path / "locations.csv", rows)

    with pytest.raises(ValueError, match="1003"):
        load_us_geoids(crosswalk, _hierarchy())


def test_load_us_geoids_fails_on_a_duplicate_fips(tmp_path: Path) -> None:
    rows = [*GOOD_ROWS[:3], (1001.0, 1002, 1), *GOOD_ROWS[4:]]
    crosswalk = _crosswalk(tmp_path / "locations.csv", rows)

    with pytest.raises(ValueError, match="01001"):
        load_us_geoids(crosswalk, _hierarchy())


def test_load_us_geoids_fails_when_a_county_is_outside_its_state(
    tmp_path: Path,
) -> None:
    # County 1003 sits under state 524 (02) but gets an Alabama FIPS.
    rows = [*GOOD_ROWS[:4], (1005.0, 1003, 1), *GOOD_ROWS[5:]]
    crosswalk = _crosswalk(tmp_path / "locations.csv", rows)

    with pytest.raises(ValueError, match="1003"):
        load_us_geoids(crosswalk, _hierarchy())


def test_insert_geoid_follows_location_id_and_leaves_the_nation_null(
    tmp_path: Path,
) -> None:
    _write_years(tmp_path, [2023])
    crosswalk = _crosswalk(tmp_path / "locations.csv", GOOD_ROWS)
    compiled = _compile(tmp_path, [2023])

    result = insert_geoid(compiled, load_us_geoids(crosswalk, _hierarchy()))

    assert list(result.columns[:3]) == ["location_id", "geoid", "level"]
    pd.testing.assert_frame_equal(result.drop(columns="geoid"), compiled)
    assert result.loc[result["location_id"] == US, "geoid"].isna().all()
    assert (result.loc[result["location_id"] == 1002, "geoid"] == "01003").all()  # noqa: PLR2004
    assert result.loc[result["location_id"] != US, "geoid"].notna().all()


# --- CLI ---


def test_cli_task_is_registered_and_requires_output_dir() -> None:
    from click.testing import CliRunner

    from climate_data.special import TASK_RUNNERS

    runner = CliRunner()
    command = TASK_RUNNERS["compile_country_person_days"]
    assert runner.invoke(command, ["--help"]).exit_code == 0
    result = runner.invoke(
        command,
        ["--cmip6-experiment", "ssp245", "--gcm-member", "CNRM-CM6-1_r4i1p1f2"],
    )
    assert result.exit_code != 0
    assert "--output-dir" in result.output
