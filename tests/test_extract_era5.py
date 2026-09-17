"""Tests for the year space and default variable set of the ERA5 extract commands."""

import click
import pytest

from climate_data import constants as cdc
from climate_data.extract import era5

# The generate stage closes a month's final day with the next month's first sample, so
# closing the last history year reaches one year past it.
LOOKAHEAD_YEAR = str(int(cdc.HISTORY_YEARS[-1]) + 1)


def _year_choices(command: click.Command) -> set[str]:
    """The values the command's `--year` option will accept."""
    for param in command.params:
        if param.name == "year":
            assert isinstance(param.type, click.Choice)
            return set(param.type.choices)
    msg = f"{command.name} has no --year option"
    raise AssertionError(msg)


def test_extract_tasks_offer_the_lookahead_year() -> None:
    """The look-ahead extract the generate stage demands must be requestable.

    `load_variable_with_lookahead` raises when the next month's extract is missing and
    tells the operator to extract it. Both extract tasks bound `--year` to
    `HISTORY_YEARS`, which stops at the last history year, so that instruction named a
    year click would reject -- there was no CLI path to produce the file the pipeline
    hard-requires. The 2024 files that made the first full run work came from a separate
    GBD-2025 pull, not from this repo.
    """
    for command in (era5.download_era5_task, era5.unzip_and_compress_era5_task):
        assert LOOKAHEAD_YEAR in _year_choices(command), command.name


def test_extract_runner_does_not_widen_its_all_expansion() -> None:
    """`-y ALL` must not quietly acquire an extra year of downloads.

    `clio.with_year(allow_all=True)` resolves `ALL` to every choice, and
    `build_task_lists` decides what to fetch by file existence, so widening the runner's
    year space would add a year to every `cdrun extract era5 -y ALL` -- on a step that
    already re-downloads terabytes when run with its defaults. Only the single-job tasks
    reach past the history range.
    """
    runner_years = _year_choices(era5.extract_era5)

    assert LOOKAHEAD_YEAR not in runner_years
    assert runner_years == {*cdc.HISTORY_YEARS, "ALL"}


def test_year_floor_blocks_the_default_year_space() -> None:
    """Regression for CLIMATE-27: the bare invocation must not refill ERF's deletion.

    `--year` defaults to ALL over `HISTORY_YEARS`, which starts in 1950, and
    `build_task_lists` reads a missing file as work to do. Without this guard a bare
    `cdrun extract era5` re-downloads every pre-1980 extract ERF deleted in Sep2026.
    """
    with pytest.raises(click.UsageError, match="below 1980"):
        era5.check_extract_year_floor(cdc.HISTORY_YEARS, allow_pre_floor=False)


def test_year_floor_allows_an_explicit_backfill() -> None:
    era5.check_extract_year_floor(cdc.HISTORY_YEARS, allow_pre_floor=True)


def test_year_floor_passes_years_at_or_above_the_floor() -> None:
    at_or_above = [
        y for y in cdc.HISTORY_YEARS if int(y) >= int(cdc.EXTRACT_YEAR_FLOOR)
    ]
    era5.check_extract_year_floor(at_or_above, allow_pre_floor=False)
    assert at_or_above[0] == cdc.EXTRACT_YEAR_FLOOR


def test_runner_exposes_the_backfill_flag() -> None:
    """The guard has to be overridable from the CLI, not only in code."""
    flags = {param.name for param in era5.extract_era5.params}
    assert "allow_pre_1980" in flags


def test_all_expansion_drops_never_read_variables() -> None:
    """Regression for CLIMATE-27: `ALL` should not mean variables nothing opens.

    `surface_pressure` has no `TRANSFORM_MAP` source entry and is not an
    `AGGREGATION_MEASURES` member, but the default expansion downloaded it every month of
    every year.
    """
    kept = era5.variables_for_full_expansion(list(cdc.ERA5_VARIABLES))

    assert cdc.ERA5_VARIABLES.surface_pressure not in kept
    assert set(kept) == set(cdc.ERA5_VARIABLES) - set(cdc.EXTRACT_UNUSED_VARIABLES)


def test_an_explicit_variable_request_is_untouched() -> None:
    """Naming a variable still extracts it, including a never-read one."""
    asked = [cdc.ERA5_VARIABLES.surface_pressure]

    assert era5.variables_for_full_expansion(asked) == asked


def test_a_partial_request_is_untouched() -> None:
    asked = [cdc.ERA5_VARIABLES.temperature, cdc.ERA5_VARIABLES.surface_pressure]

    assert era5.variables_for_full_expansion(asked) == asked
