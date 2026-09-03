"""Tests for the year space the ERA5 extract commands offer (CLIMATE-29)."""

import click

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
