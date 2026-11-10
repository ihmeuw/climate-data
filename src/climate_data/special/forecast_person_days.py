"""Forecast stage of the CLIMATE-45 person-days: global admin2, per member-year.

For one member and scenario, this generates the daily mean temperature of each forecast
year (2024-2100) in memory with ``generate_scenario_daily_main(write_output=False)``,
bins every (location, climate cell) pair of the population weight tables against it, and
writes person-days keyed by (location, year, zone), in the history stage's layout.

**Rounding.** Production wrote each forecast member-year to disk as int16 with a 0.01
scale and binned what it read back. The generated floats are rounded the same way, so
the bin-edge defect is inherited on purpose (CLIMATE-40, comment 521591), as in history.

**The zone, option B1.** Production reads the member's compiled ``temperature_zone``: the
10-year trailing mean of ERA5's annual means (before 2024) followed by the member's
(2024 on), each stored as int16 with a 0.01 scale. Those annual files predate the
calendar fix (CLIMATE-35), so for ``noleap`` members they are off in leap years. Here the
zone is built from this stage's own annual means instead. The trailing mean for year Y
needs the nine years before it, so the years of one member-scenario run as a chain, in
order: each year writes its rounded annual mean, and later years read them back. A chain
resumes from the annual means on disk, skipping finished years.
"""

import resource
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from climate_data import cli_options as clio
from climate_data import constants as cdc
from climate_data.data import (
    ClimateData,
    save_parquet,
    save_xarray,
)
from climate_data.generate import utils as gutils
from climate_data.generate.scenario_daily import generate_scenario_daily_main
from climate_data.jobmon_utils import run_parallel_maybe_dry_run
from climate_data.special import history_person_days as hpd
from climate_data.special.weight_table import frame_block_keys, missing_blocks

# How the July forecast dailies and every annual file were stored.
DAILY_SCALE = 0.01
DAILY_FILL = -32767
STORAGE_ENCODING = {"scale_factor": DAILY_SCALE, "add_offset": 0.0}

FORECAST_START_YEAR = int(cdc.FORECAST_YEARS[0])
FORECAST_PERSON_DAYS_YEARS = []
for _year in cdc.FORECAST_YEARS:
    FORECAST_PERSON_DAYS_YEARS.append(int(_year))


def forecast_person_days_path(
    output_root: str | Path,
    hierarchy: str,
    scenario: str,
    gcm_member: str,
    year: int,
) -> Path:
    """``<root>/forecast/<hierarchy>/<scenario>/<member>/<year>.parquet``."""
    return (
        Path(output_root)
        / "forecast"
        / hierarchy
        / scenario
        / gcm_member
        / f"{year}.parquet"
    )


def annual_mean_path(
    output_root: str | Path,
    scenario: str,
    gcm_member: str,
    year: int,
) -> Path:
    """``<root>/annual_mean/<scenario>/<member>/<year>.nc``; independent of hierarchy."""
    return Path(output_root) / "annual_mean" / scenario / gcm_member / f"{year}.nc"


def missing_chains(
    chains: list[tuple[str, str]],
    years: list[int],
    output_root: str | Path,
    hierarchy: str,
) -> list[tuple[str, str]]:
    """The (scenario, member) chains with at least one year not written, in input order."""
    missing = []
    for scenario, gcm_member in chains:
        for year in years:
            path = forecast_person_days_path(
                output_root, hierarchy, scenario, gcm_member, year
            )
            if not path.exists():
                missing.append((scenario, gcm_member))
                break
    return missing


def _scaled_for_storage(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """``round(x / 0.01)`` as xarray encodes it: in the data's own float dtype.

    With a ``_FillValue`` in the encoding, xarray keeps the data's dtype while scaling,
    so float32 data is divided in float32. Integers are scaled in float64.
    """
    data = np.array(values, copy=True)
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float64)
    data /= DAILY_SCALE
    return np.around(data)


def round_daily_storage(values: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Round as int16 storage with a 0.01 scale does, then decode as xarray does.

    Encode: ``around(x / 0.01)`` in the data's dtype. Decode: float64 times 0.01.
    NaN stays NaN.
    """
    rounded: npt.NDArray[np.float64] = (
        _scaled_for_storage(values).astype(np.float64) * DAILY_SCALE
    )
    return rounded


def pack_int16(values: npt.NDArray[Any]) -> npt.NDArray[np.int16]:
    """The int16 that storage writes; NaN becomes the fill value.

    ``decode_int16(pack_int16(x), 0.01, 0.0, DAILY_FILL)`` equals ``round_daily_storage(x)``.
    """
    scaled = _scaled_for_storage(values)
    packed = np.where(np.isnan(scaled), DAILY_FILL, scaled)
    return packed.astype(np.int16)


def _date_first(daily: xr.Dataset) -> xr.DataArray:
    return daily["value"].transpose("date", "latitude", "longitude")


def annual_mean_rounded(daily: xr.Dataset) -> npt.NDArray[np.float64]:
    """One year's annual mean per cell, flat, as the stored annual file holds it.

    ``utils.annual_mean`` is the production function behind the annual
    ``mean_temperature`` files; its result is rounded as their int16 storage does.
    """
    annual = gutils.annual_mean(daily)["value"].transpose(
        "year", "latitude", "longitude"
    )
    if annual.sizes["year"] != 1:
        msg = f"Expected one year of dailies, got {annual['year'].to_numpy().tolist()}."
        raise ValueError(msg)
    return round_daily_storage(annual.to_numpy().reshape(-1))


def forecast_zone(
    era5_annual: xr.DataArray,
    member_annual: xr.DataArray,
    year: int,
) -> npt.NDArray[np.float64]:
    """The zone for forecast ``year``: the compiled series' rounded 10-year trailing mean.

    Parameters
    ----------
    era5_annual
        ERA5 annual means, dims (year, <cells>), every year before 2024.
    member_annual
        The member's rounded annual means, dims (year, <cells>), every year from 2024 to
        ``year``. Later years are ignored.
    year
        The forecast year.
    """
    # Only the window's years: trailing_zone sums them in a fixed order, so the rest of
    # the series does not change the result.
    first = year - hpd.ZONE_WINDOW + 1
    member_years = member_annual["year"]
    member = member_annual.sel(year=(member_years >= first) & (member_years <= year))
    era5_years = era5_annual["year"]
    era5 = era5_annual.sel(
        year=(era5_years >= first) & (era5_years < FORECAST_START_YEAR)
    )
    series = xr.concat([era5, member], dim="year")
    return hpd.trailing_zone(xr.Dataset({"value": series}), year)


def _check_contiguous(years: list[int]) -> None:
    if not years or years != list(range(years[0], years[0] + len(years))):
        msg = f"Chain years must be contiguous and increasing, got {years}."
        raise ValueError(msg)
    if years[0] < FORECAST_START_YEAR:
        msg = f"Chain years start in {FORECAST_START_YEAR}, got {years[0]}."
        raise ValueError(msg)


def _read_packed_annual(path: Path) -> npt.NDArray[np.int16]:
    with xr.open_dataset(path, mask_and_scale=False) as ds:
        packed: npt.NDArray[np.int16] = ds["value"].to_numpy().reshape(-1)
    return packed


def _write_annual(
    rounded: npt.NDArray[np.float64],
    daily: xr.Dataset,
    year: int,
    path: Path,
) -> None:
    shape = (1, daily.sizes["latitude"], daily.sizes["longitude"])
    ds = xr.Dataset(
        {"value": (("year", "latitude", "longitude"), rounded.reshape(shape))},
        coords={
            "year": [year],
            "latitude": daily["latitude"].to_numpy(),
            "longitude": daily["longitude"].to_numpy(),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_xarray(ds, path, STORAGE_ENCODING, atomic=True)


def run_forecast_chain(
    years: list[int],
    era5_annual: xr.DataArray,
    generate: Callable[[int], xr.Dataset],
    load_weights: Callable[[int], pd.DataFrame],
    output_path: Callable[[int], Path],
    annual_path: Callable[[int], Path],
    *,
    overwrite: bool = False,
) -> list[int]:
    """Run one member-scenario's forecast years in order. Returns the years computed.

    Parameters
    ----------
    years
        Contiguous, increasing, from 2024 on.
    era5_annual
        ERA5 annual means, dims (year, cell) on the flat global grid, before 2024.
    generate
        Returns one year's daily mean temperature, dims date, latitude and longitude, on
        the grid of ``era5_annual``.
    load_weights
        Returns one year's weights (``location_id``, ``cell``, ``population``).
    output_path, annual_path
        Where a year's person-days and rounded annual mean go.
    overwrite
        Recompute years whose files exist.

    A year is skipped when both its files exist (unless ``overwrite``). Annual means for
    the years before ``years[0]``, back to 2024 at most, must already be on disk.
    """
    _check_contiguous(years)
    n_cells = era5_annual.shape[1]
    cell_dim = era5_annual.dims[1]

    # The chain's annual means so far, packed as int16: 13 MB per global year.
    packed: dict[int, npt.NDArray[np.int16]] = {}
    for year in range(FORECAST_START_YEAR, years[0]):
        path = annual_path(year)
        if not path.exists():
            msg = (
                f"Chain starts in {years[0]}, but the annual mean for {year} is not "
                f"on disk: {path}"
            )
            raise FileNotFoundError(msg)
        packed[year] = _read_packed_annual(path)

    done = []
    for year in years:
        out_path, ann_path = output_path(year), annual_path(year)
        if out_path.exists() and ann_path.exists() and not overwrite:
            packed[year] = _read_packed_annual(ann_path)
            continue

        daily = generate(year)
        rounded_annual = annual_mean_rounded(daily)
        if rounded_annual.size != n_cells:
            msg = f"Dailies for {year} have {rounded_annual.size} cells, not {n_cells}."
            raise ValueError(msg)
        _write_annual(rounded_annual, daily, year, ann_path)
        packed[year] = pack_int16(rounded_annual)

        weights = load_weights(year)
        cells = hpd.used_cells(weights)
        values = _date_first(daily).to_numpy()
        temperature = round_daily_storage(values.reshape(values.shape[0], -1)[:, cells])
        del daily, values

        # Only the zone window's member years; forecast_zone uses no others.
        member_years = []
        for member_year in sorted(packed):
            if member_year > year - hpd.ZONE_WINDOW:
                member_years.append(member_year)
        member_cells = []
        for member_year in member_years:
            member_cells.append(
                hpd.decode_int16(
                    packed[member_year][cells], DAILY_SCALE, 0.0, DAILY_FILL
                )
            )
        member = xr.DataArray(
            np.stack(member_cells),
            dims=("year", cell_dim),
            coords={"year": member_years},
        )
        zone = np.full(n_cells, np.nan)
        zone[cells] = forecast_zone(era5_annual[:, cells], member, year)

        result = hpd.history_person_days(weights, zone, temperature, year)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_parquet(result, out_path, atomic=True)
        done.append(year)
    return done


def generate_daily_mean_temperature(
    scenario: str,
    gcm_member: str,
    year: int,
    climate_data_root: str | Path,
) -> xr.Dataset:
    """One member-year of daily mean temperature, generated in memory, as the pilot did.

    ``write_output=False``: nothing is written. The settings are the production defaults
    for an additive variable.
    """
    daily = generate_scenario_daily_main(
        "mean_temperature",
        scenario,
        year,
        gcm_member,
        climate_data_root,
        write_output=False,
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_MONTHLY,
        reference_years=cdc.REFERENCE_YEARS_ARG,
        eps_floor=cdc.DEFAULT_EPS_FLOOR,
        anomaly_cap=cdc.DEFAULT_ANOMALY_CAP,
    )
    hpd.check_global_grid(daily, f"generated {scenario} {gcm_member} {year}")
    return daily


def load_era5_annual_flat(cd_data: ClimateData) -> xr.DataArray:
    """ERA5 annual mean temperature before 2024, dims (year, cell) on the global grid."""
    annual = cd_data.load_raw_annual_mfdataset("historical", "mean_temperature")
    hpd.check_global_grid(annual, "ERA5 annual mean temperature")
    value = annual["value"].transpose("year", "latitude", "longitude")
    value = value.sel(year=value["year"] < FORECAST_START_YEAR)
    flat = value.to_numpy().reshape(value.sizes["year"], -1)
    return xr.DataArray(
        flat, dims=("year", "cell"), coords={"year": value["year"].to_numpy()}
    )


def forecast_person_days_main(
    scenario: str,
    gcm_member: str,
    years: list[int],
    hierarchy: str,
    climate_data_root: str | Path,
    weights_root: str | Path,
    population_model_root: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> list[int]:
    """Run one member-scenario chain over ``years``. Returns the years computed."""
    cd_data = ClimateData(climate_data_root, read_only=True)
    block_keys = frame_block_keys(population_model_root)
    era5_annual = load_era5_annual_flat(cd_data)

    def generate(year: int) -> xr.Dataset:
        return generate_daily_mean_temperature(
            scenario, gcm_member, year, climate_data_root
        )

    def load_weights(year: int) -> pd.DataFrame:
        return hpd.load_year_weights(weights_root, hierarchy, year, block_keys)

    def output_path(year: int) -> Path:
        return forecast_person_days_path(
            output_root, hierarchy, scenario, gcm_member, year
        )

    def annual_path(year: int) -> Path:
        return annual_mean_path(output_root, scenario, gcm_member, year)

    return run_forecast_chain(
        years,
        era5_annual,
        generate,
        load_weights,
        output_path,
        annual_path,
        overwrite=overwrite,
    )


@click.command()
@clio.with_cmip6_experiment()
@clio.with_gcm_member()
@click.option("--year-start", type=int, default=FORECAST_START_YEAR, show_default=True)
@click.option(
    "--year-end",
    type=int,
    default=FORECAST_PERSON_DAYS_YEARS[-1],
    show_default=True,
)
@clio.with_hierarchy(default="lsae_1285")
@clio.with_input_directory("climate-data", cdc.MODEL_ROOT)
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@click.option(
    "--weights-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Root holding weights/<hierarchy>/. Defaults to --output-dir.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Root for forecast/ and annual_mean/. No default, by design.",
)
@clio.with_overwrite()
def build_forecast_person_days_task(
    cmip6_experiment: str,
    gcm_member: str,
    year_start: int,
    year_end: int,
    hierarchy: str,
    climate_data_dir: str,
    population_model_dir: str,
    weights_dir: str | None,
    output_dir: str,
    *,
    overwrite: bool,
) -> None:
    """Run one member-scenario's chain of forecast years."""
    years = list(range(year_start, year_end + 1))
    start = time.time()
    done = forecast_person_days_main(
        cmip6_experiment,
        gcm_member,
        years,
        hierarchy,
        climate_data_dir,
        weights_dir or output_dir,
        population_model_dir,
        output_dir,
        overwrite=overwrite,
    )
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(
        f"forecast_person_days {cmip6_experiment} {gcm_member}: {len(done)} of "
        f"{len(years)} years computed, {time.time() - start:.1f} s, "
        f"peak RSS {peak_gb:.2f} GB"
    )


@click.command()
@clio.with_cmip6_experiment(allow_all=True)
@clio.with_hierarchy(default="lsae_1285")
@clio.with_input_directory("climate-data", cdc.MODEL_ROOT)
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@click.option(
    "--weights-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Root holding weights/<hierarchy>/. Defaults to --output-dir.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Root for forecast/ and annual_mean/. No default, by design.",
)
@clio.with_queue()
@clio.with_concurrency_limit(default=50)
@clio.with_dry_run()
def build_forecast_person_days(
    cmip6_experiment: list[str],
    hierarchy: str,
    climate_data_dir: str,
    population_model_dir: str,
    weights_dir: str | None,
    output_dir: str,
    queue: str,
    concurrency_limit: int,
    dry_run: bool,
) -> None:
    """Run every member-scenario chain that has a forecast year not yet written."""
    weights_root = weights_dir or output_dir
    all_blocks = frame_block_keys(population_model_dir)
    absent = missing_blocks(all_blocks, weights_root, hierarchy)
    if absent:
        msg = f"{len(absent)} blocks have no weight table, e.g. {absent[:3]}."
        raise click.ClickException(msg)

    cd_data = ClimateData(climate_data_dir, read_only=True)
    chains = []
    for scenario in cmip6_experiment:
        for gcm_member in cd_data.list_gcm_members(scenario, "mean_temperature"):
            chains.append((scenario, gcm_member))
    jobs = missing_chains(chains, FORECAST_PERSON_DAYS_YEARS, output_dir, hierarchy)
    print(f"Running {len(jobs)} of {len(chains)} chains")

    run_parallel_maybe_dry_run(
        runner="cdtask special",
        task_name="forecast_person_days",
        flat_node_args=(("cmip6-experiment", "gcm-member"), jobs),
        task_args={
            "hierarchy": hierarchy,
            "climate-data-dir": climate_data_dir,
            "population-model-dir": population_model_dir,
            "weights-dir": weights_root,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "120G",
            "runtime": "960m",
            "project": "proj_rapidresponse",
        },
        log_root=Path(output_dir) / "logs" / "forecast_person_days",
        max_attempts=3,
        concurrency_limit=concurrency_limit,
        dry_run=dry_run,
    )
