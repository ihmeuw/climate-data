import itertools
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tqdm
from rra_tools.shell_tools import mkdir

from climate_data import cli_options as clio
from climate_data import constants as cdc
from climate_data.data import (
    ClimateAggregateData,
    ClimateData,
    PopulationModelData,
    save_parquet,
)
from climate_data.jobmon_utils import run_parallel_maybe_dry_run
from climate_data.special import utils

# First forecast year: years before this use ERA5 historical daily temperature.
FORECAST_START_YEAR = int(cdc.FORECAST_YEARS[0])


def temperature_person_days_main(
    block_key: str,
    gcm_member: str,
    scenario: str,
    hierarchy: str,
    population_model_root: str,
    climate_data_root: str,
    output_dir: str,
    *,
    progress_bar: bool = False,
) -> None:
    """Bin population into (temperature x temperature-zone) person-days per year.

    The year span is derived from the ``temperature_zone`` actually on disk (not a
    hardcoded range): for the ``historical`` scenario this is the ERA5-only product
    spanning ``EXPOSURE_START_YEAR`` through the last year present (1990-2025 as
    delivered); for a forecast scenario, daily temperature is ERA5 before
    ``FORECAST_START_YEAR`` and the GCM scenario from then on, binned against that
    scenario's own zone. A year whose daily or population input is missing fails loudly
    rather than being skipped, so a gap in a product that must be square cannot slip
    through silently.
    """
    print(f"Aggregating {gcm_member} for {block_key}")
    pm_data = PopulationModelData(population_model_root)
    cd_data = ClimateData(climate_data_root, read_only=True)
    ca_data = ClimateAggregateData(Path(output_dir) / hierarchy)

    print("Building location masks")
    climate_slice, location_ids, location_idx = utils.build_location_index(
        hierarchy, block_key, pm_data
    )

    print("Building data index")
    temperature_bins = np.arange(-35, 45, 0.1)
    temperature_zone_bins = np.arange(-25, 35, 1)
    data_idx = pd.MultiIndex.from_product(
        [location_ids, temperature_bins, temperature_zone_bins]
    )
    out_template = np.zeros(
        (len(location_ids), len(temperature_bins), len(temperature_zone_bins)),
        dtype=np.float64,
    )

    print("Building historical temperature zone index")
    temperature_zone = cd_data.load_compiled_annual_results(
        scenario, "temperature_zone", gcm_member
    ).sel(**climate_slice)
    historical_temperature_zone_idx = utils.to_idx(
        temperature_zone, temperature_zone_bins
    )

    print("Building temperature coordinates")
    temperature_coordinates = utils.get_temperature_coordinates(
        block_key, pm_data, temperature_zone
    )

    print("Aggregating temperature person days")
    # Drive the span from the zone actually on disk (historical: 1990..last present;
    # forecast: the compiled historical+forecast series). A year whose daily or
    # population input is missing now fails loudly rather than being skipped, so a gap
    # in a product that must be square can't slip through silently.
    zone_years = [int(y) for y in temperature_zone["year"].to_numpy()]
    zone_row_for_year = {y: i for i, y in enumerate(zone_years)}
    years = [y for y in zone_years if y >= cdc.EXPOSURE_START_YEAR]
    dfs = []
    for year in tqdm.tqdm(years, disable=not progress_bar):
        if scenario == "historical" or year < FORECAST_START_YEAR:
            temperature = cd_data.load_daily_results(
                "historical", "mean_temperature", year
            ).sel(**climate_slice)
        else:
            temperature = cd_data.load_raw_daily_results(
                scenario, "mean_temperature", year, gcm_member
            ).sel(**climate_slice)
        # Population nodata is stored as NaN. The aggregate stage tolerates this via
        # np.nansum, but here we accumulate into `out_arr` with `+=`, so a NaN pixel
        # would poison every output cell it touches (silently read back as 0, which
        # zeroed small locations like American Samoa). Zero-fill nodata first: no
        # modeled population contributes no person-days.
        pop_arr = pm_data.load_results(f"{year}q1", block_key)._ndarray.flatten()  # noqa: SLF001
        pop_arr = np.nan_to_num(pop_arr, nan=0.0)
        temperature_idx = utils.to_idx(temperature, temperature_bins)

        out_arr = out_template.copy()
        utils.compute_person_days(
            location_idx,
            temperature_idx,
            historical_temperature_zone_idx[zone_row_for_year[year]],
            pop_arr,
            temperature_coordinates,
            out_arr,
        )

        df = (
            pd.DataFrame({"person_days": out_arr.reshape(-1)}, index=data_idx)
            .assign(year=year)
            .set_index("year", append=True)
        )
        df.index.names = ["location_id", "temperature", "temperature_zone", "year"]
        df = df.reset_index()
        df["temperature"] = df["temperature"].round(1)
        df = df.set_index(["location_id", "year", "temperature_zone", "temperature"])[
            "person_days"
        ].unstack()
        df.columns.name = None
        dfs.append(df)

    if not dfs:
        msg = (
            f"No person-days produced for {scenario} {gcm_member} {block_key}: the "
            f"temperature zone has no year >= {cdc.EXPOSURE_START_YEAR}."
        )
        raise ValueError(msg)
    df = pd.concat(dfs)
    out_path = ca_data.person_days_path(block_key, scenario, gcm_member)
    mkdir(out_path.parent, parents=True, exist_ok=True)
    save_parquet(df, out_path)


@click.command()
@clio.with_block_key()
@clio.with_gcm_member()
@clio.with_scenario()
@clio.with_hierarchy(choices=cdc.GBD_HIERARCHIES, default="gbd_2023")
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@clio.with_input_directory("climate-data", cdc.MODEL_ROOT)
@clio.with_output_directory(cdc.AGGREGATE_ROOT)
@clio.with_run_mode()
@clio.with_progress_bar()
def temperature_person_days_task(
    block_key: str,
    gcm_member: str,
    scenario: str,
    hierarchy: str,
    population_model_dir: str,
    climate_data_dir: str,
    output_dir: str,
    run_mode: str,
    *,
    progress_bar: bool,
) -> None:
    if scenario == "historical" and gcm_member != "era5":
        msg = f"The 'historical' scenario must use the 'era5' gcm-member, got {gcm_member}"
        raise ValueError(msg)
    climate_data_dir = clio.resolve_run_mode_root(
        "climate_data_dir", climate_data_dir, run_mode
    )
    output_dir = clio.resolve_run_mode_root(
        "output_dir", output_dir, run_mode, aggregate=True
    )
    temperature_person_days_main(
        block_key,
        gcm_member,
        scenario,
        hierarchy,
        population_model_dir,
        climate_data_dir,
        output_dir,
        progress_bar=progress_bar,
    )


@click.command()
@clio.with_block_key(allow_all=True)
@clio.with_scenario(allow_all=True)
@clio.with_hierarchy(choices=cdc.GBD_HIERARCHIES, default="gbd_2023")
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@clio.with_input_directory("climate-data", cdc.MODEL_ROOT)
@clio.with_output_directory(cdc.AGGREGATE_ROOT)
@clio.with_run_mode()
@clio.with_queue()
@clio.with_dry_run()
def temperature_person_days(
    block_key: str,
    scenario: list[str],
    hierarchy: str,
    population_model_dir: str,
    climate_data_dir: str,
    output_dir: str,
    run_mode: str,
    queue: str,
    dry_run: bool,
) -> None:
    climate_data_dir = clio.resolve_run_mode_root(
        "climate_data_dir", climate_data_dir, run_mode
    )
    output_dir = clio.resolve_run_mode_root(
        "output_dir", output_dir, run_mode, aggregate=True
    )
    ca_data = ClimateAggregateData(Path(output_dir) / hierarchy)
    cd_data = ClimateData(climate_data_dir, read_only=True)
    pm_data = PopulationModelData(population_model_dir)

    modeling_frame = pm_data.load_modeling_frame()
    block_keys = modeling_frame["block_key"].unique().tolist()
    block_keys = clio.convert_choice(block_key, block_keys)

    jobs = []
    for e in scenario:
        gcm_members = (
            ["era5"]
            if e == "historical"
            else cd_data.list_gcm_members("ssp126", "mean_temperature")
        )
        for blk, g in itertools.product(block_keys, gcm_members):
            if not ca_data.person_days_path(blk, e, g).exists():
                jobs.append((blk, g, e))

    print(f"Running {len(jobs)} jobs")

    run_parallel_maybe_dry_run(
        runner="cdtask special",
        task_name="temperature_person_days",
        flat_node_args=(
            ("block-key", "gcm-member", "scenario"),
            jobs,
        ),
        task_args={
            "hierarchy": hierarchy,
            "population-model-dir": population_model_dir,
            "climate-data-dir": climate_data_dir,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "25G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        log_root=ca_data.log_dir("temperature_person_days"),
        max_attempts=3,
        dry_run=dry_run,
    )
