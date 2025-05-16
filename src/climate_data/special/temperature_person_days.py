import itertools

import click
import numpy as np
import pandas as pd
import tqdm
from rra_tools import jobmon

from climate_data import cli_options as clio
from climate_data import constants as cdc
from climate_data.data import (
    ClimateAggregateData,
    ClimateData,
    PopulationModelData,
    save_parquet,
)
from climate_data.special import utils

HIERARCHY = "gbd_2021"


def temperature_person_days_main(
    block_key: str,
    gcm_member: str,
    scenario: str,
    population_model_root: str,
    climate_data_root: str,
    output_dir: str,
    *,
    progress_bar: bool = False,
) -> None:
    print(f"Aggregating {gcm_member} for {block_key}")
    pm_data = PopulationModelData(population_model_root)
    cd_data = ClimateData(climate_data_root, read_only=True)
    ca_data = ClimateAggregateData(output_dir)

    print("Building location masks")
    climate_slice, location_ids, location_idx = utils.build_location_index(
        HIERARCHY, block_key, pm_data
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
    years = list(range(1990, 2101))
    dfs = []
    for tz_idx, year in tqdm.tqdm(list(enumerate(years)), disable=not progress_bar):
        if year < 2024:
            temperature = cd_data.load_daily_results(
                "historical", "mean_temperature", year
            ).sel(**climate_slice)
        else:
            temperature = cd_data.load_raw_daily_results(
                scenario, "mean_temperature", year, gcm_member
            ).sel(**climate_slice)
        temperature_idx = utils.to_idx(temperature, temperature_bins)

        pop_arr = pm_data.load_results(f"{year}q1", block_key)._ndarray.flatten()  # noqa: SLF001
        out_arr = out_template.copy()
        utils.compute_person_days(
            location_idx,
            temperature_idx,
            historical_temperature_zone_idx[tz_idx],
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

    df = pd.concat(dfs)
    out_path = (
        ca_data.root
        / "erf-scratch"
        / "person-days"
        / f"{scenario}_{gcm_member}_{block_key}.parquet"
    )
    save_parquet(df, out_path)


@click.command()
@clio.with_block_key()
@clio.with_gcm_member()
@clio.with_cmip6_experiment()
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@clio.with_input_directory("climate-data", cdc.MODEL_ROOT)
@clio.with_output_directory(cdc.AGGREGATE_ROOT)
@clio.with_progress_bar()
def temperature_person_days_task(
    block_key: str,
    gcm_member: str,
    cmip6_experiment: str,
    population_model_dir: str,
    climate_data_dir: str,
    output_dir: str,
    *,
    progress_bar: bool,
) -> None:
    temperature_person_days_main(
        block_key,
        gcm_member,
        cmip6_experiment,
        population_model_dir,
        climate_data_dir,
        output_dir,
        progress_bar=progress_bar,
    )


@click.command()
@clio.with_block_key(allow_all=True)
@clio.with_cmip6_experiment(allow_all=True)
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@clio.with_input_directory("climate-data", cdc.MODEL_ROOT)
@clio.with_output_directory(cdc.AGGREGATE_ROOT)
@clio.with_queue()
def temperature_person_days(
    block_key: str,
    cmip6_experiment: list[str],
    population_model_dir: str,
    climate_data_dir: str,
    output_dir: str,
    queue: str,
) -> None:
    ca_data = ClimateAggregateData(output_dir)
    cd_data = ClimateData(climate_data_dir, read_only=True)
    pm_data = PopulationModelData(population_model_dir)

    modeling_frame = pm_data.load_modeling_frame()
    block_keys = modeling_frame["block_key"].unique().tolist()
    block_keys = clio.convert_choice(block_key, block_keys)

    gcm_members = cd_data.list_gcm_members()

    jobs = []
    possible_jobs = list(itertools.product(block_keys, gcm_members, cmip6_experiment))
    for block_key, gcm_member, cmip6_experiment in possible_jobs:
        path = (
            ca_data.root
            / "erf-scratch"
            / "person-days"
            / f"{cmip6_experiment}_{gcm_member}_{block_key}.parquet"
        )
        if not path.exists():
            jobs.append((block_key, gcm_member, cmip6_experiment))

    print(f"Running {len(jobs)} jobs")

    jobmon.run_parallel(
        runner="cdtask special",
        task_name="temperature_person_days",
        flat_node_args=(
            ("block-key", "gcm-member", "cmip6-experiment"),
            jobs,
        ),
        task_args={
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
    )
