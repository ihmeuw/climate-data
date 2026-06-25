from pathlib import Path

import click
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


def compile_person_days_main(
    gcm_member: str,
    scenario: str,
    hierarchy: str,
    population_model_root: str,
    output_dir: str,
    *,
    progress_bar: bool = False,
) -> None:
    print(f"Compiling person-days for {gcm_member} {scenario}")
    pm_data = PopulationModelData(population_model_root)
    ca_data = ClimateAggregateData(Path(output_dir) / hierarchy)

    # Load hierarchy data for aggregation
    hierarchy_df = pm_data.load_subset_hierarchy(hierarchy)

    modeling_frame = pm_data.load_modeling_frame()
    block_keys = modeling_frame["block_key"].unique().tolist()

    print("Loading block data")
    block_data = []
    for block_key in tqdm.tqdm(block_keys, disable=not progress_bar):
        path = ca_data.person_days_path(block_key, scenario, gcm_member)
        block_df = pd.read_parquet(path)
        block_data.append(block_df)

    print("Aggregating most-detailed locations")
    df = (
        pd.concat(block_data)
        .groupby(["location_id", "year", "temperature_zone"])
        .sum()
        .sort_index()
        .reset_index()
        .rename(columns={"year": "year_id"})
    )

    print("Aggregating location hierarchy")
    agg_df = utils.aggregate_to_hierarchy(
        df,
        hierarchy_df,
    ).set_index(["location_id", "year_id", "temperature_zone"])

    print("Saving subset hierarchies")
    # Produce views for subset hierarchies
    subset_hierarchies = cdc.HIERARCHY_MAP[hierarchy]
    for subset_hierarchy in subset_hierarchies:
        # Load the subset hierarchy
        subset_hierarchy_df = pm_data.load_subset_hierarchy(subset_hierarchy)

        # Filter results to only include locations in the subset hierarchy

        subset_location_ids = list(
            set(agg_df.index.get_level_values("location_id")).intersection(
                subset_hierarchy_df["location_id"]
            )
        )
        subset_results = agg_df.loc[subset_location_ids]

        # Save results for the subset hierarchy
        path = ca_data.compiled_person_days_path(
            subset_hierarchy, scenario, gcm_member
        )
        mkdir(path.parent, parents=True, exist_ok=True)
        save_parquet(subset_results, path)


@click.command()
@clio.with_gcm_member()
@clio.with_scenario()
@clio.with_hierarchy(choices=cdc.GBD_HIERARCHIES, default="gbd_2023")
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@clio.with_output_directory(cdc.AGGREGATE_ROOT)
@clio.with_progress_bar()
def compile_person_days_task(
    gcm_member: str,
    scenario: str,
    hierarchy: str,
    population_model_dir: str,
    output_dir: str,
    *,
    progress_bar: bool,
) -> None:
    if scenario == "historical" and gcm_member != "era5":
        msg = f"The 'historical' scenario must use the 'era5' gcm-member, got {gcm_member}"
        raise ValueError(msg)
    compile_person_days_main(
        gcm_member,
        scenario,
        hierarchy,
        population_model_dir,
        output_dir,
        progress_bar=progress_bar,
    )


@click.command()
@clio.with_scenario(allow_all=True)
@clio.with_hierarchy(choices=cdc.GBD_HIERARCHIES, default="gbd_2023")
@clio.with_input_directory("climate-data", cdc.MODEL_ROOT)
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@clio.with_output_directory(cdc.AGGREGATE_ROOT)
@clio.with_queue()
@clio.with_dry_run()
def compile_person_days(
    scenario: list[str],
    hierarchy: str,
    climate_data_dir: str,
    population_model_dir: str,
    output_dir: str,
    queue: str,
    dry_run: bool,
) -> None:
    ca_data = ClimateAggregateData(Path(output_dir) / hierarchy)
    cd_data = ClimateData(climate_data_dir, read_only=True)

    jobs: list[tuple[str, str]] = []
    for e in scenario:
        gcm_members = (
            ["era5"]
            if e == "historical"
            else cd_data.list_gcm_members("ssp126", "mean_temperature")
        )
        jobs.extend((g, e) for g in gcm_members)

    print(f"Running {len(jobs)} jobs")

    run_parallel_maybe_dry_run(
        runner="cdtask special",
        task_name="compile_person_days",
        flat_node_args=(
            ("gcm-member", "scenario"),
            jobs,
        ),
        task_args={
            "hierarchy": hierarchy,
            "population-model-dir": population_model_dir,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "400G",
            "runtime": "100m",
            "project": "proj_rapidresponse",
        },
        log_root=ca_data.log_dir("compile_person_days"),
        max_attempts=3,
        dry_run=dry_run,
    )

    # results_version = "2026_01_12"
    # draw_map = {}
    # for d in range(100):
    #     draw = f"{d:0>3}"
    #     p = cd_data.annual_results_path("ssp126", "mean_temperature", draw).resolve()
    #     draw_map[draw] = p.stem

    # for hierarchy in cdc.HIERARCHY_MAP[HIERARCHY]:
    #     for scenario in cmip6_experiment:
    #         for draw, gcm_variant in draw_map.items():
    #             raw_path = ca_data.root / "erf-scratch" / "compiled-person-days" / hierarchy / f"{scenario}_{gcm_variant}.parquet"
    #             out_root = ca_data.results_root(results_version) / hierarchy / f"temperature_person_days_{scenario}"
    #             mkdir(out_root, parents=True, exist_ok=True)
    #             out_path = out_root / f"{draw}.parquet"
    #             out_path.symlink_to(raw_path)
