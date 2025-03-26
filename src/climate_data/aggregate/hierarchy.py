"""Runner for the collate stage of the climate aggregates pipeline.

This module provides functions to collate block-level datasets into measure-level datasets.
The compilation process:
1. Loads raw results from all blocks for a given hierarchy, measure, and scenario
2. Aggregates the data up the location hierarchy
3. Produces views for subset hierarchies
4. Saves the results as measure-level datasets
"""

import click
import pandas as pd
import tqdm
from rra_tools import jobmon

from climate_data import cli_options as clio
from climate_data import constants as cdc
from climate_data.aggregate import utils
from climate_data.data import ClimateAggregateData, PopulationModelData


def hierarchy_main(
    agg_version: str,
    hierarchy: str,
    measure: str,
    scenario: str,
    population_model_dir: str,
    output_dir: str,
    *,
    progress_bar: bool = False,
) -> None:
    """Collate block-level datasets into measure/scenario-level datasets.

    This function:
    1. Loads all block-level datasets for a given hierarchy, measure, and scenario
    2. Combines them into a single dataset
    3. Aggregates the data up the location hierarchy
    4. Produces views for subset hierarchies
    5. Saves the results as measure-level datasets

    Parameters
    ----------
    agg_version
        The version identifier
    hierarchy
        The full aggregation hierarchy to process
    measure
        The climate measure to process
    scenario
        The climate scenario to process
    population_model_dir
        Path to the population model directory
    output_dir
        Path to save results
    progress_bar
        Whether to show a progress bar
    """
    print(f"Compiling {measure} for {scenario} in {hierarchy}")
    ca_data = ClimateAggregateData(output_dir)
    pm_data = PopulationModelData(population_model_dir)

    # Load hierarchy data for aggregation
    hierarchy_df = pm_data.load_subset_hierarchy(hierarchy)

    # Get all block keys
    modeling_frame = pm_data.load_modeling_frame()
    block_keys = modeling_frame.block_key.unique()

    # Load and combine all block-level results
    desc_template = "{draw:5} {block_key:15}"
    pbar = tqdm.tqdm(
        total=len(block_keys) * len(cdc.DRAWS),
        desc=desc_template.format(draw="DRAW", block_key="BLOCK KEY"),
        disable=not progress_bar,
    )

    all_results = []
    pop_df: pd.DataFrame | None = None

    for draw in cdc.DRAWS:
        save_population = (
            measure == "mean_temperature" and scenario == "ssp245" and draw == "000"
        )
        draw_results = []
        for block_key in block_keys:
            pbar.set_description(desc_template.format(draw=draw, block_key=block_key))

            draw_df = ca_data.load_raw_results(
                agg_version,
                hierarchy,
                block_key,
                draw,
                measure=measure,
                scenario=scenario,
            ).drop(columns=["scenario", "measure"])
            draw_results.append(draw_df)

            pbar.update()
        draw_df = (
            pd.concat(draw_results, ignore_index=True)
            .groupby(["location_id", "year_id"])
            .sum()
            .reset_index()
        )

        agg_df = utils.aggregate_climate_to_hierarchy(
            draw_df,
            hierarchy_df,
        ).set_index(["location_id", "year_id"])
        all_results.append(agg_df["value"].rename(draw))

        if save_population:
            pop_df = agg_df["population"].reset_index()

    pbar.close()

    combined_results = pd.concat(all_results, axis=1)

    # Produce views for subset hierarchies
    subset_hierarchies = cdc.HIERARCHY_MAP[hierarchy]
    for subset_hierarchy in subset_hierarchies:
        # Load the subset hierarchy
        subset_hierarchy_df = pm_data.load_subset_hierarchy(subset_hierarchy)

        # Filter results to only include locations in the subset hierarchy
        subset_location_ids = subset_hierarchy_df["location_id"].tolist()
        subset_results = combined_results.loc[subset_location_ids]

        # Save results for the subset hierarchy
        ca_data.save_results(
            subset_results,
            agg_version,
            subset_hierarchy,
            scenario,
            measure,
        )

        if pop_df is not None:
            subset_pop = pop_df.loc[subset_location_ids]
            ca_data.save_population(subset_pop, agg_version, subset_hierarchy)


@click.command()
@clio.with_agg_version()
@clio.with_hierarchy()
@clio.with_agg_measure()
@clio.with_agg_scenario()
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@clio.with_output_directory(cdc.AGGREGATE_ROOT)
@clio.with_progress_bar()
def hierarchy_task(
    agg_version: str,
    hierarchy: str,
    agg_measure: str,
    agg_scenario: str,
    population_model_dir: str,
    output_dir: str,
    *,
    progress_bar: bool,
) -> None:
    """Collate block-level datasets into measure-level datasets.

    This command collates results for a specific hierarchy, measure, and scenario.
    """
    hierarchy_main(
        agg_version,
        hierarchy,
        agg_measure,
        agg_scenario,
        population_model_dir,
        output_dir,
        progress_bar=progress_bar,
    )


@click.command()
@clio.with_agg_version()
@clio.with_hierarchy(allow_all=True)
@clio.with_agg_measure(allow_all=True)
@clio.with_agg_scenario(allow_all=True)
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@clio.with_output_directory(cdc.AGGREGATE_ROOT)
@clio.with_queue()
def hierarchy(
    agg_version: str,
    hierarchy: list[str],
    agg_measure: list[str],
    agg_scenario: list[str],
    population_model_dir: str,
    output_dir: str,
    queue: str,
) -> None:
    """Collate block-level datasets into measure-level datasets.

    This command:
    1. Identifies which combinations of hierarchy, measure, and scenario need compilation
    2. Creates parallel jobs to collate each combination
    3. Runs the jobs using jobmon
    """
    ca_data = ClimateAggregateData(output_dir)

    n_jobs = len(hierarchy) * len(agg_measure) * len(agg_scenario)
    print(f"Running {n_jobs} jobs")

    jobmon.run_parallel(
        runner="cdtask aggregate",
        task_name="hierarchy",
        node_args={
            "hierarchy": hierarchy,
            "agg-measure": agg_measure,
            "agg-scenario": agg_scenario,
        },
        task_args={
            "agg-version": agg_version,
            "population-model-dir": population_model_dir,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "50G",
            "runtime": "200m",
            "project": "proj_rapidresponse",
        },
        log_root=ca_data.log_dir("aggregate_hierarchy"),
        max_attempts=3,
    )
