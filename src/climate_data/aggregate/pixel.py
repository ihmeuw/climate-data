import itertools

import click
import numpy as np
import pandas as pd
import tqdm
from rra_tools import jobmon

from climate_data import cli_options as clio
from climate_data import constants as cdc
from climate_data.aggregate import utils
from climate_data.data import (
    ClimateAggregateData,
    ClimateData,
    PopulationModelData,
)
from climate_data.utils import to_raster


def pixel_main(
    agg_version: str,
    block_key: str,
    draw: str,
    hierarchy: str,
    population_model_root: str,
    climate_data_root: str,
    output_dir: str,
    *,
    progress_bar: bool = False,
) -> None:
    print(f"Aggregating draw {draw} for {hierarchy}")
    pm_data = PopulationModelData(population_model_root)
    cd_data = ClimateData(climate_data_root, read_only=True)
    ca_data = ClimateAggregateData(output_dir)

    print("Building location masks")
    climate_slice, bounds_map = utils.build_location_masks(
        hierarchy, block_key, pm_data
    )

    years = cdc.ALL_YEARS
    measures = cdc.AGGREGATION_MEASURES
    scenarios = cdc.AGGREGATION_SCENARIOS

    total_iterations = len(years) * len(measures) * len(scenarios)
    desc_template = "{measure:10} {scenario:6} {year:4}"
    pbar = tqdm.tqdm(
        total=total_iterations,
        desc=desc_template.format(
            measure="MEASURE",
            scenario="SCENARIO",
            year="YEAR",
        ),
        disable=not progress_bar,
    )

    result_records = []
    for measure, scenario in itertools.product(measures, scenarios):
        ds = cd_data.load_draw_results(scenario, measure, draw).sel(**climate_slice)  # type: ignore[arg-type]
        for year in years:
            pbar.set_description(
                desc_template.format(measure=measure, scenario=scenario, year=year)
            )
            # Load population data and grab the underlying ndarray (we don't want the metadata)
            pop_raster = pm_data.load_results(f"{year}q1", block_key)
            pop_arr = pop_raster._ndarray  # noqa: SLF001

            # Pull out and rasterize the climate data for the current year
            clim_arr = (
                to_raster(ds.sel(year=year)["value"], no_data_value=np.nan)  # noqa: SLF001
                .resample_to(pop_raster, "nearest")
                .astype(np.float32)
                ._ndarray
            )

            weighted_clim_arr = pop_arr * clim_arr  # type: ignore[operator]

            for location_id, (rows, cols, loc_mask) in bounds_map.items():
                # Subset and mask the weighted climate and population, then sum
                # all non-nan values
                loc_weighted_clim = np.nansum(weighted_clim_arr[rows, cols][loc_mask])
                loc_pop = np.nansum(pop_arr[rows, cols][loc_mask])

                result_records.append(
                    (location_id, year, scenario, measure, loc_weighted_clim, loc_pop)
                )
            pbar.update()

    pbar.close()
    results = pd.DataFrame(
        result_records,
        columns=[
            "location_id",
            "year_id",
            "scenario",
            "measure",
            "weighted_climate",
            "population",
        ],
    ).sort_values(by=["location_id", "year_id"])
    ca_data.save_raw_results(
        results,
        agg_version,
        hierarchy,
        block_key,
        draw,
    )


@click.command()
@clio.with_agg_version()
@clio.with_block_key()
@clio.with_draw()
@clio.with_hierarchy()
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@clio.with_input_directory("climate-data", cdc.MODEL_ROOT)
@clio.with_output_directory(cdc.AGGREGATE_ROOT)
@clio.with_progress_bar()
def pixel_task(
    agg_version: str,
    block_key: str,
    draw: str,
    hierarchy: str,
    population_model_dir: str,
    climate_data_dir: str,
    output_dir: str,
    *,
    progress_bar: bool,
) -> None:
    pixel_main(
        agg_version,
        block_key,
        draw,
        hierarchy,
        population_model_dir,
        climate_data_dir,
        output_dir,
        progress_bar=progress_bar,
    )


@click.command()
@clio.with_agg_version()
@clio.with_block_key(allow_all=True)
@clio.with_draw(allow_all=True)
@clio.with_hierarchy(allow_all=True)
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@clio.with_input_directory("climate-data", cdc.MODEL_ROOT)
@clio.with_output_directory(cdc.AGGREGATE_ROOT)
@clio.with_queue()
def pixel(
    agg_version: str,
    block_key: str,
    draw: list[str],
    hierarchy: list[str],
    population_model_dir: str,
    climate_data_dir: str,
    output_dir: str,
    queue: str,
) -> None:
    ca_data = ClimateAggregateData(output_dir)
    pm_data = PopulationModelData(population_model_dir)
    modeling_frame = pm_data.load_modeling_frame()
    block_keys = modeling_frame["block_key"].unique().tolist()
    block_keys = clio.convert_choice(block_key, block_keys)

    jobs = []
    for h, b, d in itertools.product(hierarchy, block_keys, draw):
        if not ca_data.raw_results_path(agg_version, h, b, d).exists():
            jobs.append((h, b, d))
    jobs = list(set(jobs))

    print(f"Running {len(jobs)} jobs")

    jobmon.run_parallel(
        runner="cdtask aggregate",
        task_name="pixel",
        flat_node_args=(
            ("hierarchy", "block-key", "draw"),
            jobs,
        ),
        task_args={
            "agg-version": agg_version,
            "population-model-dir": population_model_dir,
            "climate-data-dir": climate_data_dir,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "8G",
            "runtime": "300m",
            "project": "proj_rapidresponse",
        },
        log_root=ca_data.log_dir("aggregate_pixel"),
        max_attempts=3,
    )
