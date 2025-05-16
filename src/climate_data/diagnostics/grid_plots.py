import click
import contextily as ctx
import matplotlib.pyplot as plt
import seaborn as sns
from requests.exceptions import HTTPError
from rra_tools import jobmon, plotting

import climate_data.cli_options as clio
import climate_data.constants as cdc
from climate_data.data import ClimateAggregateData, PopulationModelData
from climate_data.diagnostics.utils import load_climate_data, load_populations

FIG_SIZE = (35, 20)
GRID_SPEC_MARGINS = {"top": 0.92, "bottom": 0.08}
TITLE_FONTSIZE = 24
LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 12

TILE_PROVIDER = ctx.providers.Esri.WorldStreetMap


def safe_add_basemap(ax, provider=TILE_PROVIDER):
    try:
        ctx.add_basemap(ax, source=provider)
    except HTTPError as e:
        print(f"Error adding basemap: {e}")


def grid_plots_main(
    location_id: int,
    version: str,
    hierarchy_version: str,
    population_model_dir: str,
    output_dir: str,
    write: bool = True,
) -> plt.Figure:
    print(f"Running grid plots for {location_id} in {hierarchy_version}")
    pm_data = PopulationModelData(population_model_dir)
    ca_data = ClimateAggregateData(output_dir)

    print("Loading hierarchy and mapping shapes")
    hierarchy = pm_data.load_subset_hierarchy(hierarchy_version)

    a0 = pm_data.load_lsae_mapping_shapes(0)
    a1 = pm_data.load_lsae_mapping_shapes(1)
    a2 = pm_data.load_lsae_mapping_shapes(2)

    print("Loading populations")
    loc_pop, subnat_pop, raking_pop = load_populations(
        version, hierarchy_version, location_id, hierarchy, ca_data, pm_data
    )

    print("Loading climate data")
    climate_data = load_climate_data(version, hierarchy_version, location_id, ca_data)

    print("Setting up figure")
    fig = plt.figure(figsize=FIG_SIZE)

    grid_spec = fig.add_gridspec(
        ncols=1,
        nrows=2,
    )
    grid_spec.update(**GRID_SPEC_MARGINS)

    gs_top = grid_spec[0].subgridspec(
        ncols=2,
        nrows=1,
        width_ratios=[1, 2],
    )
    gs_bottom = grid_spec[1].subgridspec(
        ncols=5,
        nrows=2,
    )

    ax_map = fig.add_subplot(gs_top[0])

    print("Plotting map")
    if location_id == 1:
        bbox = (-180, -65, 180, 70)
        loc = a0.clip(bbox).to_crs("EPSG:3857")

        loc.boundary.plot(ax=ax_map, color="black", linewidth=0.5)

        xmin, ymin, xmax, ymax = loc.total_bounds
        ax_map.set_xlim(xmin, xmax)
        ax_map.set_ylim(ymin, ymax)

        safe_add_basemap(ax_map)
        plotting.strip_axes(ax_map)
    elif location_id in a0.location_id.unique():
        loc = a0[a0.location_id == location_id].to_crs("EPSG:3857")
        subnats = a1[
            a1.location_id.isin(
                hierarchy.loc[hierarchy.parent_id == location_id, "location_id"]
            )
        ].to_crs("EPSG:3857")

        subnats.boundary.plot(ax=ax_map, color="darkgrey", linewidth=0.5)
        loc.boundary.plot(ax=ax_map, color="red", linewidth=1)

        xmin, ymin, xmax, ymax = loc.total_bounds
        ax_map.set_xlim(xmin, xmax)
        ax_map.set_ylim(ymin, ymax)

        safe_add_basemap(ax_map)
        plotting.strip_axes(ax_map)
    elif location_id in a1.location_id.unique():
        loc = a1[a1.location_id == location_id].to_crs("EPSG:3857")
        parent_id = hierarchy.loc[
            hierarchy.location_id == location_id, "parent_id"
        ].values[0]
        parent = a0[a0.location_id == parent_id].to_crs("EPSG:3857")
        other_a1s = a1[
            a1.location_id.isin(
                hierarchy.loc[hierarchy.parent_id == parent_id, "location_id"]
            )
        ].to_crs("EPSG:3857")
        subnats = a2[
            a2.location_id.isin(
                hierarchy.loc[hierarchy.parent_id == location_id, "location_id"]
            )
        ].to_crs("EPSG:3857")

        subnats.boundary.plot(ax=ax_map, color="darkgrey", linewidth=0.5)
        other_a1s.boundary.plot(ax=ax_map, color="darkgrey", linewidth=1)
        parent.boundary.plot(ax=ax_map, color="black", linewidth=1)
        loc.boundary.plot(ax=ax_map, color="red", linewidth=1)

        xmin, ymin, xmax, ymax = parent.total_bounds
        ax_map.set_xlim(xmin, xmax)
        ax_map.set_ylim(ymin, ymax)

        safe_add_basemap(ax_map)
        plotting.strip_axes(ax_map)
    else:
        print(f"No shape found for {location_id}")
        ax_map.axis("off")

    print("Plotting population")
    ax_pop = fig.add_subplot(gs_top[1])
    if not raking_pop.empty:
        ax_pop.plot(
            raking_pop.year_id,
            raking_pop.population / 1e6,
            linestyle="--",
            color="black",
            linewidth=2,
        )
    ax_pop.plot(loc_pop.year_id, loc_pop.population / 1e6, color="black", linewidth=2)
    ax_pop.set_ylabel("Population (millions)", fontsize=LABEL_FONT_SIZE)
    ax_pop.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    ax_pop.set_ylim(0, None)
    sns.despine(ax=ax_pop)
    ax_pop.set_ylabel("Population (millions)", fontsize=LABEL_FONT_SIZE)

    if not subnat_pop.empty:
        ax_subnat = ax_pop.twinx()
        for loc_id in subnat_pop.columns:
            ax_subnat.plot(
                subnat_pop.index,
                subnat_pop[loc_id] / 1e6,
                color="firebrick",
                alpha=0.4,
                linewidth=1,
            )
        ax_subnat.set_ylabel(
            "Subnational Population (millions)",
            fontsize=LABEL_FONT_SIZE,
            color="firebrick",
        )

    measures = [
        ("mean_temperature", "Mean Temperature (°C)"),
        ("days_over_30C", "Days Over 30°C"),
        ("mean_low_temperature", "Mean Low Temperature (°C)"),
        ("mean_high_temperature", "Mean High Temperature (°C)"),
        ("total_precipitation", "Total Precipitation (mm)"),
        ("precipitation_days", "Precipitation Days"),
        ("malaria_suitability", "Malaria Suitability"),
        ("dengue_suitability", "Dengue Suitability"),
        ("wind_speed", "Wind Speed (m/s)"),
        ("relative_humidity", "Relative Humidity (%)"),
    ]

    print("Plotting climate data")
    for i, (measure, label) in enumerate(measures):
        col, row = divmod(i, 2)
        ax = fig.add_subplot(gs_bottom[row, col])
        for scenario, color in zip(
            cdc.AGGREGATION_SCENARIOS, ["dodgerblue", "forestgreen", "firebrick"], strict=False
        ):
            data = climate_data.loc[(measure, scenario)]
            ax.fill_between(data.index, data.lower, data.upper, alpha=0.1, color=color)
            ax.plot(data.index, data["mean"], label=scenario, color=color)
        ax.set_ylabel(label, fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
        sns.despine(ax=ax)

    location_name = hierarchy.loc[
        hierarchy.location_id == location_id, "location_name"
    ].values[0]
    fig.suptitle(f"{location_name} ({location_id})", fontsize=TITLE_FONTSIZE)

    print("Writing figure")
    if write:
        page_path = ca_data.grid_plots_page_path(
            version, hierarchy_version, location_id
        )
        plotting.write_or_show(fig, page_path)
    else:
        plotting.write_or_show(fig, None)
    return fig


@click.command()
@clio.with_agg_version()
@clio.with_location_id()
@clio.with_hierarchy(choices=["fhs_2021", "lsae_1209", "gbd_2021"])
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@clio.with_output_directory(cdc.AGGREGATE_ROOT)
def grid_plots_task(
    agg_version: str,
    location_id: int,
    hierarchy: str,
    population_model_dir: str,
    output_dir: str,
) -> None:
    grid_plots_main(
        location_id,
        agg_version,
        hierarchy,
        population_model_dir,
        output_dir,
    )


@click.command()
@clio.with_agg_version()
@clio.with_hierarchy(choices=["fhs_2021", "lsae_1209", "gbd_2021"], allow_all=True)
@clio.with_input_directory("population-model", cdc.POPULATION_MODEL_ROOT)
@clio.with_output_directory(cdc.AGGREGATE_ROOT)
@clio.with_queue()
def grid_plots(
    agg_version: str,
    hierarchy: list[str],
    population_model_dir: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(population_model_dir)
    ca_data = ClimateAggregateData(output_dir)
    jobs = []
    for h in hierarchy:
        max_level = {
            "fhs_2021": 100,
            "gbd_2021": 4,
            "lsae_1209": 2,
        }[h]

        loc_meta = pm_data.load_subset_hierarchy(h)
        loc_ids = loc_meta[loc_meta.level <= max_level].location_id.unique()
        jobs.extend((h, loc_id) for loc_id in loc_ids)

    print(f"Running {len(jobs)} jobs")

    jobmon.run_parallel(
        runner="cdtask diagnostics",
        task_name="grid_plots",
        flat_node_args=(
            ("hierarchy", "location-id"),
            jobs,
        ),
        task_args={
            "agg-version": agg_version,
            "population-model-dir": population_model_dir,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "8G",
            "runtime": "10m",
            "project": "proj_rapidresponse",
        },
        log_root=ca_data.log_dir("diagnostics_grid_plots"),
        max_attempts=3,
    )

    for h in hierarchy:
        loc_meta = pm_data.load_subset_hierarchy(h)
        plot_cache = ca_data.grid_plots_pages_root(version, h)
        output_path = ca_data.grid_plots_path(version, h)
        for loc_id in loc_meta.location_id.unique():
            if plot_cache.exists(loc_id):
                print(f"Skipping {loc_id} because it already exists")
                continue
            print(f"Processing {loc_id}")
            grid_plots_main(
                loc_id, version, h, population_model_dir, output_dir, write=False
            )
