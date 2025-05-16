import itertools

import pandas as pd

import climate_data.constants as cdc
from climate_data.data import ClimateAggregateData, PopulationModelData


def load_populations(
    version: str,
    hierarchy_version: str,
    location_id: int,
    hierarchy: pd.DataFrame,
    ca_data: ClimateAggregateData,
    pm_data: PopulationModelData,
):
    all_pop = ca_data.load_population(version, hierarchy_version)
    loc_pop = (
        all_pop.loc[all_pop.location_id == location_id]
        .drop(columns=["location_id"])
        .rename(columns={"year": "year_id"})
        .reset_index(drop=True)
    )
    subnat_ids = hierarchy.loc[hierarchy.parent_id == location_id].location_id
    subnat_mask = all_pop.location_id.isin(list(set(subnat_ids) - {location_id}))
    subnat_pop = (
        all_pop.loc[subnat_mask]
        .rename(columns={"year": "year_id"})
        .set_index(["year_id", "location_id"])
        .sort_index()
        .unstack("location_id")
    )
    subnat_pop.columns = subnat_pop.columns.droplevel().rename(None)

    raking_pop = pm_data.load_raking_populations("fhs_2021")
    if location_id in raking_pop.location_id.unique():
        raking_pop = raking_pop.loc[
            raking_pop.location_id == location_id, ["year_id", "population"]
        ]
    else:
        raking_pop = pd.DataFrame(columns=["year_id", "population"])
    return loc_pop, subnat_pop, raking_pop


def load_climate_data(
    version: str,
    hierarchy_version: str,
    location_id: int,
    ca_data: ClimateAggregateData,
):
    climate_dfs = []
    for measure, scenario in itertools.product(
        cdc.AGGREGATION_MEASURES, cdc.AGGREGATION_SCENARIOS
    ):
        df = ca_data.load_results(
            version,
            hierarchy_version,
            scenario=scenario,
            measure=measure,
            location_id=location_id,
        )
        df = (
            df.loc[df.location_id == location_id]
            .assign(scenario=scenario, measure=measure)
            .drop(columns=["location_id"])
            .assign(measure=measure)
            .set_index(["measure", "scenario", "year_id"])
        )
        df = pd.concat(
            [
                df.mean(axis=1).rename("mean"),
                df.quantile(0.025, axis=1).rename("lower"),
                df.quantile(0.975, axis=1).rename("upper"),
            ],
            axis=1,
        )
        climate_dfs.append(df)
    climate_data = pd.concat(climate_dfs).sort_index()
    return climate_data


def get_locations_depth_first(hierarchy: pd.DataFrame) -> list[int]:
    """Return location ids sorted by a depth first search of the hierarchy.

    Locations at the same level are sorted alphabetically by name.
    """

    def _get_locations(location: pd.Series):
        locs = [location.location_id]

        children = hierarchy[
            (hierarchy.parent_id == location.location_id)
            & (hierarchy.location_id != location.location_id)
        ]
        for child in children.sort_values("location_ascii_name").itertuples():
            locs.extend(_get_locations(child))
        return locs

    top_locs = hierarchy[hierarchy.location_id == hierarchy.parent_id]
    locations = []
    for top_loc in top_locs.sort_values("location_ascii_name").itertuples():
        locations.extend(_get_locations(top_loc))

    return locations
