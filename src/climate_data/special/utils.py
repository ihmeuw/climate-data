import numba
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from climate_data.aggregate.utils import build_location_masks
from climate_data.data import PopulationModelData


def build_location_index(
    hierarchy: str,
    block_key: str,
    pm_data: PopulationModelData,
):
    climate_slice, bounds_map, location_mask = build_location_masks(
        hierarchy, block_key, pm_data
    )
    l_mask = -1 * np.ones_like(location_mask, dtype=np.int64)
    for i, loc_id in enumerate(bounds_map):
        l_mask[location_mask == loc_id] = i
    location_idx = l_mask.flatten()
    return climate_slice, list(bounds_map), location_idx


def _to_idx(arr, bins):
    """Convert an array of values to an array of indices into a set of bins.T

    Parameters
    ----------
    arr
       A N-dimensional array of values to convert.
    bins
       A 1-dimensional array of bins. Bins must be sorted, increasing,
       and not contain duplicates. Every value in `arr` will be converted
       to an index into `bins`, with values outside of `bins` being clipped
       to the nearest bin edge. Bins are 0-indexed and left-inclusive,
       so idx 0 contains values from -infinity to bins[1], idx 1 contains
       values from bins[1] to bins[2], etc.

    Returns:
        The array of indices into the bins.
    """
    return np.clip(np.digitize(arr, bins), 1, len(bins)) - 1


def to_idx(ds: xr.Dataset, bins: np.ndarray) -> np.ndarray:
    arr = ds["value"].to_numpy()
    idx = _to_idx(arr, bins)
    return idx.reshape(arr.shape[0], -1)


def get_temperature_coordinates(
    block_key: str,
    pm_data: PopulationModelData,
    temperature: xr.Dataset,
):
    pop = pm_data.load_results("2020q1", block_key)
    transformer = Transformer.from_crs(pop.crs, "EPSG:4326", always_xy=True)
    longitude = temperature["longitude"].to_numpy()
    latitude = temperature["latitude"].to_numpy()

    xcoords_pop, ycoords_pop = pop.x_coordinates(), pop.y_coordinates()[::-1]
    xx_pop, yy_pop = np.meshgrid(xcoords_pop, ycoords_pop)
    xx_pop_wgs84, yy_pop_wgs84 = transformer.transform(xx_pop, yy_pop)

    xidx = _to_idx(xx_pop_wgs84, longitude - 0.05)
    yidx = _to_idx(yy_pop_wgs84, latitude - 0.05)
    temp_coords = xidx.flatten() + yidx.flatten() * len(longitude)
    return temp_coords


@numba.njit
def compute_person_days(
    location_idx,
    temp_idx,
    tz_idx,
    population,
    temp_coords,
    out,
):
    # location_idx is (high_res_pixel) with values of output index (dim 0)
    # temp_idx is (days, low_res_pixel) with values of output index (dim 1)
    # tz_idx is (low_res_pixel) with values of output index (dim 2)
    # population is (high_res_pixel) with values of population count
    # temp_coords is (high_res_pixel) with output_values (low_res_pixel)
    # out is size (num_locations, num_temperature_bins, num_temperature_zones)
    for pix in range(location_idx.shape[0]):
        loc = location_idx[pix]
        if loc == -1:
            continue
        pop = population[pix]
        t_pix = temp_coords[pix]
        tz = tz_idx[t_pix]
        for day in range(temp_idx.shape[0]):
            temp = temp_idx[day, t_pix]
            out[loc, temp, tz] += pop


def aggregate_to_hierarchy(data: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Create all aggregate climate values for a given hierarchy from most-detailed data.

    Parameters
    ----------
    data
        The most-detailed climate data to aggregate.
    hierarchy
        The hierarchy to aggregate the data to.

    Returns
    -------
    pd.DataFrame
        The climate data with values for all levels of the hierarchy.
    """
    agg_cols = sorted(
        set(data.columns) - {"location_id", "year_id", "temperature_zone"}
    )

    results = data.set_index("location_id")

    # Most detailed locations can be at multiple levels of the hierarchy,
    # so we loop over all levels from most detailed to global, aggregating
    # level by level and appending the results to the data.

    for level in reversed(list(range(1, hierarchy.level.max() + 1))):
        level_mask = hierarchy.level == level
        parent_map = hierarchy.loc[level_mask].set_index("location_id").parent_id

        subset = results.loc[results.index.intersection(parent_map.index)]
        subset["parent_id"] = parent_map

        parent_values = (
            subset.groupby(["year_id", "parent_id", "temperature_zone"])[agg_cols]
            .sum()
            .reset_index()
            .rename(columns={"parent_id": "location_id"})
            .set_index("location_id")
        )
        results = pd.concat([results, parent_values])
    results = (
        results.reset_index()
        .sort_values(["location_id", "year_id", "temperature_zone"])
        .reset_index(drop=True)
    )

    return results
