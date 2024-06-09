import itertools
from collections.abc import Callable

import gcsfs
import pandas as pd
import xarray as xr


def load_cmip_metadata(
    tables: tuple[str, ...] = ("Amon", "day"),
    variables: tuple[str, ...] = ("tas", "pr"),
    experiments: tuple[str, ...] = (
        "historical",
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp585",
    ),
) -> pd.DataFrame:
    """Loads CMIP6 metadata for the given tables, variables, and experiments.

    Parameters
    ----------
    tables
        The tables to include.
    variables
        The variables to include.
    experiments
        The experiments to include.

    Returns
    -------
    pd.DataFrame
        CMIP6 metadata containing only the institutions and sources with all
        tables, variables, and experiments.
    """
    all_models = load_raw_cmip_metadata()
    models_and_params = filter_institutions_and_sources(
        all_models,
        tables,
        variables,
        experiments,
    )

    # There should be no duplicates here, but there are. I'm not going to investigate
    # why, but I'm just going to drop them.
    member_count = models_and_params.groupby(
        ["institution_id", "source_id", "member_id"]
    )["activity_id"].count()
    expected_count = len(tables) * len(variables) * len(experiments)
    member_mask = member_count == expected_count

    final_models = (
        models_and_params.set_index(["institution_id", "source_id", "member_id"])
        .loc[member_mask[member_mask].index]
        .reset_index()
    )

    # Filter to the models we need for the anomaly analysis.
    monthly_historical = (final_models["table_id"] == "Amon") & (
        final_models["experiment_id"] == "historical"
    )
    daily_scenario = (final_models["table_id"] == "day") & (
        final_models["experiment_id"] != "historical"
    )
    return final_models.loc[monthly_historical | daily_scenario]


def load_cmip_historical_data(path: str) -> xr.Dataset:
    """Loads a CMIP6 historical dataset from a zarr path.

    Parameters
    ----------
    path
        The path to the zarr store.

    Returns
    -------
    xr.Dataset
        The CMIP6 historical dataset.
    """
    reference_period = slice("1981-01-15", "2010-12-15")
    return (
        load_cmip_data(path)
        .sel(time=reference_period)
        .groupby("time.month")
        .mean("time")
    )


def load_cmip_experiment_data(path: str, year: str) -> xr.Dataset:
    """Loads a CMIP6 experiment dataset from a zarr path by day for a given year.

    Parameters
    ----------
    path
        The path to the zarr store.
    year
        The year to load.

    Returns
    -------
    xr.Dataset
        The CMIP6 experiment dataset for the given year.
    """ ""
    time_slice = slice(f"{year}-01", f"{year}-12")
    time_range = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    return load_cmip_data(path).sel(time=time_slice).interp_calendar(time_range)


##################
# Helper methods #
##################


def load_raw_cmip_metadata() -> pd.DataFrame:
    """Loads metadata containing information about all CMIP6 models."""
    path = "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
    return pd.read_csv(path)


def load_cmip_data(zarr_path: str) -> xr.Dataset:
    """Loads a CMIP6 dataset from a zarr path."""
    gcs = gcsfs.GCSFileSystem(token="anon")  # noqa: S106
    mapper = gcs.get_mapper(zarr_path)
    ds = xr.open_zarr(mapper, consolidated=True)
    lon = (ds.lon + 180) % 360 - 180
    ds = ds.assign_coords(lon=lon).sortby("lon")
    ds = ds.drop(
        ["lat_bnds", "lon_bnds", "time_bnds", "height", "time_bounds", "bnds"],
        errors="ignore",
    )
    return ds  # type: ignore[no-any-return]


def contains_combo(
    table: str,
    variable: str,
    experiment: str,
) -> Callable[[pd.DataFrame], bool]:
    """Get a function to check if a dataset contains a given cmip metadata combination.

    Parameters
    ----------
    table
        The table to check for.
    variable
        The variable to check for.
    experiment
        The experiment to check for.

    Returns
    -------
    Callable[[pd.DataFrame], bool]
        A function that checks if a dataset contains a given cmip metadata combination.
    """

    def _check(df: pd.DataFrame) -> bool:
        return (
            df["table_id"].eq(table)
            & df["variable_id"].eq(variable)
            & df["experiment_id"].eq(experiment)
        ).any()

    return _check


def filter_institutions_and_sources(
    cmip_meta: pd.DataFrame,
    tables: tuple[str, ...],
    variables: tuple[str, ...],
    experiments: tuple[str, ...],
) -> pd.DataFrame:
    """Filters a cmip metadata dataframe to only include models that have all
    combinations of the given tables, variables, and experiments.
    Parameters
    ----------
    cmip_meta
        CMIP metadata dataframe.
    tables
        The tables to include.
    variables
        The variables to include.
    experiments
        The experiments to include.
    Returns
    -------
    pd.DataFrame
        Filtered cmip metadata containing only the institutions and sources with all
        tables, variables, and experiments.
    """
    # First we filter down to all models from the institutions and sources that have
    # all the combinations of tables, variables, and experiments.
    masks = []
    for table, variable, experiment in itertools.product(
        tables, variables, experiments
    ):
        has_combo = cmip_meta.groupby(["institution_id", "source_id"]).apply(
            contains_combo(table, variable, experiment)
        )
        masks.append(has_combo)
    mask = pd.concat(masks, axis=1).all(axis=1)

    institutions_and_sources = mask[mask].index
    models_with_all_params = (
        cmip_meta.set_index(["institution_id", "source_id"])
        .loc[institutions_and_sources]
        .reset_index()
    )

    # Now we filter down to the specific subset of table/variable/experiment
    # combinations within the institutions and sources.
    param_mask = (
        models_with_all_params["table_id"].isin(tables)
        & models_with_all_params["variable_id"].isin(variables)
        & models_with_all_params["experiment_id"].isin(experiments)
    )
    models_and_params = models_with_all_params[param_mask]
    return models_and_params
