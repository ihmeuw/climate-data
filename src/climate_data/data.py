"""
Climate Data Management
-----------------------

This module provides a class for managing the climate data used in the project. It includes methods for
loading and saving data, as well as for accessing the various directories where data is stored. This
abstraction allows for easy access to the data and ensures that all data is stored in a consistent
and organized manner. It also provides a central location for managing the data, which makes it easier
to update and maintain the path structure of the data as needed.

This module generally does not load or process data itself, though some exceptions are made for metadata
which is generally loaded and cached on disk.

The main classes are:
- PopulationModelData: Handles data from the gridded population modeling pipeline.
    This includes population estimates and projections as well as the location hierarchies
    for the population data. This class provides read-only access to the data.
- ClimateData: Handles gridded climate data from the climate downscaling pipeline.
    This includes climate data for different scenarios and measures. This class
    provides read and write access to the data.
- ClimateAggregateData: Handles the output data structure for climate aggregates.
    This includes raw results at the block level, final results at the measure level,
    and versioned results for different pipeline versions. This class provides both
    read and write access to the data.
"""

from collections.abc import Collection
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import pandas as pd
import rasterra as rt
import xarray as xr
import yaml
from rra_tools.shell_tools import mkdir, touch

from climate_data import constants as cdc


class PopulationModelData:
    """Handles population data and location hierarchies.

    This class manages:
    1. Population projections at different time points
    2. Location hierarchies (GBD, LSAE, etc.)
    3. Spatial data for aggregation

    The population data is used as weights when aggregating climate data
    to different location hierarchies.
    """

    def __init__(
        self,
        root: str | Path = cdc.POPULATION_MODEL_ROOT,
    ) -> None:
        """Initialize the population model data manager.

        Parameters
        ----------
        root : str | Path
            Path to the population model root directory
        """
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """Get the root directory for population model data."""
        return self._root

    @property
    def results(self) -> Path:
        """Get the directory containing current model results."""
        return Path(self.root, "results") / "current"

    @property
    def model_spec_path(self) -> Path:
        """Get the path to the model specification file."""
        return self.results / "specification.yaml"

    def load_model_spec(self) -> dict[str, Any]:
        """Load the model specification file.

        Returns
        -------
        dict
            The model specification containing paths and parameters
        """
        return cast(dict[str, Any], yaml.safe_load(self.model_spec_path.read_text()))

    def load_modeling_frame(self) -> gpd.GeoDataFrame:
        """Load the modeling frame containing spatial information.

        The modeling frame is a subdivision of the world into equal-area blocks.
        Each block is assigned a unique key that is used to parallelize
        pipeline steps in both population modeling and in this pipeline's
        aggregation step.

        Returns
        -------
        gpd.GeoDataFrame
            The modeling frame with spatial information and block keys
        """
        model_spec = self.load_model_spec()
        raw_root = Path(model_spec["output_root"])
        model_frame_path = raw_root.parent.parent / "modeling_frame.parquet"
        return gpd.read_parquet(model_frame_path)

    def load_results(self, time_point: str, block_key: str) -> rt.RasterArray:
        """Load population results for a specific time point and block.

        Parameters
        ----------
        time_point
            The time point to load (e.g. "2020q1")
        block_key
            The block key to load (e.g. "B-0021X-0003Y")

        Returns
        -------
        rt.RasterArray
            The population raster data
        """
        model_spec = self.load_model_spec()
        raw_root = Path(model_spec["output_root"])
        path = raw_root / "raked_predictions" / time_point / f"{block_key}.tif"
        return rt.load_raster(path)

    @property
    def raking_data(self) -> Path:
        """Get the directory containing data used to rake the population estimates.

        Raking enforces admin-level consistency between gridded population data
        and GBD/FHS population estimates. We'll use these same hierarchies to
        aggregate the climate data.

        """
        return self.root / "admin-inputs" / "raking"

    def load_raking_shapes(
        self, full_aggregation_hierarchy: str, bounds: tuple[float, float, float, float]
    ) -> gpd.GeoDataFrame:
        """Load shapes for a full aggregation hierarchy within given bounds.

        Parameters
        ----------
        full_aggregation_hierarchy
            The full aggregation hierarchy to load (e.g. "gbd_2021")
        bounds
            The bounds to load (xmin, ymin, xmax, ymax)

        Returns
        -------
        gpd.GeoDataFrame
            The shapes for the given hierarchy and bounds
        """
        if full_aggregation_hierarchy == "gbd_2021":
            shape_path = (
                self.raking_data / f"shapes_{full_aggregation_hierarchy}.parquet"
            )
            gdf = gpd.read_parquet(shape_path, bbox=bounds)

            # We're using population data here instead of a hierarchy because
            # The populations include extra locations we've supplemented that aren't
            # modeled in GBD (e.g. locations with zero population or places that
            # GBD uses population scalars from WPP to model)
            pop_path = (
                self.raking_data / f"population_{full_aggregation_hierarchy}.parquet"
            )
            pop = pd.read_parquet(pop_path)

            keep_cols = ["location_id", "location_name", "most_detailed", "parent_id"]
            keep_mask = (
                (pop.year_id == pop.year_id.max())  # Year doesn't matter
                & (pop.most_detailed == 1)
            )
            out = gdf.merge(pop.loc[keep_mask, keep_cols], on="location_id", how="left")
        elif full_aggregation_hierarchy in ["lsae_1209", "lsae_1285"]:
            # This is only a2 geoms, so already most detailed
            shape_path = (
                self.raking_data
                / "gbd-inputs"
                / f"shapes_{full_aggregation_hierarchy}_a2.parquet"
            )
            out = gpd.read_parquet(shape_path, bbox=bounds)
        else:
            msg = f"Unknown pixel hierarchy: {full_aggregation_hierarchy}"
            raise ValueError(msg)
        return out

    def load_subset_hierarchy(self, subset_hierarchy: str) -> pd.DataFrame:
        """Load a subset location hierarchy.

        The subset hierarchy might be equal to the full aggregation hierarchy,
        but it might also be a subset of the full aggregation hierarchy.
        These hierarchies are used to provide different views of aggregated
        climate data.

        Parameters
        ----------
        subset_hierarchy
            The administrative hierarchy to load (e.g. "gbd_2021")

        Returns
        -------
        pd.DataFrame
            The hierarchy data with parent-child relationships
        """
        allowed_hierarchies = ["gbd_2021", "fhs_2021", "lsae_1209", "lsae_1285"]
        if subset_hierarchy not in allowed_hierarchies:
            msg = f"Unknown admin hierarchy: {subset_hierarchy}"
            raise ValueError(msg)
        path = self.raking_data / "gbd-inputs" / f"hierarchy_{subset_hierarchy}.parquet"
        return pd.read_parquet(path)


class ClimateData:
    """Class for managing the climate data used in the project."""

    def __init__(
        self,
        root: str | Path = cdc.MODEL_ROOT,
        *,
        read_only: bool = False,
    ) -> None:
        self._root = Path(root)
        self._credentials_root = self._root / "credentials"
        self._read_only = read_only
        if not read_only:
            self._create_model_root()

    def _create_model_root(self) -> None:
        mkdir(self.root, exist_ok=True)
        mkdir(self.credentials_root, exist_ok=True)

        mkdir(self.extracted_data, exist_ok=True)
        mkdir(self.extracted_era5, exist_ok=True)
        mkdir(self.extracted_cmip6, exist_ok=True)
        mkdir(self.ncei_climate_stations, exist_ok=True)
        mkdir(self.open_topography_elevation, exist_ok=True)
        mkdir(self.rub_local_climate_zones, exist_ok=True)

        mkdir(self.downscale_model, exist_ok=True)
        mkdir(self.predictors, exist_ok=True)
        mkdir(self.training_data, exist_ok=True)

        mkdir(self.results, exist_ok=True)
        mkdir(self.results_metadata, exist_ok=True)
        mkdir(self.daily_results, exist_ok=True)
        mkdir(self.raw_daily_results, exist_ok=True)
        mkdir(self.annual_results, exist_ok=True)
        mkdir(self.raw_annual_results, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def credentials_root(self) -> Path:
        return self._credentials_root

    ##################
    # Extracted data #
    ##################

    @property
    def extracted_data(self) -> Path:
        return self.root / "extracted_data"

    @property
    def extracted_era5(self) -> Path:
        return self.extracted_data / "era5"

    def extracted_era5_path(
        self, dataset: str, variable: str, year: int | str, month: str
    ) -> Path:
        return self.extracted_era5 / f"{dataset}_{variable}_{year}_{month}.nc"

    @property
    def extracted_cmip6(self) -> Path:
        return self.extracted_data / "cmip6"

    def load_koppen_geiger_model_inclusion(
        self, *, return_full_criteria: bool = False
    ) -> pd.DataFrame:
        meta_path = self.extracted_cmip6 / "koppen_geiger_model_inclusion.parquet"

        if not meta_path.exists():
            df = pd.read_html(
                "https://www.nature.com/articles/s41597-023-02549-6/tables/3"
            )[0]
            df.columns = [  # type: ignore[assignment]
                "source_id",
                "member_count",
                "mean_trend",
                "std_dev_trend",
                "transient_climate_response",
                "equilibrium_climate_sensitivity",
                "included_raw",
            ]
            df["included"] = df["included_raw"].apply({"Yes": True, "No": False}.get)
            save_parquet(df, meta_path)

        df = pd.read_parquet(meta_path)
        if return_full_criteria:
            return df
        return df[["source_id", "included"]]

    def load_cmip6_metadata(self) -> pd.DataFrame:
        meta_path = self.extracted_cmip6 / "cmip6-metadata.parquet"

        if not meta_path.exists():
            external_path = "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
            meta = pd.read_csv(external_path)
            save_parquet(meta, meta_path)

        return pd.read_parquet(meta_path)

    def extracted_cmip6_path(
        self,
        variable: str,
        experiment: str,
        gcm_member: str,
    ) -> Path:
        return self.extracted_cmip6 / f"{variable}_{experiment}_{gcm_member}.nc"

    def get_gcms(
        self,
        source_variables: Collection[str],
    ) -> list[str]:
        inclusion_meta = self.load_scenario_inclusion_metadata()[source_variables]
        inclusion_meta = inclusion_meta[inclusion_meta.all(axis=1)]
        return [
            f"{model}_{variant}" for model, variant in inclusion_meta.index.tolist()
        ]

    @property
    def ncei_climate_stations(self) -> Path:
        return self.extracted_data / "ncei_climate_stations"

    def save_ncei_climate_stations(self, df: pd.DataFrame, year: int | str) -> None:
        if self._read_only:
            msg = "Cannot save NCEI climate stations to read-only data"
            raise ValueError(msg)
        path = self.ncei_climate_stations / f"{year}.parquet"
        save_parquet(df, path)

    def load_ncei_climate_stations(self, year: int | str) -> pd.DataFrame:
        return pd.read_parquet(self.ncei_climate_stations / f"{year}.parquet")

    @property
    def open_topography_elevation(self) -> Path:
        return self.extracted_data / "open_topography_elevation"

    @property
    def rub_local_climate_zones(self) -> Path:
        return self.extracted_data / "rub_local_climate_zones"

    ###################
    # Downscale model #
    ###################

    @property
    def downscale_model(self) -> Path:
        return self.root / "downscale_model"

    @property
    def predictors(self) -> Path:
        return self.downscale_model / "predictors"

    def save_predictor(
        self,
        predictor: rt.RasterArray,
        name: str,
        lat_start: int,
        lon_start: int,
    ) -> None:
        if self._read_only:
            msg = "Cannot save predictors to read-only data"
            raise ValueError(msg)
        path = self.predictors / f"{name}_{lat_start}_{lon_start}.tif"
        save_raster(predictor, path)

    def load_predictor(self, name: str) -> rt.RasterArray:
        paths = list(self.predictors.glob(f"{name}_*.tif"))
        return rt.load_mf_raster(paths)

    @property
    def training_data(self) -> Path:
        return self.downscale_model / "training_data"

    def save_training_data(self, df: pd.DataFrame, year: int | str) -> None:
        if self._read_only:
            msg = "Cannot save training data to read-only data"
            raise ValueError(msg)
        path = self.training_data / f"{year}.parquet"
        save_parquet(df, path)

    def load_training_data(self, year: int | str) -> pd.DataFrame:
        return pd.read_parquet(self.training_data / f"{year}.parquet")

    ###########
    # Results #
    ###########

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def results_metadata(self) -> Path:
        return self.results / "metadata"

    def save_scenario_metadata(self, df: pd.DataFrame) -> None:
        if self._read_only:
            msg = "Cannot save scenario metadata to read-only data"
            raise ValueError(msg)
        path = self.results_metadata / "scenario_metadata.parquet"
        save_parquet(df, path)

    def load_scenario_metadata(self) -> pd.DataFrame:
        path = self.results_metadata / "scenario_metadata.parquet"
        return pd.read_parquet(path)

    def save_scenario_inclusion_metadata(self, df: pd.DataFrame) -> None:
        if self._read_only:
            msg = "Cannot save scenario inclusion metadata to read-only data"
            raise ValueError(msg)
        # Need to save to our scripts directory for doc building
        scripts_root = Path(__file__).parent.parent.parent / "scripts"
        for root_dir in [self.results_metadata, scripts_root]:
            path = root_dir / "scenario_inclusion_metadata.parquet"
            save_parquet(df, path)

    def load_scenario_inclusion_metadata(self) -> pd.DataFrame:
        path = self.results_metadata / "scenario_inclusion_metadata.parquet"
        return pd.read_parquet(path)

    @property
    def daily_results(self) -> Path:
        return self.results / "daily"

    @property
    def raw_daily_results(self) -> Path:
        return self.daily_results / "raw"

    def raw_daily_results_path(
        self,
        scenario: str,
        variable: str,
        year: int | str,
        gcm_member: str,
    ) -> Path:
        return self.raw_daily_results / scenario / variable / f"{year}_{gcm_member}.nc"

    def save_raw_daily_results(
        self,
        results_ds: xr.Dataset,
        scenario: str,
        variable: str,
        year: int | str,
        gcm_member: str,
        encoding_kwargs: dict[str, Any],
    ) -> None:
        if self._read_only:
            msg = "Cannot save raw daily results to read-only data"
            raise ValueError(msg)
        path = self.raw_daily_results_path(scenario, variable, year, gcm_member)
        mkdir(path.parent, exist_ok=True, parents=True)
        save_xarray(results_ds, path, encoding_kwargs)

    def daily_results_path(
        self,
        scenario: str,
        variable: str,
        year: int | str,
    ) -> Path:
        return self.daily_results / scenario / variable / f"{year}.nc"

    def save_daily_results(
        self,
        results_ds: xr.Dataset,
        scenario: str,
        variable: str,
        year: int | str,
        encoding_kwargs: dict[str, Any],
    ) -> None:
        if self._read_only:
            msg = "Cannot save daily results to read-only data"
            raise ValueError(msg)
        path = self.daily_results_path(scenario, variable, year)
        mkdir(path.parent, exist_ok=True, parents=True)
        save_xarray(results_ds, path, encoding_kwargs)

    def load_daily_results(
        self,
        scenario: str,
        variable: str,
        year: int | str,
    ) -> xr.Dataset:
        results_path = self.daily_results_path(scenario, variable, year)
        return xr.open_dataset(results_path)

    @property
    def annual_results(self) -> Path:
        return self.results / "annual"

    @property
    def raw_annual_results(self) -> Path:
        return self.annual_results / "raw"

    def raw_annual_results_path(
        self,
        scenario: str,
        variable: str,
        year: int | str,
        gcm_member: str,
    ) -> Path:
        return self.raw_annual_results / scenario / variable / f"{year}_{gcm_member}.nc"

    def save_raw_annual_results(
        self,
        results_ds: xr.Dataset,
        scenario: str,
        variable: str,
        year: int | str,
        gcm_member: str,
        encoding_kwargs: dict[str, Any],
    ) -> None:
        if self._read_only:
            msg = "Cannot save raw annual results to read-only data"
            raise ValueError(msg)
        path = self.raw_annual_results_path(scenario, variable, year, gcm_member)
        mkdir(path.parent, exist_ok=True, parents=True)
        save_xarray(results_ds, path, encoding_kwargs)

    @property
    def compiled_annual_results(self) -> Path:
        return self.raw_annual_results / "compiled"

    def compiled_annual_results_path(
        self,
        scenario: str,
        variable: str,
        gcm_member: str,
    ) -> Path:
        return self.compiled_annual_results / scenario / variable / f"{gcm_member}.nc"

    def save_compiled_annual_results(
        self,
        results_ds: xr.Dataset,
        scenario: str,
        variable: str,
        gcm_member: str,
    ) -> None:
        if self._read_only:
            msg = "Cannot save compiled annual results to read-only data"
            raise ValueError(msg)
        path = self.compiled_annual_results_path(scenario, variable, gcm_member)
        mkdir(path.parent, exist_ok=True, parents=True)
        touch(path, clobber=True)
        results_ds.to_netcdf(path)

    def annual_results_path(
        self,
        scenario: str,
        variable: str,
        draw: int | str,
    ) -> Path:
        return self.annual_results / scenario / variable / f"{draw:0>3}.nc"

    def link_annual_draw(
        self,
        draw: int | str,
        scenario: str,
        variable: str,
        gcm_member: str,
    ) -> None:
        if self._read_only:
            msg = "Cannot link annual draw to read-only data"
            raise ValueError(msg)
        source_path = self.compiled_annual_results_path(scenario, variable, gcm_member)
        dest_path = self.annual_results_path(scenario, variable, draw)
        mkdir(dest_path.parent, exist_ok=True, parents=True)
        if dest_path.exists():
            dest_path.unlink()
        dest_path.symlink_to(source_path)

    def draw_results_path(self, scenario: str, measure: str, draw: str) -> Path:
        """Get the path to annual results for a specific scenario, measure, and draw.

        Parameters
        ----------
        scenario
            The climate scenario (e.g. "ssp126")
        measure
            The climate measure (e.g. "mean_temperature")
        draw
            The draw of the climate data to load (e.g. "000")

        Returns
        -------
        Path
            The path to the results file
        """
        return self.annual_results / scenario / measure / f"{draw}.nc"

    def load_draw_results(self, scenario: str, measure: str, draw: str) -> xr.Dataset:
        """Load annual climate results for a specific scenario, measure, and draw.

        Parameters
        ----------
        scenario
            The climate scenario (e.g. "ssp126")
        measure
            The climate measure (e.g. "mean_temperature")
        draw
            The draw of the climate data to load (e.g. "000")

        Returns
        -------
        xr.Dataset
            The climate data in xarray format
        """
        path = self.annual_results_path(scenario, measure, draw)
        ds = xr.open_dataset(path, decode_coords="all")
        ds = ds.rio.write_crs("EPSG:4326")
        return ds


def save_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a pandas DataFrame to a file with standard parameters.

    Parameters
    ----------
    df
        The DataFrame to save.
    output_path
        The path to save the DataFrame to.
    """
    touch(output_path, clobber=True)
    df.to_parquet(output_path)


class ClimateAggregateData:
    """Manages the output data structure for climate aggregates.

    This class manages the file organization and paths for:
    1. Reading and writing raw results at block level
    2. Reading and writing final results at measure and scenario level
    3. Versioning of results
    """

    def __init__(
        self,
        root: str | Path = cdc.AGGREGATE_ROOT,
    ) -> None:
        """Initialize the climate aggregate data manager.

        Parameters
        ----------
        root
            Path to the model root directory
        """
        self._root = Path(root)
        self._create_model_root()

    def _create_model_root(self) -> None:
        """Create the model root directory and logs directory."""
        mkdir(self.root, exist_ok=True)
        mkdir(self.logs, exist_ok=True)

    @property
    def root(self) -> Path:
        """Get the root directory for model data."""
        return self._root

    @property
    def logs(self) -> Path:
        """Get the directory for log files."""
        return self.root / "logs"

    def log_dir(self, step_name: str) -> Path:
        """Get the directory for logs from a specific pipeline step.

        Parameters
        ----------
        step_name
            The name of the pipeline step

        Returns
        -------
        Path
            The directory for step-specific logs
        """
        return self.logs / step_name

    def version_root(self, version: str) -> Path:
        """Get the root directory for a specific version.

        Parameters
        ----------
        version
            The version identifier

        Returns
        -------
        Path
            The directory for version-specific data
        """
        return self.root / version

    def raw_results_root(self, version: str) -> Path:
        """Get the directory for raw results (block-level).

        Parameters
        ----------
        version
            The version identifier

        Returns
        -------
        Path
            The directory for raw results
        """
        return self.version_root(version) / "raw-results"

    def raw_results_path(
        self, version: str, hierarchy: str, block_key: str, draw: str
    ) -> Path:
        """Get the path to raw results for a specific hierarchy, block, and draw.

        Parameters
        ----------
        version
            The version identifier
        hierarchy
            The location hierarchy
        block_key
            The block key
        draw
            The draw of the climate data (e.g. "000")

        Returns
        -------
        Path
            The path to the raw results file
        """
        root = self.raw_results_root(version)
        return root / hierarchy / block_key / f"{draw}.parquet"

    def save_raw_results(
        self,
        df: pd.DataFrame,
        version: str,
        hierarchy: str,
        block_key: str,
        draw: str,
    ) -> None:
        """Save raw results for a specific hierarchy, block, and draw.

        Parameters
        ----------
        df
            The results to save
        version
            The version identifier
        hierarchy
            The location hierarchy
        block_key
            The block key
        draw
            The draw of the climate data to save (e.g. "000")
        """
        path = self.raw_results_path(version, hierarchy, block_key, draw)
        mkdir(path.parent, exist_ok=True, parents=True)
        touch(path, clobber=True)
        df.to_parquet(path)

    def load_raw_results(
        self,
        version: str,
        hierarchy: str,
        block_key: str,
        draw: str,
        measure: str | None = None,
        scenario: str | None = None,
    ) -> pd.DataFrame:
        """Load raw results for a specific hierarchy, block, and draw.

        Parameters
        ----------
        version
            The version identifier
        hierarchy
            The location hierarchy
        block_key
            The block key
        draw
            The draw of the climate data to load (e.g. "000")
        measure
            If provided, filter results to only include this measure
        scenario
            If provided, filter results to only include this scenario

        Returns
        -------
        pd.DataFrame
            The raw results
        """
        path = self.raw_results_path(version, hierarchy, block_key, draw)

        # Build filters for parquet's read_parquet function
        filters = []
        if measure is not None:
            filters.append(("measure", "==", measure))
        if scenario is not None:
            filters.append(("scenario", "==", scenario))

        return pd.read_parquet(path, filters=filters)

    def results_root(self, version: str) -> Path:
        """Get the directory for final results (measure-level).

        Parameters
        ----------
        version
            The version identifier

        Returns
        -------
        Path
            The directory for final results
        """
        return self.version_root(version) / "results"

    def population_path(self, version: str, hierarchy: str) -> Path:
        """Get the path to population data for a specific hierarchy.

        Parameters
        ----------
        version
            The version identifier
        hierarchy
            The location hierarchy

        Returns
        -------
        Path
            The path to the population data file
        """
        return self.results_root(version) / hierarchy / "population.parquet"

    def save_population(self, df: pd.DataFrame, version: str, hierarchy: str) -> None:
        """Save population data for a specific hierarchy.

        Parameters
        ----------
        df
            The population data to save
        version
            The version identifier
        hierarchy
            The location hierarchy
        """
        path = self.population_path(version, hierarchy)
        mkdir(path.parent, exist_ok=True, parents=True)
        touch(path, clobber=True)
        df.to_parquet(path)

    def load_population(
        self, version: str, hierarchy: str, location_id: int | None = None
    ) -> pd.DataFrame:
        """Load population data for a specific hierarchy and optionally location.

        Parameters
        ----------
        version
            The version identifier
        hierarchy
            The location hierarchy
        location_id
            If provided, load only data for this location

        Returns
        -------
        pd.DataFrame
            The population data
        """
        path = self.population_path(version, hierarchy)
        if location_id is not None:
            filters = [("location_id", "==", location_id)]
            return pd.read_parquet(path, filters=filters)
        return pd.read_parquet(path)

    def results_path(
        self, version: str, hierarchy: str, scenario: str, measure: str
    ) -> Path:
        """Get the path to final results for a specific scenario and measure.

        Parameters
        ----------
        version
            The version identifier
        hierarchy
            The location hierarchy
        scenario
            The climate scenario
        measure
            The climate measure

        Returns
        -------
        Path
            The path to the results file
        """
        return self.results_root(version) / hierarchy / f"{measure}_{scenario}.parquet"

    def save_results(
        self,
        df: pd.DataFrame,
        version: str,
        hierarchy: str,
        scenario: str,
        measure: str,
    ) -> None:
        """Save final results for a specific scenario and measure.

        Parameters
        ----------
        df
            The results to save
        version
            The version identifier
        hierarchy
            The location hierarchy
        scenario
            The climate scenario
        measure
            The climate measure
        """
        path = self.results_path(version, hierarchy, scenario, measure)
        mkdir(path.parent, exist_ok=True, parents=True)
        touch(path, clobber=True)
        df.to_parquet(path)

    def load_results(
        self,
        version: str,
        hierarchy: str,
        scenario: str,
        measure: str,
        location_id: int | None = None,
    ) -> pd.DataFrame:
        """Load final results for a specific scenario and measure.

        Parameters
        ----------
        version
            The version identifier
        hierarchy
            The location hierarchy
        scenario
            The climate scenario
        measure
            The climate measure
        location_id
            If provided, load only data for this location

        Returns
        -------
        pd.DataFrame
            The results
        """
        path = self.results_path(version, hierarchy, scenario, measure)
        if location_id is not None:
            filters = [("location_id", "==", location_id)]
            return pd.read_parquet(path, filters=filters)
        return pd.read_parquet(path)


class FloodingData:
    """Class for managing the flooding data used in the project."""

    def __init__(
        self,
        root: str | Path = cdc.FLOOD_ROOT,
        *,
        read_only: bool = False,
    ) -> None:
        self._root = Path(root)
        self._credentials_root = self._root / "credentials"
        self._read_only = read_only
        if not read_only:
            self._create_flooding_root()

    def _create_flooding_root(self) -> None:
        mkdir(self.root, exist_ok=True)
        mkdir(self.credentials_root, exist_ok=True)

        mkdir(self.results, exist_ok=True)
        mkdir(self.results_metadata, exist_ok=True)
        mkdir(self.daily_results, exist_ok=True)
        mkdir(self.raw_daily_results, exist_ok=True)
        mkdir(self.annual_results, exist_ok=True)
        mkdir(self.raw_annual_results, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def credentials_root(self) -> Path:
        return self._credentials_root

    ###########
    # Results #
    ###########

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def results_metadata(self) -> Path:
        return self.results / "metadata"

    def save_scenario_metadata(self, df: pd.DataFrame) -> None:
        if self._read_only:
            msg = "Cannot save scenario metadata to read-only data"
            raise ValueError(msg)
        path = self.results_metadata / "scenario_metadata.parquet"
        save_parquet(df, path)

    def load_scenario_metadata(self) -> pd.DataFrame:
        path = self.results_metadata / "scenario_metadata.parquet"
        return pd.read_parquet(path)

    def save_scenario_inclusion_metadata(self, df: pd.DataFrame) -> None:
        if self._read_only:
            msg = "Cannot save scenario inclusion metadata to read-only data"
            raise ValueError(msg)
        # Need to save to our scripts directory for doc building
        scripts_root = Path(__file__).parent.parent.parent / "scripts"
        for root_dir in [self.results_metadata, scripts_root]:
            path = root_dir / "scenario_inclusion_metadata.parquet"
            save_parquet(df, path)

    def load_scenario_inclusion_metadata(self) -> pd.DataFrame:
        path = self.results_metadata / "scenario_inclusion_metadata.parquet"
        return pd.read_parquet(path)

    @property
    def daily_results(self) -> Path:
        return self.results / "daily"

    @property
    def raw_daily_results(self) -> Path:
        return self.daily_results / "raw"

    def raw_daily_results_path(
        self,
        scenario: str,
        variable: str,
        year: int | str,
        gcm_member: str,
    ) -> Path:
        return self.raw_daily_results / scenario / variable / f"{year}_{gcm_member}.nc"

    def save_raw_daily_results(
        self,
        results_ds: xr.Dataset,
        scenario: str,
        variable: str,
        year: int | str,
        gcm_member: str,
        encoding_kwargs: dict[str, Any],
    ) -> None:
        if self._read_only:
            msg = "Cannot save raw daily results to read-only data"
            raise ValueError(msg)
        path = self.raw_daily_results_path(scenario, variable, year, gcm_member)
        mkdir(path.parent, exist_ok=True, parents=True)
        save_xarray(results_ds, path, encoding_kwargs)

    def daily_results_path(
        self,
        scenario: str,
        variable: str,
        year: int | str,
    ) -> Path:
        return self.daily_results / scenario / variable / f"{year}.nc"

    def save_daily_results(
        self,
        results_ds: xr.Dataset,
        scenario: str,
        variable: str,
        year: int | str,
        encoding_kwargs: dict[str, Any],
    ) -> None:
        if self._read_only:
            msg = "Cannot save daily results to read-only data"
            raise ValueError(msg)
        path = self.daily_results_path(scenario, variable, year)
        mkdir(path.parent, exist_ok=True, parents=True)
        save_xarray(results_ds, path, encoding_kwargs)

    def load_daily_results(
        self,
        scenario: str,
        variable: str,
        year: int | str,
    ) -> xr.Dataset:
        results_path = self.daily_results_path(scenario, variable, year)
        return xr.open_dataset(results_path)

    @property
    def annual_results(self) -> Path:
        return self.results / "annual"

    @property
    def raw_annual_results(self) -> Path:
        return self.annual_results / "raw"

    def raw_annual_results_path(
        self,
        scenario: str,
        variable: str,
        year: int | str,
        gcm_member: str,
    ) -> Path:
        return self.raw_annual_results / scenario / variable / f"{year}_{gcm_member}.nc"

    def save_raw_annual_results(
        self,
        results_ds: xr.Dataset,
        scenario: str,
        variable: str,
        year: int | str,
        gcm_member: str,
        encoding_kwargs: dict[str, Any],
    ) -> None:
        if self._read_only:
            msg = "Cannot save raw annual results to read-only data"
            raise ValueError(msg)
        path = self.raw_annual_results_path(scenario, variable, year, gcm_member)
        mkdir(path.parent, exist_ok=True, parents=True)
        save_xarray(results_ds, path, encoding_kwargs)

    @property
    def compiled_annual_results(self) -> Path:
        return self.raw_annual_results / "compiled"

    def compiled_annual_results_path(
        self,
        scenario: str,
        variable: str,
        gcm_member: str,
    ) -> Path:
        return self.compiled_annual_results / scenario / variable / f"{gcm_member}.nc"

    def save_compiled_annual_results(
        self,
        results_ds: xr.Dataset,
        scenario: str,
        variable: str,
        gcm_member: str,
    ) -> None:
        if self._read_only:
            msg = "Cannot save compiled annual results to read-only data"
            raise ValueError(msg)
        path = self.compiled_annual_results_path(scenario, variable, gcm_member)
        mkdir(path.parent, exist_ok=True, parents=True)
        touch(path, clobber=True)
        results_ds.to_netcdf(path)

    def annual_results_path(
        self,
        scenario: str,
        variable: str,
        draw: int | str,
    ) -> Path:
        return self.annual_results / scenario / variable / f"{draw:0>3}.nc"

    def link_annual_draw(
        self,
        draw: int | str,
        scenario: str,
        variable: str,
        gcm_member: str,
    ) -> None:
        if self._read_only:
            msg = "Cannot link annual draw to read-only data"
            raise ValueError(msg)
        source_path = self.compiled_annual_results_path(scenario, variable, gcm_member)
        dest_path = self.annual_results_path(scenario, variable, draw)
        mkdir(dest_path.parent, exist_ok=True, parents=True)
        if dest_path.exists():
            dest_path.unlink()
        dest_path.symlink_to(source_path)

    def draw_results_path(self, scenario: str, measure: str, draw: str) -> Path:
        """Get the path to annual results for a specific scenario, measure, and draw.

        Parameters
        ----------
        scenario
            The climate scenario (e.g. "ssp126")
        measure
            The climate measure (e.g. "mean_temperature")
        draw
            The draw of the climate data to load (e.g. "000")

        Returns
        -------
        Path
            The path to the results file
        """
        return self.annual_results / scenario / measure / f"{draw}.nc"

    def load_draw_results(self, scenario: str, measure: str, draw: str) -> xr.Dataset:
        """Load annual climate results for a specific scenario, measure, and draw.

        Parameters
        ----------
        scenario
            The climate scenario (e.g. "ssp126")
        measure
            The climate measure (e.g. "mean_temperature")
        draw
            The draw of the climate data to load (e.g. "000")

        Returns
        -------
        xr.Dataset
            The climate data in xarray format
        """
        path = self.annual_results_path(scenario, measure, draw)
        ds = xr.open_dataset(path, decode_coords="all")
        ds = ds.rio.write_crs("EPSG:4326")
        return ds


def save_xarray(
    ds: xr.Dataset,
    output_path: str | Path,
    encoding_kwargs: dict[str, Any],
) -> None:
    """Save an xarray dataset to a file with standard parameters.

    Parameters
    ----------
    ds
        The dataset to save.
    output_path
        The path to save the dataset to.
    encoding_kwargs
        The encoding parameters to use when saving the dataset.
    """
    touch(output_path, clobber=True)
    encoding = {
        "dtype": "int16",
        "_FillValue": -32767,
        "zlib": True,
        "complevel": 1,
    }
    encoding.update(encoding_kwargs)
    ds.to_netcdf(output_path, encoding={"value": encoding})


def save_raster(
    raster: rt.RasterArray,
    output_path: str | Path,
    num_cores: int = 1,
    **kwargs: Any,
) -> None:
    """Save a raster to a file with standard parameters.

    Parameters
    ----------
    raster
        The raster to save.
    output_path
        The path to save the raster to.
    num_cores
        The number of cores to use for compression.
    """
    save_params = {
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "ZSTD",
        "predictor": 2,  # horizontal differencing
        "num_threads": num_cores,
        "bigtiff": "yes",
        **kwargs,
    }
    touch(output_path, clobber=True)
    raster.to_file(output_path, **save_params)


def save_raster_to_cog(
    raster: rt.RasterArray,
    output_path: str | Path,
    num_cores: int = 1,
    resampling: str = "nearest",
) -> None:
    """Save a raster to a COG file.

    A COG file is a cloud-optimized GeoTIFF that is optimized for use in cloud storage
    systems. This function saves the raster to a COG file with the specified resampling
    method.

    Parameters
    ----------
    raster
        The raster to save.
    output_path
        The path to save the raster to.
    num_cores
        The number of cores to use for compression.
    resampling
        The resampling method to use when building the overviews.
    """
    cog_save_params = {
        "driver": "COG",
        "overview_resampling": resampling,
    }
    save_raster(raster, output_path, num_cores, **cog_save_params)
