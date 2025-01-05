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
"""

from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd
import rasterra as rt
import xarray as xr
from rra_tools.shell_tools import mkdir, touch

from climate_data import constants as cdc


class ClimateData:
    """Class for managing the climate data used in the project."""

    def __init__(self, root: str | Path = cdc.MODEL_ROOT) -> None:
        self._root = Path(root)
        self._credentials_root = self._root / "credentials"
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
        path = self.predictors / f"{name}_{lat_start}_{lon_start}.tif"
        save_raster(predictor, path)

    def load_predictor(self, name: str) -> rt.RasterArray:
        paths = list(self.predictors.glob(f"{name}_*.tif"))
        return rt.load_mf_raster(paths)

    @property
    def training_data(self) -> Path:
        return self.downscale_model / "training_data"

    def save_training_data(self, df: pd.DataFrame, year: int | str) -> None:
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
        path = self.results_metadata / "scenario_metadata.parquet"
        save_parquet(df, path)

    def load_scenario_metadata(self) -> pd.DataFrame:
        path = self.results_metadata / "scenario_metadata.parquet"
        return pd.read_parquet(path)

    def save_scenario_inclusion_metadata(self, df: pd.DataFrame) -> None:
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
        source_path = self.compiled_annual_results_path(scenario, variable, gcm_member)
        dest_path = self.annual_results_path(scenario, variable, draw)
        mkdir(dest_path.parent, exist_ok=True, parents=True)
        if dest_path.exists():
            dest_path.unlink()
        dest_path.symlink_to(source_path)


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
