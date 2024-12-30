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
            touch(meta_path)
            df.to_parquet(meta_path)

        df = pd.read_parquet(meta_path)
        if return_full_criteria:
            return df
        return df[["source_id", "included"]]

    def load_cmip6_metadata(self) -> pd.DataFrame:
        meta_path = self.extracted_cmip6 / "cmip6-metadata.parquet"
        if not meta_path.exists():
            external_path = "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
            meta = pd.read_csv(external_path)
            touch(meta_path)
            meta.to_parquet(meta_path)
        return pd.read_parquet(meta_path)

    def extracted_cmip6_path(
        self, variable: str, experiment: str, source: str, member: str
    ) -> Path:
        return self.extracted_cmip6 / f"{variable}_{experiment}_{source}_{member}.nc"

    @property
    def ncei_climate_stations(self) -> Path:
        return self.extracted_data / "ncei_climate_stations"

    def save_ncei_climate_stations(self, df: pd.DataFrame, year: int | str) -> None:
        path = self.ncei_climate_stations / f"{year}.parquet"
        touch(path, exist_ok=True)
        df.to_parquet(path)

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
        save_raster(predictor, self.predictors / f"{name}_{lat_start}_{lon_start}.tif")

    def load_predictor(self, name: str) -> rt.RasterArray:
        paths = list(self.predictors.glob(f"{name}_*.tif"))
        return rt.load_mf_raster(paths)

    @property
    def training_data(self) -> Path:
        return self.downscale_model / "training_data"

    def save_training_data(self, df: pd.DataFrame, year: int | str) -> None:
        path = self.training_data / f"{year}.parquet"
        if path.exists():
            path.unlink()
        touch(path)
        df.to_parquet(path)

    def load_training_data(self, year: int | str) -> pd.DataFrame:
        return pd.read_parquet(self.training_data / f"{year}.parquet")

    ###########
    # Results #
    ###########

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def daily_results(self) -> Path:
        return self.results / "daily"

    @property
    def results_metadata(self) -> Path:
        return self.results / "metadata"

    def save_scenario_metadata(self, df: pd.DataFrame) -> None:
        path = self.results_metadata / "scenario_metadata.parquet"
        if path.exists():
            path.unlink()
        touch(path)
        df.to_parquet(path)

    def load_scenario_metadata(self) -> pd.DataFrame:
        path = self.results_metadata / "scenario_metadata.parquet"
        return pd.read_parquet(path)

    def save_scenario_inclusion_metadata(self, df: pd.DataFrame) -> None:
        # Need to save to our scripts directory for doc building
        scripts_root = Path(__file__).parent.parent.parent / "scripts"
        for root_dir in [self.results_metadata, scripts_root]:
            path = root_dir / "scenario_inclusion_metadata.parquet"
            if path.exists():
                path.unlink()
            touch(path)
            df.to_parquet(path)

    def load_scenario_inclusion_metadata(self) -> pd.DataFrame:
        path = self.results_metadata / "scenario_inclusion_metadata.parquet"
        return pd.read_parquet(path)

    def daily_results_path(
        self,
        scenario: str,
        variable: str,
        year: int | str,
        draw: int | str | None = None,
    ) -> Path:
        file_name = f"{year}.nc" if draw is None else f"{year}_{draw}.nc"
        return self.daily_results / scenario / variable / file_name

    def save_daily_results(
        self,
        results_ds: xr.Dataset,
        scenario: str,
        variable: str,
        year: int | str,
        draw: int | str | None,
        encoding_kwargs: dict[str, Any],
    ) -> None:
        path = self.daily_results_path(scenario, variable, year, draw)
        if path.exists():
            path.unlink()
        mkdir(path.parent, exist_ok=True, parents=True)
        touch(path, exist_ok=True)

        encoding = {
            "dtype": "int16",
            "_FillValue": -32767,
            "zlib": True,
            "complevel": 1,
        }
        encoding.update(encoding_kwargs)

        results_ds.to_netcdf(path, encoding={"value": encoding})

    def load_daily_results(
        self,
        scenario: str,
        variable: str,
        year: int | str,
        draw: int | str | None = None,
    ) -> xr.Dataset:
        results_path = self.daily_results_path(scenario, variable, year, draw)
        return xr.open_dataset(results_path)

    @property
    def annual_results(self) -> Path:
        return self.results / "annual"

    def annual_results_path(
        self, scenario: str, variable: str, year: int | str, draw: int | str | None
    ) -> Path:
        file_name = f"{year}.nc" if draw is None else f"{year}_{int(draw):0>3}.nc"
        return self.annual_results / scenario / variable / file_name

    def save_annual_results(
        self,
        results_ds: xr.Dataset,
        scenario: str,
        variable: str,
        year: int | str,
        draw: int | str | None,
        encoding_kwargs: dict[str, Any],
    ) -> None:
        path = self.annual_results_path(scenario, variable, year, draw)
        if path.exists():
            path.unlink()
        mkdir(path.parent, exist_ok=True, parents=True)
        touch(path)

        encoding = {
            "dtype": "int16",
            "_FillValue": -32767,
            "zlib": True,
            "complevel": 1,
        }
        encoding.update(encoding_kwargs)
        results_ds.to_netcdf(path, encoding={"value": encoding})


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
    touch(output_path, exist_ok=True)
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
    touch(output_path, exist_ok=True)
    save_raster(raster, output_path, num_cores, **cog_save_params)
