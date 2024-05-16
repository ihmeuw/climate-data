from pathlib import Path
from typing import Any

import pandas as pd
import rasterra as rt
import xarray as xr
from rra_tools.shell_tools import touch

DEFAULT_ROOT = "/mnt/share/erf/climate_downscale/"


class ClimateDownscaleData:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._credentials_root = self._root / "credentials"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def credentials_root(self) -> Path:
        return self._credentials_root

    @property
    def extracted_data(self) -> Path:
        return self.root / "extracted_data"

    @property
    def era5_temperature_daily_mean(self) -> Path:
        return self.extracted_data / "era5_temperature_daily_mean"

    def load_era5_temperature_daily_mean(self, year: int | str) -> xr.Dataset:
        return xr.load_dataset(
            self.era5_temperature_daily_mean / f"{year}_era5_temp_daily.nc"
        )

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
    def srtm_elevation_gl1(self) -> Path:
        return self.extracted_data / "srtm_elevation_gl1"

    @property
    def open_topography_elevation(self) -> Path:
        return self.extracted_data / "open_topography_elevation"

    @property
    def rub_local_climate_zones(self) -> Path:
        return self.extracted_data / "rub_local_climate_zones"

    @property
    def model(self) -> Path:
        return self.root / "model"

    @property
    def predictors(self) -> Path:
        return self.model / "predictors"

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
        return self.model / "training_data"

    def save_training_data(self, df: pd.DataFrame, year: int | str) -> None:
        path = self.training_data / f"{year}.parquet"
        touch(path, exist_ok=True)
        df.to_parquet(path)

    def load_training_data(self, year: int | str) -> pd.DataFrame:
        return pd.read_parquet(self.training_data / f"{year}.parquet")


def save_raster(
    raster: rt.RasterArray,
    output_path: str | Path,
    num_cores: int = 1,
    **kwargs: Any,
) -> None:
    """Save a raster to a file with standard parameters."""
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
    """Save a raster to a COG file."""
    cog_save_params = {
        "driver": "COG",
        "overview_resampling": resampling,
    }
    touch(output_path, exist_ok=True)
    save_raster(raster, output_path, num_cores, **cog_save_params)
