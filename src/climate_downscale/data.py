from pathlib import Path
from typing import Any

import rasterra as rt

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

    @property
    def ncei_climate_stations(self) -> Path:
        return self.extracted_data / "ncei_climate_stations"

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

    def save_predictor(self, predictor: rt.RasterArray, name: str) -> None:
        save_raster(predictor, self.predictors / f"{name}.tif")

    def load_predictor(self, name: str) -> rt.RasterArray:
        return rt.load_raster(self.predictors / f"{name}.tif")


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
    save_raster(raster, output_path, num_cores, **cog_save_params)
