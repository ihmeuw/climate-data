from pathlib import Path

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
    def rub_local_climate_zones(self) -> Path:
        return self.extracted_data / "rub_local_climate_zones"
