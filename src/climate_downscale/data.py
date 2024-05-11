from pathlib import Path


DEFAULT_ROOT = "/mnt/share/erf/ERA5/"


class ClimateDownscaleData:

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._credentials_root = self._root / ".credentials"

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
    def era5(self) -> Path:
        return self.extracted_data / "era5"

    @property
    def ncei_climate_stations(self) -> Path:
        return self.extracted_data / "ncei_climate_stations"
