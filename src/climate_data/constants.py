from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import xarray as xr

##############
# File roots #
##############

MODEL_ROOT = Path("/mnt/share/erf/climate_downscale/")

######################
# Pipeline variables #
######################

# Time

HISTORY_YEARS = [str(y) for y in range(1950, 2024)]
REFERENCE_YEARS = HISTORY_YEARS[-5:]
REFERENCE_PERIOD = slice(
    f"{REFERENCE_YEARS[0]}-01-01",
    f"{REFERENCE_YEARS[-1]}-12-31",
)
FORECAST_YEARS = [str(y) for y in range(2024, 2101)]

MONTHS = [f"{i:02d}" for i in range(1, 13)]

# Space

TARGET_LON = xr.DataArray(
    np.round(np.arange(-180.0, 180.0, 0.1, dtype="float32"), 1), dims="longitude"
)
TARGET_LAT = xr.DataArray(
    np.round(np.arange(-90.0, 90.1, 0.1, dtype="float32"), 1), dims="latitude"
)

# Extraction Constants


class _ERA5Datasets(NamedTuple):
    # Use named tuple so that we can access the dataset names as attributes
    reanalysis_era5_land: str = "reanalysis-era5-land"
    reanalysis_era5_single_levels: str = "reanalysis-era5-single-levels"


ERA5_DATASETS = _ERA5Datasets()


class _ERA5Variables(NamedTuple):
    u_component_of_wind: str = "10m_u_component_of_wind"
    v_component_of_wind: str = "10m_v_component_of_wind"
    dewpoint_temperature: str = "2m_dewpoint_temperature"
    temperature: str = "2m_temperature"
    surface_pressure: str = "surface_pressure"
    total_precipitation: str = "total_precipitation"
    sea_surface_temperature: str = "sea_surface_temperature"


ERA5_VARIABLES = _ERA5Variables()

CMIP6_SOURCES = [
    "ACCESS-CM2",
    "AWI-CM-1-1-MR",
    "BCC-CSM2-MR",
    "CAMS-CSM1-0",
    "CESM2-WACCM",
    "CMCC-CM2-SR5",
    "CMCC-ESM2",
    "CNRM-CM6-1",
    "CNRM-CM6-1-HR",
    "CNRM-ESM2-1",
    "FGOALS-g3",
    "GFDL-ESM4",
    "GISS-E2-1-G",
    "IITM-ESM",
    "INM-CM4-8",
    "INM-CM5-0",
    "MIROC-ES2L",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NorESM2-MM",
]


class _CMIP6Experiments(NamedTuple):
    ssp126: str = "ssp126"
    ssp245: str = "ssp245"
    ssp585: str = "ssp585"


CMIP6_EXPERIMENTS = _CMIP6Experiments()


class CMIP6Variable(NamedTuple):
    name: str
    description: str
    encoding_offset: float
    encoding_scale: float
    table_id: Literal["day", "Oday"]


class _CMIP6Variables(NamedTuple):
    uas: CMIP6Variable = CMIP6Variable(
        name="uas",
        description="Eastward Near-Surface Wind",
        encoding_offset=0.0,
        encoding_scale=0.01,
        table_id="day",
    )
    vas: CMIP6Variable = CMIP6Variable(
        name="vas",
        description="Northward Near-Surface Wind",
        encoding_offset=0.0,
        encoding_scale=0.01,
        table_id="day",
    )
    hurs: CMIP6Variable = CMIP6Variable(
        name="hurs",
        description="Near-Surface Relative Humidity",
        encoding_offset=0.0,
        encoding_scale=0.01,
        table_id="day",
    )
    tas: CMIP6Variable = CMIP6Variable(
        name="tas",
        description="Near-Surface Air Temperature",
        encoding_offset=273.15,
        encoding_scale=0.01,
        table_id="day",
    )
    tasmin: CMIP6Variable = CMIP6Variable(
        name="tasmin",
        description="Near-Surface Minimum Air Temperature",
        encoding_offset=273.15,
        encoding_scale=0.01,
        table_id="day",
    )
    tasmax: CMIP6Variable = CMIP6Variable(
        name="tasmax",
        description="Near-Surface Maximum Air Temperature",
        encoding_offset=273.15,
        encoding_scale=0.01,
        table_id="day",
    )
    tos: CMIP6Variable = CMIP6Variable(
        name="tos",
        description="Sea Surface Temperature",
        encoding_offset=273.15,
        encoding_scale=0.01,
        table_id="Oday",
    )
    pr: CMIP6Variable = CMIP6Variable(
        name="pr",
        description="Precipitation",
        encoding_offset=0.0,
        encoding_scale=1e-9,
        table_id="day",
    )

    def names(self) -> list[str]:
        return [v.name for v in self]

    def get(self, name: str) -> CMIP6Variable:
        return getattr(self, name)  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, CMIP6Variable]:
        return {v.name: v for v in self}


CMIP6_VARIABLES = _CMIP6Variables()


# Processing Constants

TARGET_LON = xr.DataArray(
    np.round(np.arange(-180.0, 180.0, 0.1, dtype="float32"), 1), dims="longitude"
)
TARGET_LAT = xr.DataArray(
    np.round(np.arange(-90.0, 90.1, 0.1, dtype="float32"), 1), dims="latitude"
)

DRAWS = [str(d) for d in range(100)]


class _Scenarios(NamedTuple):
    historical: str = "historical"
    ssp126: str = "ssp126"
    ssp245: str = "ssp245"
    ssp585: str = "ssp585"


SCENARIOS = _Scenarios()
