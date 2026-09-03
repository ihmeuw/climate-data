from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import xarray as xr

##############
# File roots #
##############

# RRA team root directory
RRA_ROOT = Path("/mnt/team/rapidresponse/pub/")
# Contains gridded population estimates and projections
POPULATION_MODEL_ROOT = RRA_ROOT / "population-model"

# Run modes select the storage-root profile. ``forecast`` (the default) uses the
# production roots; ``historical`` routes the GBD historical exposure product to
# the geospatial working area. Selected per run via the ``--run-mode`` CLI option.
RUN_MODES = ("forecast", "historical")
RunMode = Literal["forecast", "historical"]

_MODEL_ROOTS: dict[str, Path] = {
    "forecast": Path("/mnt/share/erf/climate_downscale/"),
    "historical": Path("/mnt/share/geospatial/climate/"),
}


def model_root(run_mode: str) -> Path:
    """Downscaling working directory for the given run mode."""
    try:
        return _MODEL_ROOTS[run_mode]
    except KeyError:
        msg = f"Unknown run_mode {run_mode!r}; expected one of {list(RUN_MODES)}"
        raise ValueError(msg) from None


def aggregate_root(run_mode: str) -> Path:
    """Aggregation working directory for the given run mode."""
    if run_mode == "forecast":
        return RRA_ROOT / "climate-aggregates"
    if run_mode == "historical":
        return model_root("historical") / "aggregates"
    msg = f"Unknown run_mode {run_mode!r}; expected one of {list(RUN_MODES)}"
    raise ValueError(msg)


# Production defaults (used as the module-level roots and the CLI option defaults).
MODEL_ROOT = model_root("forecast")
AGGREGATE_ROOT = aggregate_root("forecast")


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
ALL_YEARS = HISTORY_YEARS + FORECAST_YEARS
# Accumulation windows are stamped by their end, so generating a month reads the first
# sample of the next one and the last history year reaches into the year after it. The
# single-job extract tasks span this; the runner deliberately does not, so `-y ALL` keeps
# meaning "the history range" and not "one more year to download".
EXTRACT_YEARS = [*HISTORY_YEARS, str(int(HISTORY_YEARS[-1]) + 1)]
# Start of the temperature-exposure / person-days analysis window (the historical
# person-days product spans this through the last year present on disk).
EXPOSURE_START_YEAR = 1990

MONTHS = [f"{i:02d}" for i in range(1, 13)]

# Space

ERA5_LAND_LONGITUDE = xr.DataArray(
    np.round(np.arange(-180.0, 180.0, 0.1, dtype=np.float64), 1), dims="longitude"
)
ERA5_LAND_LATITUDE = xr.DataArray(
    np.round(np.arange(-90.0, 90.1, 0.1, dtype=np.float64), 1), dims="latitude"
)


TARGET_LONGITUDE = xr.DataArray(
    np.round(np.arange(-179.95, 180.0, 0.1, dtype=np.float64), 2), dims="longitude"
)
TARGET_LATITUDE = xr.DataArray(
    np.round(np.arange(-89.95, 90.0, 0.1, dtype=np.float64), 2), dims="latitude"
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
    # Signed unless the variable cannot be negative. `int16` spends half its range on
    # negatives, which a flux like precipitation never uses -- `uint16` doubles the
    # representable ceiling at identical resolution. Read this by attribute, never by
    # positional unpacking: two call sites used `*_, offset, scale, table_id` and would
    # have silently misbound when this field was added.
    encoding_dtype: Literal["int16", "uint16"] = "int16"


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
        # `pr` is a flux in kg m-2 s-1, so the scale must span daily rainfall expressed
        # per second. At 1e-9 the int16 ceiling was 2.83 mm/day and anything wetter
        # wrapped modulo 65536, decoding as garbage including negative precipitation.
        #
        # Resolution is the thing to protect: at 1e-6 the quantum is 0.0864 mm/day, just
        # under the 0.1 mm/day the daily product downstream stores, so the input is not
        # the limiting factor. Coarsening the scale to buy headroom would invert that --
        # 2e-6 gives 0.173 mm/day, worse than the output it feeds. So headroom comes from
        # the dtype instead: precipitation is never negative, and `uint16` puts the whole
        # 65535-code range above zero for a 5662 mm/day ceiling at the same quantum.
        # `_FillValue` sits at the top of that range, so zero rainfall encodes as 0 and
        # can never collide with it.
        #
        # Measured: one member peaked at 1012 mm/day and ACCESS-CM2 ssp585 at 3108, which
        # overflowed the 2831 mm/day signed ceiling by 9.8% and is 55% of the unsigned
        # one. `1e-7` was rejected outright -- a 283 mm/day ceiling fails on both.
        #
        # Settle any change before a re-extract, not after: every `pr_*.nc` carries
        # whichever encoding is here when it is written.
        encoding_scale=1e-6,
        table_id="day",
        encoding_dtype="uint16",
    )

    def names(self) -> list[str]:
        return [v.name for v in self]

    def get(self, name: str) -> CMIP6Variable:
        return getattr(self, name)  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, CMIP6Variable]:
        return {v.name: v for v in self}


CMIP6_VARIABLES = _CMIP6Variables()


# Processing Constants

# Draws for uncertainty quantification
# Each draw represents a different variant of a specific GCM in the CMIP6 ensemble
DRAWS = [f"{d:>03}" for d in range(100)]  # 100 draws


class _Scenarios(NamedTuple):
    historical: str = "historical"
    ssp126: str = "ssp126"
    ssp245: str = "ssp245"
    ssp585: str = "ssp585"


SCENARIOS = _Scenarios()


# Resolution settings for raster data
RESOLUTION = "100"  # 100m resolution
TARGET_RESOLUTION = f"world_cylindrical_{RESOLUTION}"

AGGREGATION_SCENARIOS = [
    SCENARIOS.ssp126,
    SCENARIOS.ssp245,
    SCENARIOS.ssp585,
]

# Climate measures to calculate
AGGREGATION_MEASURES = [
    # Temperature metrics
    "mean_temperature",  # Average daily temperature
    "mean_high_temperature",  # Average daily maximum temperature
    "mean_low_temperature",  # Average daily minimum temperature
    "days_over_30C",  # Number of days with temperature > 30°C
    # Disease suitability metrics
    "malaria_suitability",  # Climate suitability for malaria transmission
    "dengue_suitability",  # Climate suitability for dengue transmission
    # Other climate metrics
    "wind_speed",  # Average wind speed
    "relative_humidity",  # Average relative humidity
    "total_precipitation",  # Total precipitation
    "precipitation_days",  # Number of days with precipitation
]

# This is a mapping between full aggregation hierarchies and subset hierarchies.
# The most-detailed units in the full aggregation hierarchies are the administrative units
# that we aggregate pixel-level climate and population data to in the `aggregate` step
# of the pipeline. In the `compile` step, we then aggregate the most-detailed units up to
# to all locations in the full aggregation hierarchies. The subset hierarchies are then
# used to provide different views of the full aggregation hierarchies as a full hierarchy
# is a superset of any of the subset hierarchies.
#
# Key concepts:
# - Full aggregation hierarchies: These are hierarchies of locations that contain all
#   locations in the subset hierarchies.
# - Subset hierarchies: These are hierarchies of locations that are a subset of the full
#   aggregation hierarchies.
HIERARCHY_MAP = {
    "gbd_2021": [
        "gbd_2021",
        "fhs_2021",
    ],  # GBD pixel hierarchy maps to GBD and FHS locations
    "gbd_2023": [
        "gbd_2023",
        "fhs_2023",
    ],  # GBD pixel hierarchy maps to GBD and FHS locations
    "gbd_2025": [
        "gbd_2025",
    ],  # No fhs_2025 raking files yet; GBD-only view for now.
    "lsae_1209": ["lsae_1209"],  # LSAE pixel hierarchy maps to LSAE locations
    "lsae_1285": ["lsae_1285"],  # LSAE pixel hierarchy maps to LSAE locations
}
# GBD pixel hierarchies -- the version axis for the special stage.
GBD_HIERARCHIES = ["gbd_2021", "gbd_2023", "gbd_2025"]
