from climate_downscale.extract.era5 import (
    extract_era5,
    extract_era5_task,
)
from climate_downscale.extract.ncei_climate_stations import (
    extract_ncei_climate_stations,
)
from climate_downscale.extract.rub_local_climate_zones import (
    extract_rub_local_climate_zones,
)
from climate_downscale.extract.srtm_elevation import (
    extract_srtm_elevation,
    extract_srtm_elevation_task,
)

RUNNERS = {
    "ncei": extract_ncei_climate_stations,
    "era5": extract_era5,
    "lcz": extract_rub_local_climate_zones,
    "elevation": extract_srtm_elevation,
}

TASK_RUNNERS = {
    "ncei": extract_ncei_climate_stations,
    "era5": extract_era5_task,
    "lcz": extract_rub_local_climate_zones,
    "elevation": extract_srtm_elevation_task,
}
