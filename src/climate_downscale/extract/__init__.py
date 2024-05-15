from climate_downscale.extract.elevation import (
    extract_elevation,
    extract_elevation_task,
)
from climate_downscale.extract.era5 import (
    extract_era5,
    extract_era5_task,
)
from climate_downscale.extract.ncei_climate_stations import (
    extract_ncei_climate_stations,
    extract_ncei_climate_stations_task,
)
from climate_downscale.extract.rub_local_climate_zones import (
    extract_rub_local_climate_zones,
)

RUNNERS = {
    "ncei": extract_ncei_climate_stations,
    "era5": extract_era5,
    "lcz": extract_rub_local_climate_zones,
    "elevation": extract_elevation,
}

TASK_RUNNERS = {
    "ncei": extract_ncei_climate_stations_task,
    "era5": extract_era5_task,
    "lcz": extract_rub_local_climate_zones,
    "elevation": extract_elevation_task,
}
