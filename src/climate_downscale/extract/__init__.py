from climate_downscale.extract.ncei_climate_stations import (
    extract_ncei_climate_stations,
)
from climate_downscale.extract.era5 import (
    extract_era5,
    extract_era5_task,
)

RUNNERS = {
    'ncei': extract_ncei_climate_stations,
    "era5": extract_era5,
}

TASK_RUNNERS = {
    'ncei': extract_ncei_climate_stations,
    "era5": extract_era5_task,
}
