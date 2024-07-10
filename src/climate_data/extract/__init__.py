from climate_data.extract.cmip6 import (
    extract_cmip6,
    extract_cmip6_task,
)
from climate_data.extract.elevation import (
    extract_elevation,
    extract_elevation_task,
)
from climate_data.extract.era5 import (
    download_era5_task,
    extract_era5,
    unzip_and_compress_era5_task,
)
from climate_data.extract.ncei_climate_stations import (
    extract_ncei_climate_stations,
    extract_ncei_climate_stations_task,
)
from climate_data.extract.rub_local_climate_zones import (
    extract_rub_local_climate_zones,
)

RUNNERS = {
    "ncei": extract_ncei_climate_stations,
    "era5": extract_era5,
    "cmip6": extract_cmip6,
    "lcz": extract_rub_local_climate_zones,
    "elevation": extract_elevation,
}

TASK_RUNNERS = {
    "ncei": extract_ncei_climate_stations_task,
    "cmip6": extract_cmip6_task,
    "era5_download": download_era5_task,
    "era5_compress": unzip_and_compress_era5_task,
    "lcz": extract_rub_local_climate_zones,
    "elevation": extract_elevation_task,
}
