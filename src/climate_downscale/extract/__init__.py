from climate_downscale.extract.ncei_climate_stations import (
    extract_ncei_climate_stations,
)

RUNNERS = {
    'ncei': extract_ncei_climate_stations,
}

TASK_RUNNERS = {
    'ncei': extract_ncei_climate_stations,
}
