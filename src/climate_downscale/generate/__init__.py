from climate_downscale.generate.era5_daily import (
    generate_era5_daily,
    generate_era5_daily_task,
)

RUNNERS = {
    "era5_daily": generate_era5_daily,
}

TASK_RUNNERS = {
    "era5_daily": generate_era5_daily_task,
}
