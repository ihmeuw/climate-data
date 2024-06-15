from climate_downscale.generate.historical_daily import (
    generate_historical_daily,
    generate_historical_daily_task,
)

RUNNERS = {
    "historical_daily": generate_historical_daily,
}

TASK_RUNNERS = {
    "historical_daily": generate_historical_daily_task,
}
