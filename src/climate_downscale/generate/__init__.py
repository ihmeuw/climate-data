from climate_downscale.generate.historical_daily import (
    generate_historical_daily,
    generate_historical_daily_task,
)
from climate_downscale.generate.historical_reference import (
    generate_historical_reference,
    generate_historical_reference_task,
)

RUNNERS = {
    "historical_daily": generate_historical_daily,
    "historical_reference": generate_historical_reference,
}

TASK_RUNNERS = {
    "historical_daily": generate_historical_daily_task,
    "historical_reference": generate_historical_reference_task,
}
