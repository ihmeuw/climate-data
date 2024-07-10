from climate_downscale.generate.derived_daily import (
    generate_derived_daily,
    generate_derived_daily_task,
)
from climate_downscale.generate.historical_daily import (
    generate_historical_daily,
    generate_historical_daily_task,
)
from climate_downscale.generate.historical_reference import (
    generate_historical_reference,
    generate_historical_reference_task,
)
from climate_downscale.generate.scenario_annual import (
    generate_scenario_annual,
    generate_scenario_annual_task,
)
from climate_downscale.generate.scenario_daily import (
    generate_scenario_daily,
    generate_scenario_daily_task,
)
from climate_downscale.generate.scenario_inclusion import (
    generate_scenario_inclusion,
)

RUNNERS = {
    "historical_daily": generate_historical_daily,
    "historical_reference": generate_historical_reference,
    "scenario_inclusion": generate_scenario_inclusion,
    "scenario_daily": generate_scenario_daily,
    "derived_daily": generate_derived_daily,
    "scenario_annual": generate_scenario_annual,
}

TASK_RUNNERS = {
    "historical_daily": generate_historical_daily_task,
    "historical_reference": generate_historical_reference_task,
    "scenario_inclusion": generate_scenario_inclusion,
    "scenario_daily": generate_scenario_daily_task,
    "derived_daily": generate_derived_daily_task,
    "scenario_annual": generate_scenario_annual_task,
}
