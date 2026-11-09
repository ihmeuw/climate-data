from climate_data.special.compile_person_days import (
    compile_person_days,
    compile_person_days_task,
)
from climate_data.special.download_era5_uncertainty import (
    download_era5_uncertainty,
    download_era5_uncertainty_task,
)
from climate_data.special.history_person_days import (
    build_history_person_days,
    build_history_person_days_task,
)
from climate_data.special.temperature_person_days import (
    temperature_person_days,
    temperature_person_days_task,
)
from climate_data.special.temperature_zone import (
    generate_temperature_zone,
    generate_temperature_zone_task,
)
from climate_data.special.weight_table import (
    build_weight_table_task,
    build_weight_tables,
)

RUNNERS = {
    "temperature_zone": generate_temperature_zone,
    "temperature_person_days": temperature_person_days,
    "compile_person_days": compile_person_days,
    "download_era5_uncertainty": download_era5_uncertainty,
    "weight_table": build_weight_tables,
    "history_person_days": build_history_person_days,
}

TASK_RUNNERS = {
    "temperature_zone": generate_temperature_zone_task,
    "temperature_person_days": temperature_person_days_task,
    "compile_person_days": compile_person_days_task,
    "download_era5_uncertainty": download_era5_uncertainty_task,
    "weight_table": build_weight_table_task,
    "history_person_days": build_history_person_days_task,
}
