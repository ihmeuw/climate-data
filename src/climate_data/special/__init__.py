from climate_data.special.compile_person_days import (
    compile_person_days,
    compile_person_days_task,
)
from climate_data.special.temperature_person_days import (
    temperature_person_days,
    temperature_person_days_task,
)
from climate_data.special.temperature_zone import (
    generate_temperature_zone,
    generate_temperature_zone_task,
)

RUNNERS = {
    "temperature_zone": generate_temperature_zone,
    "temperature_person_days": temperature_person_days,
    "compile_person_days": compile_person_days,
}

TASK_RUNNERS = {
    "temperature_zone": generate_temperature_zone_task,
    "temperature_person_days": temperature_person_days_task,
    "compile_person_days": compile_person_days_task,
}
