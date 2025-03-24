from climate_data.aggregate.hierarchy import (
    hierarchy,
    hierarchy_task,
)
from climate_data.aggregate.pixel import (
    pixel,
    pixel_task,
)

RUNNERS = {
    "hierarchy": hierarchy,
    "pixel": pixel,
}

TASK_RUNNERS = {
    "hierarchy": hierarchy_task,
    "pixel": pixel_task,
}
