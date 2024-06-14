from climate_downscale.downscale.prepare_predictors import (
    prepare_predictors,
    prepare_predictors_task,
)
from climate_downscale.downscale.prepare_training_data import (
    prepare_training_data,
    prepare_training_data_task,
)

RUNNERS = {
    "prepare_predictors": prepare_predictors,
    "prepare_training_data": prepare_training_data,
}

TASK_RUNNERS = {
    "prepare_predictors": prepare_predictors_task,
    "prepare_training_data": prepare_training_data_task,
}
