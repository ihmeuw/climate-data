from climate_downscale.model.prepare_predictors import (
    prepare_predictors,
    prepare_predictors_task,
)

RUNNERS = {
    "prepare_predictors": prepare_predictors,
}

TASK_RUNNERS = {
    "prepare_predictors": prepare_predictors_task,
}
