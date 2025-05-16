from climate_data.diagnostics.grid_plots import (
    grid_plots,
    grid_plots_task,
)

RUNNERS = {
    "grid_plots": grid_plots,
}

TASK_RUNNERS = {
    "grid_plots": grid_plots_task,
}
