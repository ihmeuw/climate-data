import click

from climate_data import aggregate, diagnostics, downscale, extract, generate, special


@click.group()
def cdrun() -> None:
    """Entry point for running climate downscale workflows."""


@click.group()
def cdtask() -> None:
    """Entry point for running climate downscale tasks."""


for module in [extract, downscale, generate, aggregate, special, diagnostics]:
    runners = getattr(module, "RUNNERS", {})
    task_runners = getattr(module, "TASK_RUNNERS", {})

    if not runners or not task_runners:
        continue

    command_name = module.__name__.split(".")[-1]

    @click.group(name=command_name)
    def workflow_runner() -> None:
        pass

    for name, runner in runners.items():
        workflow_runner.add_command(runner, name)

    cdrun.add_command(workflow_runner)

    @click.group(name=command_name)
    def task_runner() -> None:
        pass

    for name, runner in task_runners.items():
        task_runner.add_command(runner, name)

    cdtask.add_command(task_runner)
