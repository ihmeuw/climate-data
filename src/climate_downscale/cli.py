import click

from climate_downscale import extract


@click.group()
def cdrun() -> None:
    """Entry point for running climate downscale workflows."""
    pass


@click.group()
def cdtask() -> None:
    """Entry point for running climate downscale tasks."""
    pass


for module in [extract]:
    runners = getattr(module, 'RUNNERS', {})
    task_runners = getattr(module, 'TASK_RUNNERS', {})

    if not runners or not task_runners:
        continue

    command_name = module.__name__.split('.')[-1]

    @click.group(name=command_name)
    def workflow_runner():
        pass

    for name, runner in runners.items():
        workflow_runner.add_command(runner, name)

    cdrun.add_command(workflow_runner)

    @click.group(name=command_name)
    def task_runner():
        pass

    for name, runner in task_runners.items():
        task_runner.add_command(runner, name)

    cdtask.add_command(task_runner)
