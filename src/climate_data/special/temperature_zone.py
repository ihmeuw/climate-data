import itertools
from pathlib import Path

import click
from rra_tools import jobmon

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData


def generate_temperature_zone_main(
    gcm_member: str,
    cmip6_experiment: str,
    output_dir: str | Path,
) -> None:
    """Generate the temperature zone for a given scenario and gcm member.

    Parameters
    ----------
    gcm_member
        The gcm member to generate the temperature zone for.
    cmip6_experiment
        The cmip6 experiment to generate the temperature zone for.
    output_dir
        The directory to save the temperature zone to.
    """
    print(f"Generating temperature zone for {cmip6_experiment} {gcm_member}")
    cdata = ClimateData(output_dir)
    ds = cdata.load_compiled_annual_results(
        cmip6_experiment, "mean_temperature", gcm_member
    )
    temperature_zone = ds.rolling(year=10).mean().sel(year=slice(1990, 2100))
    print(f"Saving temperature zone for {cmip6_experiment} {gcm_member}")
    cdata.save_compiled_annual_results(
        temperature_zone,
        scenario=cmip6_experiment,
        variable="temperature_zone",
        gcm_member=gcm_member,
        encoding_kwargs={"scale_factor": 0.01, "add_offset": 0.0},
    )


@click.command()
@clio.with_gcm_member()
@clio.with_cmip6_experiment()
@clio.with_output_directory(cdc.MODEL_ROOT)
def generate_temperature_zone_task(
    gcm_member: str,
    cmip6_experiment: str,
    output_dir: str,
) -> None:
    generate_temperature_zone_main(gcm_member, cmip6_experiment, output_dir)


@click.command()
@clio.with_cmip6_experiment(allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_queue()
@clio.with_overwrite()
def generate_temperature_zone(
    cmip6_experiment: list[str],
    output_dir: str,
    queue: str,
    overwrite: bool,
) -> None:
    cdata = ClimateData(output_dir)
    gcm_members = cdata.list_gcm_members("ssp126", "mean_temperature")

    complete = []
    to_run = []
    for e, g in itertools.product(cmip6_experiment, gcm_members):
        path = cdata.compiled_annual_results_path(e, "temperature_zone", g)
        if not path.exists() or overwrite:
            to_run.append((e, g))
        else:
            complete.append((e, g))

    print(f"{len(complete)} tasks already done. Launching {len(to_run)} tasks")

    jobmon.run_parallel(
        runner="cdtask special",
        task_name="temperature_zone",
        flat_node_args=(
            ("cmip6-experiment", "gcm-member"),
            to_run,
        ),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 4,
            "memory": "50G",
            "runtime": "30m",
            "project": "proj_rapidresponse",
        },
        max_attempts=2,
    )
