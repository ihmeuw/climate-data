from pathlib import Path

import click
import xarray as xr

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData
from climate_data.jobmon_utils import run_parallel_maybe_dry_run


def generate_temperature_zone_main(
    gcm_member: str,
    scenario: str,
    output_dir: str | Path,
) -> None:
    """Generate the temperature zone for a given scenario and gcm member.

    Parameters
    ----------
    gcm_member
        The gcm member to generate the temperature zone for.
    scenario
        The scenario to generate the temperature zone for.  Pass ``historical``
        (with ``gcm_member="era5"``) to build a pure-ERA5 zone from the raw
        historical annual mean temperature (1950-2025) rather than the compiled
        historical+forecast series; the output then spans 1990-2025.
    output_dir
        The directory to save the temperature zone to.
    """
    print(f"Generating temperature zone for {scenario} {gcm_member}")
    cdata = ClimateData(output_dir)
    if scenario == "historical":
        paths = sorted(
            (cdata.raw_annual_results / "historical" / "mean_temperature").glob("*.nc")
        )
        ds = xr.open_mfdataset(paths, combine="by_coords").sortby("year").compute()
    else:
        ds = cdata.load_compiled_annual_results(
            scenario, "mean_temperature", gcm_member
        )
    temperature_zone = ds.rolling(year=10).mean().sel(year=slice(1990, 2100))
    print(f"Saving temperature zone for {scenario} {gcm_member}")
    cdata.save_compiled_annual_results(
        temperature_zone,
        scenario=scenario,
        variable="temperature_zone",
        gcm_member=gcm_member,
        encoding_kwargs={"scale_factor": 0.01, "add_offset": 0.0},
    )


@click.command()
@clio.with_gcm_member()
@clio.with_scenario()
@clio.with_output_directory(cdc.MODEL_ROOT)
def generate_temperature_zone_task(
    gcm_member: str,
    scenario: str,
    output_dir: str,
) -> None:
    if scenario == "historical" and gcm_member != "era5":
        msg = f"The 'historical' scenario must use the 'era5' gcm-member, got {gcm_member}"
        raise ValueError(msg)
    generate_temperature_zone_main(gcm_member, scenario, output_dir)


@click.command()
@clio.with_scenario(allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_queue()
@clio.with_overwrite()
@clio.with_dry_run()
def generate_temperature_zone(
    scenario: list[str],
    output_dir: str,
    queue: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    cdata = ClimateData(output_dir)

    complete = []
    to_run = []
    for e in scenario:
        gcm_members = (
            ["era5"]
            if e == "historical"
            else cdata.list_gcm_members("ssp126", "mean_temperature")
        )
        for g in gcm_members:
            path = cdata.compiled_annual_results_path(e, "temperature_zone", g)
            if not path.exists() or overwrite:
                to_run.append((e, g))
            else:
                complete.append((e, g))

    print(f"{len(complete)} tasks already done. Launching {len(to_run)} tasks")

    run_parallel_maybe_dry_run(
        runner="cdtask special",
        task_name="temperature_zone",
        flat_node_args=(
            ("scenario", "gcm-member"),
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
        dry_run=dry_run,
    )
