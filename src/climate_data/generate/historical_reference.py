import click
import xarray as xr
from rra_tools import jobmon

from climate_data import cli_options as clio
from climate_data.data import DEFAULT_ROOT, ClimateData
from climate_data.generate.historical_daily import (
    TRANSFORM_MAP,
)


def generate_historical_reference_main(
    output_dir: str,
    target_variable: str,
) -> None:
    cd_data = ClimateData(output_dir)
    paths = [
        cd_data.daily_results_path("historical", target_variable, year)
        for year in clio.VALID_REFERENCE_YEARS
    ]
    print(f"Building reference data from: {len(paths)} files.")

    reference_data = []
    for path in paths:
        print(f"Loading: {path}")
        ds = xr.load_dataset(path)
        print("Computing monthly means")
        ds = ds.groupby("date.month").mean("date")
        reference_data.append(ds)

    old_encoding = {
        k: v
        for k, v in xr.open_dataset(paths[0])["value"].encoding.items()
        if k in ["dtype", "_FillValue", "scale_factor", "add_offset"]
    }
    encoding_kwargs = {
        "zlib": True,
        "complevel": 1,
        **old_encoding,
    }

    print("Averaging years by month")
    reference = sum(reference_data) / len(reference_data)
    print("Saving reference data")
    cd_data.save_daily_results(
        reference,  # type: ignore[arg-type]
        scenario="historical",
        variable=target_variable,
        year="reference",
        draw=None,
        encoding_kwargs=encoding_kwargs,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_target_variable(variable_names=list(TRANSFORM_MAP))
def generate_historical_reference_task(
    output_dir: str,
    target_variable: str,
) -> None:
    generate_historical_reference_main(output_dir, target_variable)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_target_variable(allow_all=True, variable_names=list(TRANSFORM_MAP))
@clio.with_queue()
def generate_historical_reference(
    output_dir: str,
    target_variable: str,
    queue: str,
) -> None:
    variables = (
        list(TRANSFORM_MAP) if target_variable == clio.RUN_ALL else [target_variable]
    )

    jobmon.run_parallel(
        runner="cdtask",
        task_name="generate historical_reference",
        node_args={
            "target-variable": variables,
        },
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "100G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
    )
