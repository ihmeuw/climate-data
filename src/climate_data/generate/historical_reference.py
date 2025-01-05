import click
import xarray as xr
from rra_tools import jobmon

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData
from climate_data.generate.historical_daily import (
    TRANSFORM_MAP,
)


def generate_historical_reference_main(
    target_variable: str,
    output_dir: str,
) -> None:
    cdata = ClimateData(output_dir)
    paths = [
        cdata.daily_results_path("historical", target_variable, year)
        for year in cdc.REFERENCE_YEARS
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

    print("Averaging years by month")
    reference = sum(reference_data) / len(reference_data)
    print("Saving reference data")
    cdata.save_daily_results(
        reference,  # type: ignore[arg-type]
        scenario="historical",
        variable=target_variable,
        year="reference",
        encoding_kwargs=old_encoding,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_target_variable(TRANSFORM_MAP)
@clio.with_output_directory(cdc.MODEL_ROOT)
def generate_historical_reference_task(
    target_variable: str,
    output_dir: str,
) -> None:
    generate_historical_reference_main(target_variable, output_dir)


@click.command()  # type: ignore[arg-type]
@clio.with_target_variable(TRANSFORM_MAP, allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_queue()
def generate_historical_reference(
    target_variable: list[str],
    output_dir: str,
    queue: str,
) -> None:
    jobmon.run_parallel(
        runner="cdtask",
        task_name="generate historical_reference",
        node_args={
            "target-variable": target_variable,
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
