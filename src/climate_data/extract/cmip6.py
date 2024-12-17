from pathlib import Path
from typing import Literal, NamedTuple, ParamSpec, TypeVar

import click
import gcsfs
import xarray as xr
from rra_tools import cli_tools, jobmon, shell_tools

from climate_data import cli_options as clio
from climate_data.data import DEFAULT_ROOT, ClimateData


class VariableSpec(NamedTuple):
    shift: float
    scale: float
    table_id: Literal["day", "Oday"] = "day"


CMIP6_VARIABLES = {
    "uas": VariableSpec(0.0, 0.01),
    "vas": VariableSpec(0.0, 0.01),
    "hurs": VariableSpec(0.0, 0.01),
    "tas": VariableSpec(273.15, 0.01),
    "tasmin": VariableSpec(273.15, 0.01),
    "tasmax": VariableSpec(273.15, 0.01),
    "tos": VariableSpec(273.15, 0.01, table_id="Oday"),
    "pr": VariableSpec(0.0, 1e-9),
}

_T = TypeVar("_T")
_P = ParamSpec("_P")


def with_cmip6_variable(
    *,
    allow_all: bool = False,
) -> cli_tools.ClickOption[_P, _T]:
    return cli_tools.with_choice(
        "cmip6-variable",
        "x",
        allow_all=allow_all,
        choices=list(CMIP6_VARIABLES),
        help="CMIP6 variable to extract.",
    )


def load_cmip_data(zarr_path: str) -> xr.Dataset:
    """Loads a CMIP6 dataset from a zarr path."""
    gcs = gcsfs.GCSFileSystem(token="anon")  # noqa: S106
    mapper = gcs.get_mapper(zarr_path)
    ds = xr.open_zarr(mapper, consolidated=True)
    ds = ds.drop_vars(
        ["lat_bnds", "lon_bnds", "time_bnds", "height", "time_bounds", "bnds"],
        errors="ignore",
    )
    return ds  # type: ignore[no-any-return]


def extract_cmip6_main(
    output_dir: str | Path,
    cmip6_source: str,
    cmip6_experiment: str,
    cmip6_variable: str,
    overwrite: bool,
) -> None:
    print(f"Checking metadata for {cmip6_source} {cmip6_experiment} {cmip6_variable}")
    cd_data = ClimateData(output_dir)
    meta = cd_data.load_cmip6_metadata()

    shift, scale, table_id = CMIP6_VARIABLES[cmip6_variable]

    mask = (
        (meta.source_id == cmip6_source)
        & (meta.experiment_id == cmip6_experiment)
        & (meta.variable_id == cmip6_variable)
        & (meta.table_id == table_id)
    )

    meta_subset = meta[mask].set_index("member_id").zstore.to_dict()
    print(f"Extracting {len(meta_subset)} members...")

    for i, (member, zstore_path) in enumerate(meta_subset.items()):
        item = f"{i}/{len(meta_subset)} {member}"
        out_path = cd_data.extracted_cmip6_path(
            cmip6_variable,
            cmip6_experiment,
            cmip6_source,
            member,
        )
        if out_path.exists() and not overwrite:
            print("Skipping", item)
            continue

        try:
            print("Extracting", item)
            cmip_data = load_cmip_data(zstore_path)

            shell_tools.touch(out_path, exist_ok=True)
            print("Writing to", out_path)
            cmip_data.to_netcdf(
                out_path,
                encoding={
                    cmip6_variable: {
                        "dtype": "int16",
                        "scale_factor": scale,
                        "add_offset": shift,
                        "_FillValue": -32767,
                        "zlib": True,
                        "complevel": 1,
                    }
                },
            )
        except Exception as e:
            if out_path.exists():
                out_path.unlink()
            raise e


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_cmip6_source()
@clio.with_cmip6_experiment()
@with_cmip6_variable()
@clio.with_overwrite()
def extract_cmip6_task(
    output_dir: str,
    cmip6_source: str,
    cmip6_experiment: str,
    cmip6_variable: str,
    overwrite: bool,
) -> None:
    extract_cmip6_main(
        output_dir, cmip6_source, cmip6_experiment, cmip6_variable, overwrite
    )


@click.command()  # type: ignore[arg-type]
@clio.with_cmip6_source(allow_all=True)
@clio.with_cmip6_experiment(allow_all=True)
@with_cmip6_variable(allow_all=True)
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_queue()
@clio.with_overwrite()
def extract_cmip6(
    cmip6_source: str,
    cmip6_experiment: str,
    cmip6_variable: str,
    output_dir: str,
    queue: str,
    overwrite: bool,
) -> None:
    """Extract CMIP6 data.

    Extracts CMIP6 data for the given source, experiment, and variable. We use the
    the table at https://www.nature.com/articles/s41597-023-02549-6/tables/3 to determine
    which CMIP6 source_ids to include. See `ClimateData.load_koppen_geiger_model_inclusion`
    to load and examine this table. The extraction criteria does not completely
    capture model inclusion criteria as it does not account for the year range avaialable
    in the data. This determiniation is made when we proccess the data in later steps.
    """
    sources = (
        clio.VALID_CMIP6_SOURCES if cmip6_source == clio.RUN_ALL else [cmip6_source]
    )
    experiments = (
        clio.VALID_CMIP6_EXPERIMENTS
        if cmip6_experiment == clio.RUN_ALL
        else [cmip6_experiment]
    )
    variables = (
        list(CMIP6_VARIABLES) if cmip6_variable == clio.RUN_ALL else [cmip6_variable]
    )

    overwrite_arg = {"overwrite": None} if overwrite else {}

    jobmon.run_parallel(
        runner="cdtask",
        task_name="extract cmip6",
        node_args={
            "cmip6-source": sources,
            "cmip6-experiment": experiments,
            "cmip6-variable": variables,
        },
        task_args={
            "output-dir": output_dir,
            **overwrite_arg,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "600m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        concurrency_limit=50,
    )
