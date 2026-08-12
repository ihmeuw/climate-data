"""
CMIP6 Data Extraction
---------------------
"""

from pathlib import Path

import click
import gcsfs
import xarray as xr
from rra_tools import shell_tools

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData
from climate_data.jobmon_utils import run_parallel_maybe_dry_run

INT16_MAX = 32767


def check_encoding_covers(
    data_min: float,
    data_max: float,
    offset: float,
    scale: float,
    variable: str,
) -> None:
    """Refuse to write values the int16 encoding cannot represent.

    Packing to int16 stores `(value - offset) / scale`, and anything beyond +/-32767
    wraps modulo 65536. `to_netcdf` does this silently, so the corruption is invisible
    until someone plots the result and finds negative rainfall. `pr` shipped with
    `scale_factor=1e-9` for two years -- a 2.83 mm/day ceiling -- and produced 295 files
    in which 26.4% of sampled cells were wrong and 12.4% were negative.

    Raising here makes the next such mistake a failed extract rather than a corrupt
    archive.
    """
    for label, value in (("minimum", data_min), ("maximum", data_max)):
        stored = (value - offset) / scale
        if abs(stored) > INT16_MAX:
            msg = (
                f"The {label} value of {variable} ({value:g}) cannot be represented by"
                f" its int16 encoding (offset={offset:g}, scale={scale:g}): it would be"
                f" stored as {stored:.0f}, outside +/-{INT16_MAX}, and would wrap modulo"
                f" 65536. Widen scale_factor in constants.CMIP6_VARIABLES."
            )
            raise ValueError(msg)


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
    cmip6_source: str,
    cmip6_experiment: str,
    cmip6_variable: str,
    output_dir: str | Path,
    overwrite: bool,
) -> None:
    print(f"Checking metadata for {cmip6_source} {cmip6_experiment} {cmip6_variable}")
    cdata = ClimateData(output_dir)
    meta = cdata.load_cmip6_metadata()

    *_, offset, scale, table_id = cdc.CMIP6_VARIABLES.get(cmip6_variable)

    mask = (
        (meta.source_id == cmip6_source)
        & (meta.experiment_id == cmip6_experiment)
        & (meta.variable_id == cmip6_variable)
        & (meta.table_id == table_id)
    )

    meta_subset = meta[mask].set_index("member_id").zstore.to_dict()
    print(f"Extracting {len(meta_subset)} members...")

    for i, (member, zstore_path) in enumerate(meta_subset.items()):
        item = f"{i + 1}/{len(meta_subset)} {member}"
        # Keywords, not positionals: this call previously passed
        # (experiment, variable, member) into a (variable, experiment, gcm_member)
        # signature, so it wrote `ssp126_pr_<member>.nc` while the generate stage looks
        # up `pr_ssp126_<member>.nc`. Naming the arguments makes that unrepeatable.
        out_path = cdata.extracted_cmip6_path(
            variable=cmip6_variable,
            experiment=cmip6_experiment,
            gcm_member=str(member),
        )
        if out_path.exists() and not overwrite:
            print("Skipping", item)
            continue

        try:
            print("Extracting", item)
            cmip_data = load_cmip_data(zstore_path)

            # Costs one extra pass over the data, which `to_netcdf` would read anyway.
            # Worth it: a silently wrapped extract is only detectable downstream, and
            # only by someone who notices negative rainfall.
            check_encoding_covers(
                data_min=float(cmip_data[cmip6_variable].min()),
                data_max=float(cmip_data[cmip6_variable].max()),
                offset=offset,
                scale=scale,
                variable=cmip6_variable,
            )

            print("Writing to", out_path)
            shell_tools.touch(out_path, clobber=True)

            cmip_data.to_netcdf(
                out_path,
                encoding={
                    cmip6_variable: {
                        "dtype": "int16",
                        "scale_factor": scale,
                        "add_offset": offset,
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


@click.command()
@clio.with_cmip6_source()
@clio.with_cmip6_experiment()
@clio.with_cmip6_variable()
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_overwrite()
def extract_cmip6_task(
    cmip6_source: str,
    cmip6_experiment: str,
    cmip6_variable: str,
    output_dir: str,
    overwrite: bool,
) -> None:
    extract_cmip6_main(
        cmip6_source,
        cmip6_experiment,
        cmip6_variable,
        output_dir,
        overwrite,
    )


@click.command()
@clio.with_cmip6_source(allow_all=True)
@clio.with_cmip6_experiment(allow_all=True)
@clio.with_cmip6_variable(allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_queue()
@clio.with_overwrite()
@clio.with_dry_run()
def extract_cmip6(
    cmip6_source: list[str],
    cmip6_experiment: list[str],
    cmip6_variable: list[str],
    output_dir: str,
    queue: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """Extract CMIP6 data.

    Extracts CMIP6 data for the given source, experiment, and variable. We use the
    the table at https://www.nature.com/articles/s41597-023-02549-6/tables/3 to determine
    which CMIP6 source_ids to include. See `ClimateData.load_koppen_geiger_model_inclusion`
    to load and examine this table. The extraction criteria does not completely
    capture model inclusion criteria as it does not account for the year range avaialable
    in the data. This determiniation is made when we proccess the data in later steps.
    """
    overwrite_arg = {"overwrite": None} if overwrite else {}

    run_parallel_maybe_dry_run(
        runner="cdtask",
        task_name="extract cmip6",
        node_args={
            "cmip6-source": cmip6_source,
            "cmip6-experiment": cmip6_experiment,
            "cmip6-variable": cmip6_variable,
        },
        task_args={
            "output-dir": output_dir,
            **overwrite_arg,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "3000m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        concurrency_limit=50,
        dry_run=dry_run,
    )
