"""
CMIP6 Data Extraction
---------------------
"""

from pathlib import Path

import click
import gcsfs
import pandas as pd
import xarray as xr
from rra_tools import shell_tools

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData, gcm_member_id
from climate_data.jobmon_utils import run_parallel_maybe_dry_run

INT16_MAX = 32767
UINT16_MAX = 65535

# The code reserved for missing data, and the range of codes left for real values. Both
# dtypes give up one code to `_FillValue`; the difference is where it sits. Signed puts it
# at the bottom, so the usable floor is one above it. Unsigned puts it at the top, which
# leaves zero -- overwhelmingly the most common precipitation value -- encoding as 0,
# where it cannot be mistaken for missing.
FILL_VALUE = {"int16": -INT16_MAX, "uint16": UINT16_MAX}
STORED_LIMITS = {
    "int16": (-INT16_MAX + 1, INT16_MAX),
    "uint16": (0, UINT16_MAX - 1),
}


def check_encoding_covers(
    data_min: float,
    data_max: float,
    offset: float,
    scale: float,
    variable: str,
    dtype: str = "int16",
) -> None:
    """Refuse to write values the declared encoding cannot represent.

    Packing to a 16-bit integer stores `(value - offset) / scale`, and anything outside
    the type's range wraps modulo 65536. `to_netcdf` does this silently, so the corruption
    is invisible until someone plots the result and finds negative rainfall. `pr` shipped
    with `scale_factor=1e-9` for two years -- a 2.83 mm/day ceiling -- and produced 295
    files in which 26.4% of sampled cells were wrong and 12.4% were negative.

    Raising here makes the next such mistake a failed extract rather than a corrupt
    archive. An unsigned dtype also makes a negative value an error rather than a wrap,
    which for a flux like precipitation is the honest outcome.
    """
    low, high = STORED_LIMITS[dtype]
    for label, value in (("minimum", data_min), ("maximum", data_max)):
        stored = (value - offset) / scale
        if not low <= stored <= high:
            msg = (
                f"The {label} value of {variable} ({value:g}) cannot be represented by"
                f" its {dtype} encoding (offset={offset:g}, scale={scale:g}): it would be"
                f" stored as {stored:.0f}, outside [{low}, {high}], and would wrap modulo"
                f" 65536. Widen encoding_scale for {variable} in"
                f" constants.CMIP6_VARIABLES."
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


def select_members(
    meta: pd.DataFrame,
    cmip6_source: str,
    cmip6_experiment: str,
    cmip6_variable: str,
    gcm_member: str | None = None,
) -> dict[str, str]:
    """The `{member_id: zstore}` this job should extract.

    With `gcm_member` given, narrows to that one ensemble member so the runner can put
    each member in its own job. Shared with the runner, which enumerates the same space
    to build its task list -- if these two disagreed, the runner would submit jobs whose
    member does not exist and they would silently extract nothing.
    """
    table_id = cdc.CMIP6_VARIABLES.get(cmip6_variable).table_id
    mask = (
        (meta.source_id == cmip6_source)
        & (meta.experiment_id == cmip6_experiment)
        & (meta.variable_id == cmip6_variable)
        & (meta.table_id == table_id)
    )
    # `to_dict` on a pandas index is `dict[Hashable, Any]`; coerce once here so callers
    # get plain strings and do not each have to re-cast the variant.
    members: dict[str, str] = {}
    for variant, zstore in meta[mask].set_index("member_id").zstore.to_dict().items():
        members[str(variant)] = str(zstore)
    if gcm_member is None:
        return members

    selected = {}
    for variant, zstore in members.items():
        if gcm_member_id(cmip6_source, variant) == gcm_member:
            selected[variant] = zstore
    if not selected:
        available = []
        for variant in members:
            available.append(gcm_member_id(cmip6_source, variant))
        msg = (
            f"No CMIP6 member {gcm_member!r} for {cmip6_source} {cmip6_experiment}"
            f" {cmip6_variable}. Available: {sorted(available)}"
        )
        raise ValueError(msg)
    return selected


def extract_cmip6_main(
    cmip6_source: str,
    cmip6_experiment: str,
    cmip6_variable: str,
    output_dir: str | Path,
    overwrite: bool,
    gcm_member: str | None = None,
) -> None:
    print(f"Checking metadata for {cmip6_source} {cmip6_experiment} {cmip6_variable}")
    cdata = ClimateData(output_dir)
    meta = cdata.load_cmip6_metadata()

    # Attributes, not positional unpacking: the record grew an `encoding_dtype` field and
    # a `*_, offset, scale, table_id` unpack would have silently shifted by one.
    variable_spec = cdc.CMIP6_VARIABLES.get(cmip6_variable)
    offset = variable_spec.encoding_offset
    scale = variable_spec.encoding_scale
    dtype = variable_spec.encoding_dtype
    meta_subset = select_members(
        meta, cmip6_source, cmip6_experiment, cmip6_variable, gcm_member
    )
    print(f"Extracting {len(meta_subset)} members...")

    for i, (member, zstore_path) in enumerate(meta_subset.items()):
        item = f"{i + 1}/{len(meta_subset)} {cmip6_source} {member}"
        # Keywords, not positionals: this call previously passed
        # (experiment, variable, member) into a (variable, experiment, gcm_member)
        # signature, so it wrote `ssp126_pr_<member>.nc` while the generate stage looks
        # up `pr_ssp126_<gcm_member>.nc`. Naming the arguments makes that unrepeatable.
        #
        # `gcm_member` is `<source>_<variant>`, not the bare `member_id` this loop
        # iterates: `member_id` is unique only within a source, so keying on it alone
        # made every source's `r1i1p1f1` write the same file. `data.gcm_member_id` is
        # shared with `ClimateData.get_gcms`, which is what reads these back.
        out_path = cdata.extracted_cmip6_path(
            variable=cmip6_variable,
            experiment=cmip6_experiment,
            gcm_member=gcm_member_id(cmip6_source, member),
        )
        if out_path.exists() and not overwrite:
            print("Skipping", item)
            continue

        started_write = False
        try:
            print("Extracting", item)
            cmip_data = load_cmip_data(zstore_path)

            # Costs one extra pass over the data, which `to_netcdf` would read anyway.
            # Worth it: a silently wrapped extract is only detectable downstream, and
            # only by someone who notices negative rainfall. Both bounds are reduced in
            # one `compute` so it stays a single traversal of the GCS-backed zarr --
            # computing them separately streamed the whole array twice.
            arr = cmip_data[cmip6_variable]
            bounds = xr.Dataset({"min": arr.min(), "max": arr.max()}).compute()
            check_encoding_covers(
                data_min=float(bounds["min"]),
                data_max=float(bounds["max"]),
                offset=offset,
                scale=scale,
                variable=cmip6_variable,
                dtype=dtype,
            )

            print("Writing to", out_path)
            shell_tools.touch(out_path, clobber=True)
            started_write = True

            cmip_data.to_netcdf(
                out_path,
                encoding={
                    cmip6_variable: {
                        "dtype": dtype,
                        "scale_factor": scale,
                        "add_offset": offset,
                        "_FillValue": FILL_VALUE[dtype],
                        "zlib": True,
                        "complevel": 1,
                    }
                },
            )
        except Exception as e:
            # Only clear a file this invocation started writing. Everything before
            # `touch` -- opening the zarr, and the encoding guard rejecting the data --
            # fails with the previous extract still intact on disk, and deleting that
            # leaves neither a corrected file nor the one it was meant to replace.
            if started_write and out_path.exists():
                out_path.unlink()
            raise e


@click.command()
@clio.with_cmip6_source()
@clio.with_cmip6_experiment()
@clio.with_cmip6_variable()
@clio.with_gcm_member()
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_overwrite()
def extract_cmip6_task(
    cmip6_source: str,
    cmip6_experiment: str,
    cmip6_variable: str,
    gcm_member: str | None,
    output_dir: str,
    overwrite: bool,
) -> None:
    extract_cmip6_main(
        cmip6_source,
        cmip6_experiment,
        cmip6_variable,
        output_dir,
        overwrite,
        gcm_member,
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

    Fans out one job per ensemble member rather than one per (source, experiment). The
    member counts are wildly uneven -- MIROC6 has 50 `pr` members where most sources have
    one -- so grouping them made three jobs carry fifty times the work of a typical one.
    More importantly, `extract_cmip6_main` re-raises on failure, so a member the encoding
    guard rejects used to abandon every member behind it in the same job. One job per
    member contains that to the member that failed, and makes a resumed run skip the
    members already written instead of redoing whole groups.

    The member space is not a cartesian product -- not every source publishes every
    variant for every experiment -- so it is enumerated from the metadata and passed as
    `flat_node_args`.
    """
    overwrite_arg = {"overwrite": None} if overwrite else {}

    cdata = ClimateData(output_dir)
    meta = cdata.load_cmip6_metadata()

    to_run = []
    complete = []
    for variable in cmip6_variable:
        for experiment in cmip6_experiment:
            for source in cmip6_source:
                members = select_members(meta, source, experiment, variable)
                for variant in members:
                    member = gcm_member_id(source, variant)
                    path = cdata.extracted_cmip6_path(variable, experiment, member)
                    if path.exists() and not overwrite:
                        complete.append(member)
                    else:
                        to_run.append((source, experiment, variable, member))

    if not to_run:
        print("All tasks already done.")
        return

    print(f"{len(complete)} tasks already done. Launching {len(to_run)} tasks")
    run_parallel_maybe_dry_run(
        runner="cdtask",
        task_name="extract cmip6",
        flat_node_args=(
            ("cmip6-source", "cmip6-experiment", "cmip6-variable", "gcm-member"),
            to_run,
        ),
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
