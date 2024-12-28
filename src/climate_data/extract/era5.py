"""
ERA5 Data Extraction
--------------------
"""

import itertools
import zipfile
from pathlib import Path

import cdsapi
import click
import xarray as xr
import yaml
from rra_tools import jobmon
from rra_tools.shell_tools import touch

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData

_NETCDF_VALID_ENCODINGS = {
    "zlib",
    "complevel",
    "fletcher32",
    "contiguous",
    "chunksizes",
    "shuffle",
    "_FillValue",
    "dtype",
    "compression",
    "significant_digits",
    "quantize_mode",
    "blosc_shuffle",
    "szip_coding",
    "szip_pixels_per_block",
    "endian",
}


def download_era5_main(
    era5_dataset: str,
    era5_variable: str,
    month: str,
    year: int | str,
    user: str,
    output_dir: str | Path,
) -> None:
    cdata = ClimateData(output_dir)

    final_out_path = cdata.extracted_era5_path(era5_dataset, era5_variable, year, month)
    download_path = final_out_path.with_suffix(".zip")
    data_format, download_format = "netcdf", "zip"

    if download_path.exists():
        print("Already downloaded:", download_path)
        return

    try:
        touch(download_path)

        print("Connecting to copernicus")

        cred_path = cdata.credentials_root / "copernicus.yaml"
        credentials = yaml.safe_load(cred_path.read_text())
        url = credentials["url"]
        key = credentials["keys"][user]
        copernicus = cdsapi.Client(url=url, key=key)

        print("Downloading...")
        kwargs = {
            "product_type": ["reanalysis"],
            "variable": [era5_variable],
            "year": [year],
            "month": [month],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": data_format,
            "download_format": download_format,
        }

        result = copernicus.retrieve(
            era5_dataset,
            kwargs,
        )
        result.download(download_path)
    except Exception as e:
        print(f"Failed to download {era5_dataset} {era5_variable} {year} {month}")
        if download_path.exists():
            download_path.unlink()
        raise e


def check_zipfile(zip_path: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path):
            pass
    except zipfile.BadZipFile as e:
        # Download failed or was interrupted, delete the zipfile
        zip_path.unlink()
        raise e


def unzip_and_compress_era5(
    era5_dataset: str,
    era5_variable: str,
    month: str,
    year: int | str,
    output_dir: str | Path,
) -> None:
    cdata = ClimateData(output_dir)

    final_out_path = cdata.extracted_era5_path(era5_dataset, era5_variable, year, month)

    zip_path = final_out_path.with_suffix(".zip")
    check_zipfile(zip_path)

    uncompressed_path = final_out_path.with_stem(f"{final_out_path.stem}_raw")
    if uncompressed_path.exists():
        uncompressed_path.unlink()
    touch(uncompressed_path)

    print("Unzipping...")
    with zipfile.ZipFile(zip_path) as zf:
        zinfo = zf.infolist()
        if len(zinfo) != 1:
            msg = f"Expected a single file in {zip_path}"
            raise ValueError(msg)
        with uncompressed_path.open("wb") as f:
            f.write(zf.read(zinfo[0]))

    print("Compressing")
    if final_out_path.exists():
        final_out_path.unlink()
    touch(final_out_path)
    ds = xr.open_dataset(uncompressed_path)
    var_name = next(iter(ds))  # These are all single variable datasets
    og_encoding = ds[var_name].encoding
    og_encoding = {k: v for k, v in og_encoding.items() if k in _NETCDF_VALID_ENCODINGS}
    ds.to_netcdf(
        final_out_path,
        encoding={
            var_name: {
                **og_encoding,
                "zlib": True,
                "complevel": 1,
            }
        },
    )

    if zip_path.exists():
        zip_path.unlink()
    uncompressed_path.unlink()


@click.command()  # type: ignore[arg-type]
@clio.with_era5_dataset()
@clio.with_era5_variable()
@clio.with_month()
@clio.with_year(years=cdc.FULL_HISTORY_YEARS)
@click.option("--user", type=str)
@clio.with_output_directory(cdc.MODEL_ROOT)
def download_era5_task(
    era5_dataset: str,
    era5_variable: str,
    month: str,
    year: str,
    user: str,
    output_dir: str,
) -> None:
    download_era5_main(
        era5_dataset,
        era5_variable,
        month,
        year,
        user,
        output_dir,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_era5_dataset()
@clio.with_era5_variable()
@clio.with_month()
@clio.with_year(years=cdc.FULL_HISTORY_YEARS)
@clio.with_output_directory(cdc.MODEL_ROOT)
def unzip_and_compress_era5_task(
    era5_dataset: str,
    era5_variable: str,
    month: str,
    year: str,
    output_dir: str,
) -> None:
    unzip_and_compress_era5(
        era5_dataset,
        era5_variable,
        month,
        year,
        output_dir,
    )


def build_task_lists(
    cdata: ClimateData,
    *spec_variables: list[str],
) -> tuple[list[tuple[str, ...]], ...]:
    to_download = []
    to_compress = []
    complete = []
    for spec in itertools.product(*spec_variables):
        dataset, variable, *_ = spec
        if (
            variable == cdc.ERA5_VARIABLES.sea_surface_temperature
            and dataset != cdc.ERA5_DATASETS.reanalysis_era5_single_levels
        ):
            # This variable is only available in the single levels dataset
            continue

        final_out_path = cdata.extracted_era5_path(*spec)
        zip_path = final_out_path.with_suffix(".zip")
        uncompressed_path = final_out_path.with_stem(f"{final_out_path.stem}_raw")

        if zip_path.exists() and uncompressed_path.exists():
            # We broke in the middle of processing this file. Don't re-download,
            # just reprocess.
            uncompressed_path.unlink()
            to_compress.append(spec)
        elif uncompressed_path.exists() and final_out_path.exists():
            # We broke while compressing. Just re-compress
            final_out_path.unlink()
            to_compress.append(spec)
        elif final_out_path.exists() and final_out_path.stat().st_size == 0:
            # Some other kind of error happened
            final_out_path.unlink()
            to_download.append(spec)
            to_compress.append(spec)
        elif zip_path.exists() and zip_path.stat().st_size == 0:
            # We broke while downloading. Assume this file is invalid and re-download
            zip_path.unlink()
            to_download.append(spec)
            to_compress.append(spec)
        elif zip_path.exists():
            to_compress.append(spec)
        elif final_out_path.exists():
            # We've already extracted this dataset
            # (deleting the download path is the last step)
            complete.append(spec)
            continue
        else:
            to_download.append(spec)
            to_compress.append(spec)

    return to_download, to_compress, complete


@click.command()  # type: ignore[arg-type]
@clio.with_era5_dataset(allow_all=True)
@clio.with_era5_variable(allow_all=True)
@clio.with_month(allow_all=True)
@clio.with_year(years=cdc.FULL_HISTORY_YEARS, allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_queue()
def extract_era5(
    output_dir: str,
    era5_dataset: str,
    era5_variable: str,
    year: str,
    month: str,
    queue: str,
) -> None:
    cdata = ClimateData(output_dir)
    cred_path = cdata.credentials_root / "copernicus.yaml"
    credentials = yaml.safe_load(cred_path.read_text())
    users = list(credentials["keys"])
    jobs_per_user = 20

    datasets = (
        list(cdc.ERA5_DATASETS) if era5_dataset == clio.RUN_ALL else [era5_dataset]
    )
    variables = (
        list(cdc.ERA5_VARIABLES) if era5_variable == clio.RUN_ALL else [era5_variable]
    )
    years = cdc.FULL_HISTORY_YEARS if year == clio.RUN_ALL else [year]
    months = cdc.MONTHS if month == clio.RUN_ALL else [month]

    to_download, to_compress, complete = build_task_lists(
        cdata,
        datasets,
        variables,
        years,
        months,
    )

    if not to_download:
        print("No datasets to download")

    while to_download:
        downloads_left = len(to_download)

        download_batch = []
        for _ in range(jobs_per_user):
            for user in users:
                if to_download:
                    download_batch.append((*to_download.pop(), user))  # noqa: PERF401
        if len(download_batch) != min(len(users) * jobs_per_user, downloads_left):
            msg = "Download batch size is incorrect"
            raise ValueError(msg)

        print(
            len(to_download) + len(download_batch),
            "remaining.  Launching next",
            len(download_batch),
            "jobs",
        )

        jobmon.run_parallel(
            runner="cdtask",
            task_name="extract era5_download",
            flat_node_args=(
                ("era5-dataset", "era5-variable", "year", "month", "user"),
                download_batch,
            ),
            task_args={
                "output-dir": output_dir,
            },
            task_resources={
                "queue": queue,
                "cores": 1,
                "memory": "10G",
                "runtime": "3600m",
                "project": "proj_rapidresponse",
            },
            max_attempts=1,
        )

    if not to_compress:
        print("No datasets to compress.")
        return

    jobmon.run_parallel(
        runner="cdtask",
        task_name="extract era5_compress",
        flat_node_args=(
            ("era5-dataset", "era5-variable", "year", "month"),
            to_compress,
        ),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "125G",
            "runtime": "30m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        concurrency_limit=500,
    )
