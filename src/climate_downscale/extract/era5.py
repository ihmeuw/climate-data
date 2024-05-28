import itertools
import zipfile
from pathlib import Path

import cdsapi
import click
import xarray as xr
from rra_tools import jobmon
from rra_tools.shell_tools import touch

from climate_downscale import cli_options as clio
from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData


def get_download_spec(
    final_out_path: Path,
) -> tuple[Path, str]:
    if "land" in final_out_path.stem:
        download_path = final_out_path.with_suffix(".zip")
        download_format = "netcdf.zip"
    else:
        download_path = final_out_path.with_stem(f"{final_out_path.stem}_raw")
        download_format = "netcdf"
    return download_path, download_format


def download_era5_main(
    output_dir: str | Path,
    era5_dataset: str,
    climate_variable: str,
    year: int | str,
    month: str,
) -> None:
    cddata = ClimateDownscaleData(output_dir)

    final_out_path = cddata.era5_path(era5_dataset, climate_variable, year, month)
    download_path, download_format = get_download_spec(final_out_path)

    if download_path.exists():
        print("Already downloaded:", download_path)
        return

    try:
        touch(download_path)

        print("Connecting to copernicus")

        cred_path = cddata.credentials_root / "copernicus.txt"
        url, key = cred_path.read_text().strip().split("\n")
        copernicus = cdsapi.Client(url=url, key=key)

        print("Downloading...")
        kwargs = {
            "product_type": "reanalysis",
            "variable": climate_variable,
            "year": year,
            "month": month,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "format": download_format,
        }

        result = copernicus.retrieve(
            era5_dataset,
            kwargs,
        )
        result.download(download_path)
    except Exception as e:
        print(f"Failed to download {era5_dataset} {climate_variable} {year} {month}")
        if download_path.exists():
            download_path.unlink()
        raise e  # noqa: TRY201


def unzip_and_compress_era5(
    output_dir: str | Path,
    era5_dataset: str,
    climate_variable: str,
    year: int | str,
    month: str,
) -> None:
    cddata = ClimateDownscaleData(output_dir)
    final_out_path = cddata.era5_path(era5_dataset, climate_variable, year, month)
    uncompressed_path = final_out_path.with_stem(f"{final_out_path.stem}_raw")

    if era5_dataset == "reanalysis-era5-land":
        print("Unzipping...")
        # This data needs to be unzipped first.
        zip_path = final_out_path.with_suffix(".zip")
        touch(uncompressed_path)
        with zipfile.ZipFile(zip_path) as zf:
            zinfo = zf.infolist()
            if len(zinfo) != 1:
                msg = f"Expected a single file in {zip_path}"
                raise ValueError(msg)
            zf.extract(zinfo[0], uncompressed_path)

    touch(final_out_path)
    ds = xr.open_dataset(final_out_path)
    var_name = next(iter(ds))  # These are all single variable datasets
    og_encoding = ds[var_name].encoding
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


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_era5_dataset()
@clio.with_climate_variable()
@clio.with_year()
@clio.with_month()
def download_era5_task(
    output_dir: str,
    era5_dataset: str,
    climate_variable: str,
    year: str,
    month: str,
) -> None:
    download_era5_main(
        output_dir,
        era5_dataset,
        climate_variable,
        year,
        month,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_era5_dataset()
@clio.with_climate_variable()
@clio.with_year()
@clio.with_month()
def unzip_and_compress_era5_task(
    output_dir: str,
    era5_dataset: str,
    climate_variable: str,
    year: str,
    month: str,
) -> None:
    unzip_and_compress_era5(
        output_dir,
        era5_dataset,
        climate_variable,
        year,
        month,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_era5_dataset(allow_all=True)
@clio.with_climate_variable(allow_all=True)
@clio.with_year(allow_all=True)
@clio.with_month(allow_all=True)
@clio.with_queue()
def extract_era5(  # noqa: PLR0913
    output_dir: str,
    era5_dataset: str,
    climate_variable: str,
    year: str,
    month: str,
    queue: str,
) -> None:
    cddata = ClimateDownscaleData(output_dir)

    datasets = (
        clio.VALID_ERA5_DATASETS if era5_dataset == clio.RUN_ALL else [era5_dataset]
    )
    variables = (
        clio.VALID_CLIMATE_VARIABLES
        if climate_variable == clio.RUN_ALL
        else [climate_variable]
    )
    years = clio.VALID_YEARS if year == clio.RUN_ALL else [year]
    months = clio.VALID_MONTHS if month == clio.RUN_ALL else [month]

    to_download = []
    to_compress = []
    for dataset, variable, year, month in itertools.product(
        datasets, variables, years, months
    ):
        final_out_path = cddata.era5_path(era5_dataset, climate_variable, year, month)
        download_path, _ = get_download_spec(final_out_path)

        if final_out_path.exists() and download_path.exists():
            # We broke in the middle of processing this file. Don't re-download,
            # just reprocess.
            final_out_path.unlink()
            to_compress.append((dataset, variable, year, month))
        elif final_out_path.exists():
            # We've already extracted this dataset
            continue

        to_download.append((dataset, variable, year, month))
        to_compress.append((dataset, variable, year, month))

    jobmon.run_parallel(
        runner="cdtask",
        task_name="extract era5_download",
        flat_node_args=(
            ("era5-dataset", "climate-variable", "year", "month"),
            to_compress,
        ),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "600m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        concurrency_limit=25,
    )

    jobmon.run_parallel(
        runner="cdtask",
        task_name="extract era5_compress",
        flat_node_args=(
            ("era5-dataset", "climate-variable", "year", "month"),
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
