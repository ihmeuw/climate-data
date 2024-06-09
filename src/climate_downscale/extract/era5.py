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

import yaml


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
    user: str,
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

        cred_path = cddata.credentials_root / "copernicus.yaml"
        credentials = yaml.safe_load(cred_path.read_text())
        url = credentials['url']
        key = credentials['keys'][user]
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
    zip_path = final_out_path.with_suffix(".zip")
    uncompressed_path = final_out_path.with_stem(f"{final_out_path.stem}_raw")
    
    if era5_dataset == "reanalysis-era5-land":
        print("Unzipping...")
        # This data needs to be unzipped first.    
        if uncompressed_path.exists():
            uncompressed_path.unlink()
        touch(uncompressed_path)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                pass
        except zipfile.BadZipFile as e:
            # Download failed or was interrupted, delete the zipfile
            zip_path.unlink()
            raise e
            
        with zipfile.ZipFile(zip_path) as zf:
            zinfo = zf.infolist()
            if len(zinfo) != 1:
                msg = f"Expected a single file in {zip_path}"
                raise ValueError(msg)
            with uncompressed_path.open('wb') as f:
                f.write(zf.read(zinfo[0]))
        

    print("Compressing")
    touch(final_out_path)
    ds = xr.open_dataset(uncompressed_path)
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
    if zip_path.exists():
        zip_path.unlink()
    uncompressed_path.unlink()


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_era5_dataset()
@clio.with_climate_variable()
@clio.with_year()
@clio.with_month()
@click.option(
    "--user", 
    type=str,
)
def download_era5_task(
    output_dir: str,
    era5_dataset: str,
    climate_variable: str,
    year: str,
    month: str,
    user: str,
) -> None:
    download_era5_main(
        output_dir,
        era5_dataset,
        climate_variable,
        year,
        month,
        user,
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
    cred_path = cddata.credentials_root / "copernicus.yaml"
    credentials = yaml.safe_load(cred_path.read_text())
    users = list(credentials['keys'])
    jobs_per_user = 20
    
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
    complete = []
    for spec in itertools.product(
        datasets, variables, years, months
    ):
        final_out_path = cddata.era5_path(*spec)
        download_path, _ = get_download_spec(final_out_path)        

        if final_out_path.exists() and download_path.exists():
            # We broke in the middle of processing this file. Don't re-download,
            # just reprocess.
            final_out_path.unlink()
            to_compress.append(spec)
        elif final_out_path.exists() and final_out_path.stat().st_size == 0:
            # Some other kind of error happened
            final_out_path.unlink()
            to_download.append(spec)
            to_compress.append(spec)
        elif download_path.exists() and download_path.stat().st_size == 0:
            # We broke while downloading. Assume this file is invalid and re-download            
            download_path.unlink()
            to_download.append(spec)
            to_compress.append(spec)    
        elif download_path.exists():
            to_compress.append(spec)
        elif final_out_path.exists():
            # We've already extracted this dataset (deleting the download path is the last step)
            complete.append(spec)
            continue
        else:
            to_download.append(spec)
            to_compress.append(spec)

    while to_download:
        downloads_left = len(to_download)
        
        
        download_batch = []
        for i in range(jobs_per_user):
            for user in users:
                if to_download:
                    download_batch.append(
                        (*to_download.pop(), user)
                    )
        assert len(download_batch) == min(len(users) * jobs_per_user, downloads_left)        
        
        print(len(to_download) + len(download_batch), "remaining.  Launching next", len(download_batch), "jobs")

        jobmon.run_parallel(
            runner="cdtask",
            task_name="extract era5_download",
            flat_node_args=(
                ("era5-dataset", "climate-variable", "year", "month", "user"),
                download_batch,
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
