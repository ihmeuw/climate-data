from pathlib import Path

import cdsapi
import click
from rra_tools import jobmon
from rra_tools.shell_tools import touch
import xarray as xr

from climate_downscale import cli_options as clio
from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData


def extract_era5_main(
    output_dir: str | Path,
    era5_dataset: str,
    climate_variable: str,
    year: int | str,
    month: str,
) -> None:
    cddata = ClimateDownscaleData(output_dir)
    cred_path = cddata.credentials_root / "copernicus.txt"
    url, key = cred_path.read_text().strip().split("\n")
    
    out_path = cddata.era5_path(era5_dataset, climate_variable, year, month)
    raw_out_path = out_path.with_stem(f"{out_path.stem}_raw")
    
    if out_path.exists():
        if raw_out_path.exists():
            # We ran into an error before completing compression, likely a
            # memory error. Delete and retry.
            out_path.unlink()
        else:
            print("Already extracted:", out_path)
            return
    
    try:
        if not raw_out_path.exists():
            return
            touch(raw_out_path)

            print('Connecting to copernicus')
            copernicus = cdsapi.Client(url=url, key=key)
            kwargs = {
                "product_type": "reanalysis",
                "variable": climate_variable,
                "year": year,
                "month": month,
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "format": "netcdf",
            }
            print("Downloading...")
            result = copernicus.retrieve(
                era5_dataset,
                kwargs,
            )
            result.download(raw_out_path)
        else:
            print("Already downloaded:", raw_out_path)
    except Exception as e:
        print(f"Failed to download {era5_dataset} {climate_variable} {year} {month}")
        if raw_out_path.exists():
            raw_out_path.unlink()        
        raise e  # noqa: TRY201

    touch(out_path)
    try:
        print("Compressing...")
        ds = xr.open_dataset(raw_out_path)
        var_name = list(ds)[0]  # These are all single variable datasets    
        og_encoding = ds[var_name].encoding
        ds.to_netcdf(
            out_path,
            encoding={
                var_name:{
                    **og_encoding,
                    "zlib": True,
                    "complevel": 1,
                }
            }
        )
        
    except Exception as e:
        print(f'Failed to compress {era5_dataset} {climate_variable} {year} {month}')
        if out_path.exists():
            out_path.unlink()
        raise e  # noqa: TRY201

    raw_out_path.unlink()

@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_era5_dataset()
@clio.with_climate_variable()
@clio.with_year()
@clio.with_month()
def extract_era5_task(
    output_dir: str,
    era5_dataset: str,
    climate_variable: str,
    year: str,
    month: str,
) -> None:
    extract_era5_main(
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

    jobmon.run_parallel(
        runner="cdtask",
        task_name="extract era5",
        node_args={
            "era5-dataset": datasets,
            "climate-variable": variables,
            "year": years,
            "month": months,
        },
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "120G",
            "runtime": "600m",
            "project": "proj_rapidresponse",
        },
    )
