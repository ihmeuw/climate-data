"""
ERA5 Uncertainty Download (CLIMATE-22 gap-fill)
-----------------------------------------------

Downloads whole-year ERA5 ``reanalysis`` and ``ensemble_spread`` 2m temperature
files from the Copernicus CDS, matching the layout Katrin Burkart produced by
hand in ``/ihme/erf/ERA5/katrin_download_20240208/`` for 2022/2023:

    era5_{product_type}_{variable}_{year}.nc   (one file per whole year)

This fills the 2024/2025 gap that GBD temperature PAF work depends on. It is a
standalone gap-fill: the ``ensemble_spread`` product is not (yet) consumed by
the rest of this repo's pipeline, which only uses reanalysis mean temperature.

Notes / gotchas baked in below:
* The ensemble products only exist on ``reanalysis-era5-single-levels`` (0.25°
  HRES / ~0.5° EDA), never on ``reanalysis-era5-land``.
* Both products are pulled 3-hourly to match the canonical GBD download and
  Katrin's 2022/2023 files. ``ensemble_spread`` (the EDA) is only 3-hourly;
  ``reanalysis`` (HRES) is subsampled from hourly to the same 3-hourly axis.
* Credentials come from the caller's ``~/.cdsapirc`` (``cdsapi.Client()`` with
  no args), so no dependency on the shared per-user ``copernicus.yaml`` keyring
  (which does not carry every user).
"""

import zipfile
from pathlib import Path

import cdsapi
import click
from rra_tools.shell_tools import mkdir, touch

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.jobmon_utils import run_parallel_maybe_dry_run

# The single-levels dataset is the only one exposing ensemble products.
ERA5_UNCERTAINTY_DATASET = cdc.ERA5_DATASETS.reanalysis_era5_single_levels

# Product types to pull for each year.
ERA5_PRODUCT_TYPES = ["reanalysis", "ensemble_spread"]

# The years this historical-extension gap-fill targets (2022/2023 already exist from
# Katrin's pull). Kept explicit because the output path below is fixed under
# ``/ihme/erf/ERA5`` and is independent of MODEL_ROOT / the ``--run-mode`` profile.
ERA5_UNCERTAINTY_YEARS = ["2024", "2025"]

# Default landing directory; override with ``--output-dir`` to a dated,
# Katrin-style subdirectory, e.g. ``/ihme/erf/ERA5/billg_download_20260701``.
DEFAULT_OUTPUT_DIR = Path("/ihme/erf/ERA5")


# Both products are downloaded 3-hourly to match the canonical GBD ERA5 download
# (ihme-internal/temperature: code_GBD2021/A_ExposureData/ERA5_download/era5_download.py)
# and Katrin Burkart's 2022/2023 files. ``ensemble_spread`` (the EDA) is only
# available at 3-hourly resolution; ``reanalysis`` (HRES) is natively hourly but is
# subsampled to 3-hourly here so both products share the same time axis, exactly as
# the historical GBD person-days inputs were produced.
ERA5_TIME_STEPS = [f"{h:02d}:00" for h in range(0, 24, 3)]


def download_era5_uncertainty_main(
    year: str,
    product_type: str,
    variable: str,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    mkdir(output_dir, parents=True, exist_ok=True)

    # Katrin's exact naming: one file per whole year.
    out_path = output_dir / f"era5_{product_type}_{variable}_{year}.nc"
    if out_path.exists() and out_path.stat().st_size > 0:
        print("Already downloaded:", out_path)
        return

    # New CDS may wrap netcdf in a zip; download to a temp path and finalize.
    download_path = out_path.with_suffix(".download")

    try:
        touch(download_path, clobber=True)

        print("Connecting to copernicus")
        client = cdsapi.Client()  # reads ~/.cdsapirc

        request = {
            "product_type": [product_type],
            "variable": [variable],
            "year": [year],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ERA5_TIME_STEPS,
            "data_format": "netcdf",
        }

        print(f"Downloading {product_type} {variable} {year}...")
        client.retrieve(ERA5_UNCERTAINTY_DATASET, request).download(str(download_path))

        _finalize_download(download_path, out_path)
    except Exception:
        print(f"Failed to download {product_type} {variable} {year}")
        # Remove any partial artifacts so a later run doesn't skip this as "already
        # downloaded" (see the size>0 check above). out_path is only ever created by
        # the atomic replace in _finalize_download, so removing it here is safe.
        for tmp in (download_path, out_path):
            if tmp.exists():
                tmp.unlink()
        raise


def _finalize_download(download_path: Path, out_path: Path) -> None:
    """Move the downloaded payload into place, unwrapping a zip if present.

    ``out_path`` is created only by a final atomic ``replace``, so a failure part-way
    through (corrupt member, disk full, killed job) never leaves a truncated file that
    a later run would mistake for a completed download.
    """
    if not zipfile.is_zipfile(download_path):
        download_path.replace(out_path)
        return

    print("Unzipping...")
    unzipped = download_path.with_suffix(".unzipped")
    try:
        with zipfile.ZipFile(download_path) as zf:
            members = zf.infolist()
            if len(members) != 1:
                msg = f"Expected a single file in {download_path}, got {len(members)}"
                raise ValueError(msg)
            with unzipped.open("wb") as f:
                f.write(zf.read(members[0]))
        unzipped.replace(out_path)
    finally:
        if unzipped.exists():
            unzipped.unlink()
    download_path.unlink()


@click.command()
@clio.with_year(years=cdc.HISTORY_YEARS)
@click.option(
    "--product-type",
    required=True,
    type=click.Choice(ERA5_PRODUCT_TYPES),
    help="ERA5 product type to download.",
)
@clio.with_era5_variable()
@clio.with_output_directory(DEFAULT_OUTPUT_DIR)
def download_era5_uncertainty_task(
    year: str,
    product_type: str,
    era5_variable: str,
    output_dir: str,
) -> None:
    """Download one whole-year ERA5 file (single year, single product type)."""
    download_era5_uncertainty_main(year, product_type, era5_variable, output_dir)


@click.command()
@clio.with_era5_variable()
@clio.with_output_directory(DEFAULT_OUTPUT_DIR)
@clio.with_queue()
@clio.with_dry_run()
def download_era5_uncertainty(
    era5_variable: str,
    output_dir: str,
    queue: str,
    dry_run: bool,
) -> None:
    """Download 2024/2025 ERA5 reanalysis + ensemble_spread (CLIMATE-22 gap-fill)."""
    node_args = {
        "year": ERA5_UNCERTAINTY_YEARS,
        "product-type": ERA5_PRODUCT_TYPES,
    }

    run_parallel_maybe_dry_run(
        runner="cdtask special",
        task_name="download_era5_uncertainty",
        node_args=node_args,
        task_args={
            "era5-variable": era5_variable,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "20G",
            "runtime": "720m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        dry_run=dry_run,
    )
