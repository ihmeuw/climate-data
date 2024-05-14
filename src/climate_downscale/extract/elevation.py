from pathlib import Path

import click
import requests
from rra_tools import jobmon
from rra_tools.cli_tools import (
    with_output_directory,
    with_queue,
)

from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData

API_ENDPOINT = "https://portal.opentopography.org/API/globaldem"

ELEVATION_MODELS = [
    "SRTMGL3",  # SRTM Global 3 arc second (90m)
    "SRTMGL1",  # SRTM Global 1 arc second (30m)
    "SRTMGL1_E",  # SRTM Global 1 arc second ellipsoidal height (30m)
    "AW3D30",  # ALOS World 3D 30m
    "AW3D30_E",  # ALOS World 3D 30m ellipsoidal height
    "SRTM15Plus",  # SRTM 15 arc second (500m)
    "NASADEM",  # NASA DEM 1 arc second (30m)
    "COP30",  # Copernicus 1 arc second (30m)
    "COP90",  # Copernicus 3 arc second (90m)
]

FETCH_SIZE = 5  # degrees, should be small enough for any model


def extract_elevation_main(
    output_dir: str | Path,
    model_name: str,
    lat_start: int,
    lon_start: int,
) -> None:
    cd_data = ClimateDownscaleData(output_dir)
    cred_path = cd_data.credentials_root / "open_topography.txt"
    key = cred_path.read_text().strip()

    params: dict[str, int | str] = {
        "dem_type": model_name,
        "south": lat_start,
        "north": lat_start + FETCH_SIZE,
        "west": lon_start,
        "east": lon_start + FETCH_SIZE,
        "ext": "tif",
        "API_Key": key,
    }

    response = requests.get(API_ENDPOINT, params=params, stream=True, timeout=10)
    response.raise_for_status()

    out_path = cd_data.elevation / f"{model_name}_{lat_start}_{lon_start}.tif"
    with out_path.open("wb") as fp:
        for chunk in response.iter_content(chunk_size=None):
            fp.write(chunk)


@click.command()  # type: ignore[arg-type]
@with_output_directory(DEFAULT_ROOT)
@click.option(
    "--model-name",
    required=True,
    type=click.Choice(ELEVATION_MODELS),
    help="Name of the elevation model to download.",
)
@click.option(
    "--lat-start",
    required=True,
    type=int,
    help="Latitude of the top-left corner of the tile.",
)
@click.option(
    "--lon-start",
    required=int,
    type=float,
    help="Longitude of the top-left corner of the tile.",
)
def extract_elevation_task(
    output_dir: str,
    model_name: str,
    lat_start: int,
    lon_start: int,
) -> None:
    """Download elevation data from Open Topography."""
    extract_elevation_main(output_dir, model_name, lat_start, lon_start)


@click.command()  # type: ignore[arg-type]
@with_output_directory(DEFAULT_ROOT)
@click.option(
    "--model-name",
    required=True,
    type=click.Choice(ELEVATION_MODELS),
    help="Name of the elevation model to download.",
)
@with_queue()
def extract_elevation(
    output_dir: str,
    model_name: str,
    queue: str,
) -> None:
    """Download elevation data from Open Topography."""
    lat_starts = list(range(-90, 90, FETCH_SIZE))
    lon_starts = list(range(-180, 180, FETCH_SIZE))

    jobmon.run_parallel(
        task_name="extract_era5",
        node_args={
            "output-dir": [output_dir],
            "model-name": [model_name],
            "lat-start": lat_starts,
            "lon-start": lon_starts,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        runner="cdtask",
    )
