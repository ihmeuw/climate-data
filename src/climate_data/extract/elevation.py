from pathlib import Path

import click
import requests
import tqdm
from rra_tools import jobmon

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData

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
    model_name: str,
    lat_start: int,
    lon_start: int,
    output_dir: str | Path,
) -> None:
    cdata = ClimateData(output_dir)
    cred_path = cdata.credentials_root / "open_topography.txt"
    key = cred_path.read_text().strip()

    params: dict[str, int | str] = {
        "demtype": model_name,
        "south": lat_start,
        "north": lat_start + FETCH_SIZE,
        "west": lon_start,
        "east": lon_start + FETCH_SIZE,
        "ext": "tif",
        "API_Key": key,
    }

    response = requests.get(API_ENDPOINT, params=params, stream=True, timeout=30)
    response.raise_for_status()

    out_path = (
        cdata.open_topography_elevation / f"{model_name}_{lat_start}_{lon_start}.tif"
    )
    with out_path.open("wb") as fp:
        for chunk in tqdm.tqdm(response.iter_content(chunk_size=64 * 1024**2)):
            fp.write(chunk)


@click.command()
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
    required=True,
    type=int,
    help="Longitude of the top-left corner of the tile.",
)
@clio.with_output_directory(cdc.MODEL_ROOT)
def extract_elevation_task(
    model_name: str,
    lat_start: int,
    lon_start: int,
    output_dir: str,
) -> None:
    """Download elevation data from Open Topography."""
    invalid = True
    if invalid:
        msg = "Downloaded using aws cli, this implementation is not valid"
        raise NotImplementedError(msg)

    extract_elevation_main(model_name, lat_start, lon_start, output_dir)


@click.command()
@click.option(
    "--generate-name",
    required=True,
    type=click.Choice(ELEVATION_MODELS),
    help="Name of the elevation model to download.",
)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_queue()
def extract_elevation(
    model_name: str,
    output_dir: str,
    queue: str,
) -> None:
    """Download elevation data from Open Topography."""
    invalid = True
    if invalid:
        msg = "Downloaded using aws cli, this implementation is not valid"
        raise NotImplementedError(msg)

    lat_starts = list(range(-90, 90, FETCH_SIZE))
    lon_starts = list(range(-180, 180, FETCH_SIZE))

    jobmon.run_parallel(
        runner="cdtask",
        task_name="extract elevation",
        node_args={
            "model-name": [model_name],
            "lat-start": lat_starts,
            "lon-start": lon_starts,
        },
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
    )
