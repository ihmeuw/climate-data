from collections.abc import Sequence
from pathlib import Path

import click
import rasterra as rt
from rra_tools import jobmon
from rra_tools.cli_tools import (
    with_choice,
    with_output_directory,
    with_queue,
)

from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData
from climate_downscale.utils import make_raster_template

# Degrees

STRIDE = 30
LATITUDES = [str(lat) for lat in range(-90, 90, STRIDE)]
LONGITUDES = [str(lon) for lon in range(-180, 180, STRIDE)]
PAD = 1


def load_elevation(
    cd_data: ClimateDownscaleData,
    latitudes: Sequence[int],
    longitudes: Sequence[int],
) -> rt.RasterArray:
    data_root = cd_data.open_topography_elevation / "SRTM_GL3_srtm"
    paths = []
    for lon in longitudes:
        lon_stub = f"E{lon:03}" if lon >= 0 else f"W{-lon:03}"
        for lat in latitudes:
            if lat >= 30:  # noqa: PLR2004
                rel_path = f"North/North_30_60/N{lat:02}{lon_stub}.tif"
            elif lat >= 0:
                rel_path = f"North/North_0_29/N{lat:02}{lon_stub}.tif"
            else:
                rel_path = f"South/S{-lat:02}{lon_stub}.tif"

            p = data_root / rel_path

            if p.exists():
                paths.append(p)

    return rt.load_mf_raster(paths)


def prepare_predictors_main(
    output_dir: str | Path, lat_start: int, lon_start: int
) -> None:
    cd_data = ClimateDownscaleData(output_dir)
    predictors = {}

    longitudes = range(lon_start - PAD, lon_start + STRIDE + PAD)
    latitudes = range(lat_start - PAD, lat_start + STRIDE + PAD)

    # Make upscale templates, one at ERA5 resolution and one at the target
    # resolution for the predictors
    template_era5 = make_raster_template(
        x_min=lon_start,
        y_min=lat_start,
        stride=STRIDE,
        resolution=0.1,
    )
    template_target = make_raster_template(
        x_min=lon_start,
        y_min=lat_start,
        stride=STRIDE,
        resolution=0.01,
    )

    elevation = load_elevation(cd_data, latitudes, longitudes)

    predictors["elevation_target"] = elevation.resample_to(
        template_target, resampling="average"
    )
    predictors["elevation_era5"] = elevation.resample_to(
        template_era5, resampling="average"
    ).resample_to(template_target, resampling="nearest")
    predictors["elevation_anomaly"] = (
        predictors["elevation_era5"] - predictors["elevation_target"]
    )

    for name, predictor in predictors.items():
        cd_data.save_predictor(predictor, f"{name}_{lat_start}_{lon_start}")


@click.command()  # type: ignore[arg-type]
@with_choice("lat_start", allow_all=True, choices=LATITUDES)
@with_choice("lon_start", allow_all=True, choices=LONGITUDES)
@with_output_directory(DEFAULT_ROOT)
def prepare_predictors_task(
    lat_start: int, lon_start: int, output_dir: str | Path
) -> None:
    prepare_predictors_main(output_dir, lat_start, lon_start)


@click.command()  # type: ignore[arg-type]
@with_output_directory(DEFAULT_ROOT)
@with_queue()
def prepare_predictors(output_dir: str, queue: str) -> None:
    jobmon.run_parallel(
        "model prepare_predictors",
        node_args={
            "output-dir": [output_dir],
            "lat_start": LATITUDES,
            "lon_start": LONGITUDES,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "45m",
            "project": "proj_rapidresponse",
        },
        runner="cdtask",
    )
