from collections.abc import Sequence
from pathlib import Path

import click
import numpy as np
import rasterra as rt
from rra_tools import jobmon

from climate_data import cli_options as clio
from climate_data.data import DEFAULT_ROOT, ClimateDownscaleData
from climate_data.utils import make_raster_template

PAD = 1
STRIDE = clio.STRIDE


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
    if paths:
        raster = rt.load_mf_raster(paths)
    else:
        template = make_raster_template(
            x_min=longitudes[0],
            y_min=latitudes[0],
            stride=STRIDE,
            resolution=0.1,
        )
        no_data = -32768
        arr = np.full((len(latitudes), len(longitudes)), no_data, dtype=np.int16)
        raster = rt.RasterArray(
            data=arr,
            transform=template.transform,
            crs=template.crs,
            no_data_value=-32768,
        )
    return raster


def load_lcz_data(
    cd_data: ClimateDownscaleData, latitudes: Sequence[int], longitudes: Sequence[int]
) -> rt.RasterArray:
    path = cd_data.rub_local_climate_zones / "lcz_filter_v3.tif"
    bounds = (longitudes[0], latitudes[0], longitudes[-1], latitudes[-1])
    return rt.load_raster(path, bounds=bounds)


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
    lcz = load_lcz_data(cd_data, latitudes, longitudes)

    predictors["elevation_target"] = elevation.resample_to(
        template_target, resampling="average"
    )
    predictors["elevation_era5"] = elevation.resample_to(
        template_era5, resampling="average"
    ).resample_to(template_target, resampling="nearest")
    predictors["elevation_anomaly"] = (
        predictors["elevation_era5"] - predictors["elevation_target"]
    )
    predictors["lcz_era5"] = lcz.resample_to(template_era5, resampling="mode")
    predictors["lcz_target"] = lcz.resample_to(template_target, resampling="mode")

    for name, predictor in predictors.items():
        cd_data.save_predictor(predictor, name, lat_start, lon_start)


@click.command()  # type: ignore[arg-type]
@clio.with_lat_start(allow_all=False)
@clio.with_lon_start(allow_all=False)
@clio.with_output_directory(DEFAULT_ROOT)
def prepare_predictors_task(
    lat_start: str,
    lon_start: str,
    output_dir: str,
) -> None:
    prepare_predictors_main(output_dir, int(lat_start), int(lon_start))


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_queue()
def prepare_predictors(output_dir: str, queue: str) -> None:
    jobmon.run_parallel(
        runner="cdtask",
        task_name="downscale prepare_predictors",
        node_args={
            "lat-start": clio.LATITUDES,
            "lon-start": clio.LONGITUDES,
        },
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "45m",
            "project": "proj_rapidresponse",
        },
    )
