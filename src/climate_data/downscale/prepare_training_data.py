from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from rra_tools import jobmon

from climate_data import cli_options as clio
from climate_data.data import DEFAULT_ROOT, ClimateDownscaleData


def load_and_clean_climate_stations(
    cd_data: ClimateDownscaleData,
    year: int | str,
) -> pd.DataFrame:
    climate_stations = cd_data.load_ncei_climate_stations(year)
    column_map = {
        "DATE": "date",
        "LATITUDE": "lat",
        "LONGITUDE": "lon",
        "TEMP": "temperature",
        "ELEVATION": "ncei_elevation",
        "STATION": "station_id",
    }
    climate_stations = (
        climate_stations.rename(columns=column_map)
        .loc[:, list(column_map.values())]
        .dropna()
        .reset_index(drop=True)
        .assign(
            date=lambda df: pd.to_datetime(df["date"]),
            year=lambda df: df["date"].dt.year,
            dayofyear=lambda df: df["date"].dt.dayofyear,
            temperature=lambda df: 5 / 9 * (df["temperature"] - 32),
        )
    )
    return climate_stations


def get_era5_temperature(
    cd_data: ClimateDownscaleData,
    year: int | str,
    coords: dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    lat = xr.DataArray(coords["lat"], dims=["points"])
    lon = xr.DataArray(coords["lon"], dims=["points"])
    time = xr.DataArray(coords["date"], dims=["points"])

    era5 = cd_data.load_daily_results("historical", "tas", year)
    era5 = (
        era5.assign_coords(longitude=(((era5.longitude + 180) % 360) - 180))
        .sortby(["latitude", "longitude"])
        .sel(latitude=lat, longitude=lon, time=time, method="nearest")
    )

    if "expver" in era5.coords:
        # expver == 1 is final data.  expver == 5 is provisional data
        # and has a very strong nonsense seasonal trend.
        era5 = era5.sel(expver=1)
    return era5["value"].to_numpy()


def prepare_training_data_main(output_dir: str | Path, year: str) -> None:
    cd_data = ClimateDownscaleData(output_dir)

    data = load_and_clean_climate_stations(cd_data, year)
    coords = {
        "lon": data["lon"].to_numpy(),
        "lat": data["lat"].to_numpy(),
        "date": data["date"].to_numpy(),
    }

    data["era5_temperature"] = get_era5_temperature(cd_data, year, coords)

    # Elevation pieces
    data["target_elevation"] = cd_data.load_predictor("elevation_target").select(
        coords["lon"], coords["lat"]
    )
    data["era5_elevation"] = cd_data.load_predictor("elevation_era5").select(
        coords["lon"], coords["lat"]
    )

    data["elevation"] = data["ncei_elevation"]
    nodata_val = -999
    missing_elevation = data["elevation"] < nodata_val
    data.loc[missing_elevation, "elevation"] = data.loc[
        missing_elevation, "target_elevation"
    ]

    # Local climate zone
    data["target_lcz"] = cd_data.load_predictor("lcz_target").select(
        coords["lon"], coords["lat"]
    )
    data["era5_lcz"] = cd_data.load_predictor("lcz_era5").select(
        coords["lon"], coords["lat"]
    )

    cd_data.save_training_data(data, year)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_year(years=clio.VALID_HISTORY_YEARS)
def prepare_training_data_task(output_dir: str, year: str) -> None:
    prepare_training_data_main(output_dir, year)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_queue()
def prepare_training_data(output_dir: str, queue: str) -> None:
    jobmon.run_parallel(
        runner="cdtask",
        task_name="downscale prepare_training_data",
        node_args={
            "year": clio.VALID_HISTORY_YEARS,
        },
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "30G",
            "runtime": "30m",
            "project": "proj_rapidresponse",
        },
    )
