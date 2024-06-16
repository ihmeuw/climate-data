from pathlib import Path

import pandas as pd
import tqdm
import xarray as xr

from climate_downscale.data import ClimateDownscaleData
from climate_downscale.generate import utils

TRANSFORM_MAP = {
    "tas": (utils.kelvin_to_celsius, "additive"),
    "pr": (utils.precipitation_flux_to_rainfall, "multiplicative"),
}


def compute_anomaly(
    reference: xr.DataArray, target: xr.DataArray, anomaly_type: str
) -> xr.Dataset:
    if anomaly_type == "additive":
        anomaly = target.groupby("time.month") - reference
    elif anomaly_type == "multiplicative":
        anomaly = (target.groupby("time.month") + 1) / (reference + 1)
    else:
        msg = f"Unknown anomaly type: {anomaly_type}"
        raise ValueError(msg)

    anomaly = (
        anomaly.drop_vars("month")
        .rename({"lat": "latitude", "lon": "longitude", "time": "date"})
        .assign_coords(longitude=(anomaly.longitude + 180) % 360 - 180)
        .sortby("longitude")
    )
    anomaly = utils.interpolate_to_target_latlon(anomaly)
    return anomaly





def load_reference_and_target(
    path: str | Path, year: str | int
) -> tuple[xr.Dataset, xr.Dataset]:
    reference = (
        xr.open_dataset(path)
        .sel(time=utils.REFERENCE_PERIOD)
        .compute()  # Load the subset before computing the mean, otherwise it's slow
        .groupby("time.month")
        .mean("time")
    )

    time_slice = slice(f"{year}-01", f"{year}-12")
    time_range = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    target = xr.open_dataset(path).sel(time=time_slice).compute()
    target = (
        target.assign_coords(time=target.time.dt.floor("D"))
        .interp_calendar(time_range)
        .interpolate_na(dim="time", method="nearest", fill_value="extrapolate")
    )
    return reference, target


def generate_cmip6_daily_main(
    output_dir: str | Path,
    year: str | int,
    target_variable: str,
    cmip_scenario: str,
    rerefk,
) -> None:
    cd_data = ClimateDownscaleData(output_dir)
    paths = cd_data.cmip6.glob(f"{target_variable}_{cmip_scenario}*.nc")


def compute_anomaly(path, year):
    reference_period = slice("2015-01-01", "2024-12-31")

    anomaly = target.groupby("time.month") - ref
    anomaly = anomaly.rename({"lat": "latitude", "lon": "longitude"})
    anomaly = anomaly.assign_coords(
        longitude=(anomaly.longitude + 180) % 360 - 180
    ).sortby("longitude")
    anomaly = utils.interpolate_to_target_latlon(
        anomaly, target_lat=TARGET_LAT, target_lon=TARGET_LON
    )

    return anomaly


variable = "tas"
scenario = "ssp119"
year = "2024"

paths = sorted(
    list(
        Path("/mnt/share/erf/climate_downscale/extracted_data/cmip6").glob(
            "tas_ssp119*.nc"
        )
    )
)
p = paths[0]


a = 1 / len(paths) * compute_anomaly(paths[0], year)

for p in tqdm.tqdm(paths[1:]):
    a += 1 / len(paths) * compute_anomaly(p, year)
