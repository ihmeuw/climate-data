from pathlib import Path

import pandas as pd
import tqdm
import xarray as xr

from climate_downscale.data import ClimateDownscaleData
from climate_downscale.generate import utils


# Map from source variable to a unit conversion function
CONVERT_MAP = {
    "tas": utils.kelvin_to_celsius,
    "pr": utils.precipitation_flux_to_rainfall,
}


def load_and_shift_longitude(
    ds_path: str | Path,
    time_slice: slice,
) -> xr.Dataset:
    ds = xr.open_dataset(ds_path).sel(time=time_slice).compute()
    ds = (
        ds
        .rename({"lat": "latitude", "lon": "longitude", "time": "date"})
        .assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
        .sortby("longitude")
    )
    return ds


def load_variable(
    member_path: str | Path,
    variable: str,
    year: str,
) -> xr.Dataset:
    if year == "reference":
        ds = load_and_shift_longitude(member_path, utils.REFERENCE_PERIOD)
        ds = ds.groupby("date.month").mean("date")
    else:
        time_slice = slice(f"{year}-01-01", f"{year}-12-31")
        time_range = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        ds = load_and_shift_longitude(member_path, time_slice)
        ds = (
            ds.assign_coords(date=ds.date.dt.floor("D"))
            .interp(date=time_range)
            .interpolate_na(dim="date", method="nearest", fill_value="extrapolate")
        )
    conversion = CONVERT_MAP[variable]
    ds = conversion(utils.rename_val_column(ds))
    return ds

def compute_anomaly(
    reference: xr.Dataset, target: xr.Dataset, anomaly_type: str
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

def generate_scenario_daily_main(
    output_dir: str | Path,
    year: str | int,
    target_variable: str,
    cmip_scenario: str,
) -> None:
    cd_data = ClimateDownscaleData(output_dir)
    paths = cd_data.extracted_cmip6.glob(f"{target_variable}_{cmip_scenario}*.nc")

    for path in paths:
        reference = load_variable(path, target_variable, "reference")
        target = load_variable(path, target_variable, year)

        anomaly_type = TRANSFORM_MAP[target_variable][1]
        anomaly = compute_anomaly(reference, target, anomaly_type)
        cd_data.save_daily_results(
            anomaly,
            scenario=cmip_scenario,
            variable=target_variable,
            year=year,
        )

