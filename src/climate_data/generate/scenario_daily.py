import itertools
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr
from rra_tools import jobmon

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData
from climate_data.generate import utils

# Map from source variable to a unit conversion function
CONVERT_MAP = {
    "uas": utils.scale_wind_speed_height,
    "vas": utils.scale_wind_speed_height,
    "hurs": utils.identity,
    "tas": utils.kelvin_to_celsius,
    "tasmin": utils.kelvin_to_celsius,
    "tasmax": utils.kelvin_to_celsius,
    "pr": utils.precipitation_flux_to_rainfall,
}

# Map from target variable to:
#  - a list of source variables
#  - a transformation function
#  - a tuple of offset and scale factors for the output for serialization
#  - an anomaly type
TRANSFORM_MAP: dict[str, tuple[utils.Transform, str]] = {
    "mean_temperature": (
        utils.Transform(
            source_variables=["tas"],
            transform_funcs=[utils.identity],
            encoding_scale=0.01,
        ),
        "additive",
    ),
    "max_temperature": (
        utils.Transform(
            source_variables=["tasmax"],
            transform_funcs=[utils.identity],
            encoding_scale=0.01,
        ),
        "additive",
    ),
    "min_temperature": (
        utils.Transform(
            source_variables=["tasmin"],
            transform_funcs=[utils.identity],
            encoding_scale=0.01,
        ),
        "additive",
    ),
    "wind_speed": (
        utils.Transform(
            source_variables=["uas", "vas"],
            transform_funcs=[utils.vector_magnitude],
            encoding_scale=0.01,
        ),
        "multiplicative",
    ),
    "relative_humidity": (
        utils.Transform(
            source_variables=["hurs"],
            transform_funcs=[utils.identity],
            encoding_scale=0.01,
        ),
        "multiplicative",
    ),
    "total_precipitation": (
        utils.Transform(
            source_variables=["pr"],
            transform_funcs=[utils.identity],
            encoding_scale=0.1,
        ),
        "multiplicative",
    ),
}


def load_and_shift_longitude(
    member_path: str | Path,
    time_slice: slice,
) -> xr.Dataset:
    ds = xr.open_dataset(member_path).sortby("time").sel(time=time_slice).compute()
    if ds.time.size == 0:
        msg = "No data in slice"
        raise KeyError(msg)
    ds = (
        ds.assign_coords(lon=(ds.lon + 180) % 360 - 180)
        .sortby("lon")
        .rename({"lat": "latitude", "lon": "longitude"})
    )
    return ds


def load_and_shift_longitude_and_correct_time(
    member_path: str | Path,
    year: str,
) -> xr.Dataset:
    time_slice = slice(f"{year}-01-01", f"{year}-12-31")
    time_range = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    ds = load_and_shift_longitude(member_path, time_slice)
    ds = (
        ds.assign_coords(time=ds.time.dt.floor("D"))
        .interp_calendar(time_range)
        .interpolate_na(dim="time", method="nearest", fill_value="extrapolate")
        .rename({"time": "date"})
    )
    return ds


def load_variable(
    member_path: str | Path,
    year: str | int,
) -> xr.Dataset:
    if year == "reference":
        ds = load_and_shift_longitude(member_path, cdc.REFERENCE_PERIOD).rename(
            {"time": "date"}
        )
    else:
        try:
            ds = load_and_shift_longitude_and_correct_time(member_path, str(year))
        except KeyError as e:
            if int(year) == 2100:  # noqa: PLR2004
                # Some datasets stop in 2099.  Just reuse the last year
                ds = load_and_shift_longitude_and_correct_time(member_path, "2099")
                ds = ds.assign_coords(date=ds.date + np.timedelta64(ds.date.size, "D"))
            else:
                raise e

    variable = str(next(iter(ds)))
    conversion = CONVERT_MAP[variable]
    ds = conversion(utils.rename_val_column(ds))
    return ds


def compute_anomaly(
    reference: xr.Dataset, target: xr.Dataset, anomaly_type: str
) -> xr.Dataset:
    reference = reference.groupby("date.month").mean("date")
    if anomaly_type == "additive":
        anomaly = target.groupby("date.month") - reference
    elif anomaly_type == "multiplicative":
        anomaly = (target + 1).groupby("date.month") / (reference + 1)
    else:
        msg = f"Unknown anomaly type: {anomaly_type}"
        raise ValueError(msg)
    anomaly = anomaly.drop_vars("month")
    return anomaly


def generate_scenario_daily_main(
    target_variable: str,
    cmip6_experiment: str,
    year: str | int,
    gcm_member: str,
    output_dir: str | Path,
    write_output: bool = True,
) -> xr.Dataset:
    cdata = ClimateData(output_dir)

    transform, anomaly_type = TRANSFORM_MAP[target_variable]
    source_paths = [
        cdata.extracted_cmip6_path(source_variable, cmip6_experiment, gcm_member)
        for source_variable in transform.source_variables
    ]

    print("loading historical reference")
    historical_reference = cdata.load_daily_results(
        scenario="historical",
        variable=target_variable,
        year="reference",
    )
    # compute anomaly, resample anomaly and compute scenario data
    # load reference (monthly) and target (daily for a given year)
    print(f"{gcm_member}: Loading reference")
    sref = transform(*[load_variable(vp, "reference") for vp in source_paths])

    print(f"{gcm_member}: Loading target")
    target = transform(*[load_variable(vp, year) for vp in source_paths])

    print(f"{gcm_member}: computing anomaly")
    v_anomaly = compute_anomaly(sref, target, anomaly_type)

    print(f"{gcm_member}: resampling anomaly")
    resampled_anomaly = utils.interpolate_to_target_latlon(v_anomaly, method="linear")

    print(f"{gcm_member}: computing scenario data")
    if anomaly_type == "additive":
        scenario_data = historical_reference + resampled_anomaly.groupby("date.month")
    else:
        scenario_data = historical_reference * resampled_anomaly.groupby("date.month")
    if write_output is True:
        print(f"{gcm_member}: Writing output")
        cdata.save_raw_daily_results(
            scenario_data,
            scenario=cmip6_experiment,
            variable=target_variable,
            year=year,
            gcm_member=gcm_member,
            encoding_kwargs=transform.encoding_kwargs,
        )
    else:
        print(f"{gcm_member}: Returning output")

    return scenario_data


@click.command()  # type: ignore[arg-type]
@clio.with_target_variable(list(TRANSFORM_MAP))
@clio.with_cmip6_experiment()
@clio.with_year(cdc.FORECAST_YEARS)
@clio.with_gcm_member()
@clio.with_output_directory(cdc.MODEL_ROOT)
def generate_scenario_daily_task(
    target_variable: str,
    cmip6_experiment: str,
    year: str,
    gcm_member: str,
    output_dir: str,
) -> None:
    generate_scenario_daily_main(
        target_variable,
        cmip6_experiment,
        year,
        gcm_member,
        output_dir,
        write_output=True,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_target_variable(TRANSFORM_MAP, allow_all=True)
@clio.with_cmip6_experiment(allow_all=True)
@clio.with_year(cdc.FORECAST_YEARS, allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_queue()
@clio.with_overwrite()
def generate_scenario_daily(
    target_variable: list[str],
    cmip6_experiment: list[str],
    year: list[str],
    output_dir: str,
    queue: str,
    overwrite: bool,
) -> None:
    cdata = ClimateData(output_dir)

    veyg = []
    complete = []
    for v, e, y in itertools.product(target_variable, cmip6_experiment, year):
        source_variables = TRANSFORM_MAP[v][0].source_variables
        gcms = cdata.get_gcms(source_variables)
        for g in gcms:
            path = cdata.raw_daily_results_path(e, v, y, g)
            if not path.exists() or overwrite:
                veyg.append((g, y, v, e))
            else:
                complete.append((g, y, v, e))

    if not veyg:
        print("All tasks already done.")
        return

    print(f"{len(complete)} tasks already done. " f"Launching {len(veyg)} tasks")
    jobmon.run_parallel(
        runner="cdtask",
        task_name="generate scenario_daily",
        flat_node_args=(
            ("target-variable", "cmip6-experiment", "year", "gcm-member"),
            veyg,
        ),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "90G",
            "runtime": "400m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
    )
