import itertools
import typing
from pathlib import Path

import click
import pandas as pd
import xarray as xr
from rra_tools import jobmon

from climate_downscale import cli_options as clio
from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData
from climate_downscale.generate import utils

VALID_YEARS = [str(y) for y in range(max(utils.REFERENCE_YEARS) + 1, 2101)]

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
TRANSFORM_MAP = {
    "mean_temperature": (
        ["tas"],
        utils.identity,
        (273.15, 0.01),
        "additive",
    ),
    "max_temperature": (
        ["tasmax"],
        utils.identity,
        (273.15, 0.01),
        "additive",
    ),
    "min_temperature": (
        ["tasmin"],
        utils.identity,
        (273.15, 0.01),
        "additive",
    ),
    "wind_speed": (
        ["uas", "vas"],
        utils.vector_magnitude,
        (0, 0.01),
        "multiplicative",
    ),
    "relative_humidity": (
        ["hurs"],
        utils.identity,
        (0, 0.01),
        "multiplicative",
    ),
    "total_precipitation": (
        ["pr"],
        utils.identity,
        (0, 0.1),
        "multiplicative",
    ),
}


_P = typing.ParamSpec("_P")
_T = typing.TypeVar("_T")


def with_target_variable(
    *,
    allow_all: bool = False,
) -> clio.ClickOption[_P, _T]:
    return clio.with_choice(
        "target-variable",
        "t",
        allow_all=allow_all,
        choices=list(TRANSFORM_MAP.keys()),
        help="Variable to generate.",
    )


def load_and_shift_longitude(
    ds_path: str | Path,
    time_slice: slice,
) -> xr.Dataset:
    ds = xr.open_dataset(ds_path).sel(time=time_slice).compute()
    ds = (
        ds.rename({"lat": "latitude", "lon": "longitude", "time": "date"})
        .assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
        .sortby("longitude")
    )
    return ds


def load_variable(
    member_path: str | Path,
    variable: str,
    year: str | int,
) -> xr.Dataset:
    if year == "reference":
        ds = load_and_shift_longitude(member_path, utils.REFERENCE_PERIOD)
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
        anomaly = (target.groupby("time.month") + 1) / (reference + 1)  # type: ignore[operator]
    else:
        msg = f"Unknown anomaly type: {anomaly_type}"
        raise ValueError(msg)

    anomaly = (
        anomaly.drop_vars("month")
        .rename({"lat": "latitude", "lon": "longitude", "time": "date"})
        .assign_coords(longitude=(anomaly.longitude + 180) % 360 - 180)
        .sortby("longitude")
    )
    return anomaly


def generate_scenario_daily_main(
    output_dir: str | Path,
    year: str | int,
    target_variable: str,
    cmip6_experiment: str,
) -> None:
    cd_data = ClimateDownscaleData(output_dir)

    (source_variables, transform_fun, (e_offset, e_scale), anomaly_type) = (
        TRANSFORM_MAP[target_variable]
    )

    paths_by_var = [
        list(cd_data.extracted_cmip6.glob(f"{source_variable}_{cmip6_experiment}*.nc"))
        for source_variable in source_variables
    ]
    source_paths = list(zip(*paths_by_var, strict=True))

    print("loading historical reference")
    historical_reference = cd_data.load_daily_results(
        scenario="historical",
        variable=target_variable,
        year="reference",
    )

    print("Making memory buffer")
    scale = 1 / len(source_paths)
    anomaly = xr.zeros_like(historical_reference)
    for i, sps in enumerate(source_paths):
        pid = f"{i}/{len(source_paths)}"
        print(f"{pid}: Loading reference")
        scenario_reference = transform_fun(  # type: ignore[operator]
            *[load_variable(sp, target_variable, "reference") for sp in sps]
        )
        print(f"{pid}: Loading target")
        target = transform_fun(  # type: ignore[operator]
            *[load_variable(sp, target_variable, year) for sp in sps]
        )
        print(f"{pid}: computing anomaly")
        s_anomaly = scale * compute_anomaly(scenario_reference, target, anomaly_type)
        print(f"{pid}: downscaling anomaly")
        anomaly += utils.interpolate_to_target_latlon(s_anomaly, method="linear")

    print("Computing scenario data")
    if anomaly_type == "additive":
        scenario_data = historical_reference + anomaly
    else:
        scenario_data = historical_reference * anomaly

    print("Saving")
    cd_data.save_daily_results(
        scenario_data,
        scenario=cmip6_experiment,
        variable=target_variable,
        year=year,
        encoding_kwargs={
            "add_offset": e_offset,
            "scale_factor": e_scale,
        },
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_year(years=VALID_YEARS)
@with_target_variable()
@clio.with_cmip6_experiment()
def generate_scenario_daily_task(
    output_dir: str, year: str, target_variable: str, cmip6_experiment: str
) -> None:
    generate_scenario_daily_main(output_dir, year, target_variable, cmip6_experiment)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_year(years=VALID_YEARS, allow_all=True)
@with_target_variable(allow_all=True)
@clio.with_cmip6_experiment(allow_all=True)
@clio.with_queue()
@clio.with_overwrite()
def generate_scenario_daily(
    output_dir: str,
    year: str,
    target_variable: str,
    cmip6_experiment: str,
    queue: str,
    overwrite: bool,  # noqa: FBT001
) -> None:
    cd_data = ClimateDownscaleData(output_dir)

    years = VALID_YEARS if year == clio.RUN_ALL else [year]
    variables = (
        list(TRANSFORM_MAP.keys())
        if target_variable == clio.RUN_ALL
        else [target_variable]
    )
    experiments = (
        list(clio.VALID_CMIP6_EXPERIMENTS)
        if cmip6_experiment == clio.RUN_ALL
        else [cmip6_experiment]
    )

    yve = []
    complete = []
    for y, v, e in itertools.product(years, variables, experiments):
        path = cd_data.daily_results_path(y, v, e)
        if not path.exists() or overwrite:
            yve.append((y, v, e))
        else:
            complete.append((y, v, e))

    print(f"{len(complete)} tasks already done. " f"Launching {len(yve)} tasks")

    jobmon.run_parallel(
        runner="cdtask",
        task_name="generate scenario_daily",
        flat_node_args=(
            ("year", "target-variable", "cmip-experiment"),
            yve,
        ),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 5,
            "memory": "200G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
    )
