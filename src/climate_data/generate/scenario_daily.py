import itertools
import random
from collections import defaultdict
from pathlib import Path

import click
import pandas as pd
import xarray as xr
from rra_tools import jobmon

from climate_data import cli_options as clio
from climate_data.data import DEFAULT_ROOT, ClimateData
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


def get_source_paths(
    cd_data: ClimateData,
    source_variables: list[str],
    cmip6_experiment: str,
) -> dict[str, list[list[Path]]]:
    inclusion_meta = cd_data.load_scenario_inclusion_metadata()[source_variables]
    inclusion_meta = inclusion_meta[inclusion_meta.all(axis=1)]
    source_paths = defaultdict(list)
    for source, variant in inclusion_meta.index.tolist():
        source_paths[source].append(
            [
                cd_data.extracted_cmip6_path(v, cmip6_experiment, source, variant)
                for v in source_variables
            ]
        )

    return source_paths


def load_and_shift_longitude(
    ds_path: str | Path,
    time_slice: slice,
) -> xr.Dataset:
    ds = xr.open_dataset(ds_path).sel(time=time_slice).compute()
    if ds.time.size == 0:
        msg = "No data in slice"
        raise KeyError(msg)
    ds = (
        ds.assign_coords(lon=(ds.lon + 180) % 360 - 180)
        .sortby("lon")
        .rename({"lat": "latitude", "lon": "longitude"})
    )
    return ds


def load_variable(
    member_path: str | Path,
    year: str | int,
) -> xr.Dataset:
    if year == "reference":
        ds = load_and_shift_longitude(member_path, utils.REFERENCE_PERIOD).rename(
            {"time": "date"}
        )
    else:
        time_slice = slice(f"{year}-01-01", f"{year}-12-31")
        time_range = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        ds = load_and_shift_longitude(member_path, time_slice)
        ds = (
            ds.assign_coords(time=ds.time.dt.floor("D"))
            .interp_calendar(time_range)
            .interpolate_na(dim="time", method="nearest", fill_value="extrapolate")
            .rename({"time": "date"})
        )
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
    output_dir: str | Path,
    year: str | int,
    draw: str | int,
    target_variable: str,
    cmip6_experiment: str,
    write_output: bool = True,
) -> xr.Dataset:
    # make repeatable
    random.seed(int(draw))
    cd_data = ClimateData(output_dir)

    transform, anomaly_type = TRANSFORM_MAP[target_variable]

    source_paths = get_source_paths(
        cd_data, transform.source_variables, cmip6_experiment
    )

    print("loading historical reference")
    historical_reference = cd_data.load_daily_results(
        scenario="historical",
        variable=target_variable,
        year="reference",
    )
    # randomly select source, variant, load reference and target,
    # compute anomaly, resample anomaly and compute scenario data
    source_key = random.choice(list(source_paths.keys()))
    variant_paths = random.choice(source_paths[source_key])
    s_variant = f"{variant_paths[0].stem.split('_')[-1]}"
    vid = f"{source_key}, Variant : {s_variant}"
    # load reference (monthly) and target (daily for a given year)
    print(f"{vid}: Loading reference")
    sref = transform(*[load_variable(vp, "reference") for vp in variant_paths])

    print(f"{vid}: Loading target")
    target = transform(*[load_variable(vp, year) for vp in variant_paths])

    print(f"{vid}: computing anomaly")
    v_anomaly = compute_anomaly(sref, target, anomaly_type)

    print(f"{vid}: resampling anomaly")
    resampled_anomaly = utils.interpolate_to_target_latlon(v_anomaly, method="linear")

    print(f"{vid}: computing scenario data")
    if anomaly_type == "additive":
        scenario_data = historical_reference + resampled_anomaly.groupby("date.month")
    else:
        scenario_data = historical_reference * resampled_anomaly.groupby("date.month")
    # write global attributes to track prevenance
    scenario_data.attrs["source"] = source_key
    scenario_data.attrs["variant"] = s_variant
    if write_output is True:
        print(f"{vid}: Writing draw {draw} from {source_key}-{s_variant}")
        cd_data.save_daily_results(
            scenario_data,
            scenario=cmip6_experiment,
            variable=target_variable,
            year=year,
            draw=draw,
            encoding_kwargs=transform.encoding_kwargs,
        )
    else:
        print(f"{vid}: returning draw {draw} from {source_key}-{s_variant}")

    return scenario_data


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_year(years=clio.VALID_FORECAST_YEARS)
@clio.with_draw(draws=clio.VALID_DRAWS, allow_all=False)
@clio.with_target_variable(variable_names=list(TRANSFORM_MAP))
@clio.with_cmip6_experiment()
def generate_scenario_daily_task(
    output_dir: str, year: str, draw: str, target_variable: str, cmip6_experiment: str
) -> None:
    generate_scenario_daily_main(
        output_dir, year, draw, target_variable, cmip6_experiment, write_output=True
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_year(years=clio.VALID_FORECAST_YEARS, allow_all=True)
@clio.with_draw(draws=clio.VALID_DRAWS, allow_all=True)
@clio.with_target_variable(variable_names=list(TRANSFORM_MAP), allow_all=True)
@clio.with_cmip6_experiment(allow_all=True)
@clio.with_queue()
@clio.with_overwrite()
def generate_scenario_daily(
    output_dir: str,
    year: str,
    draw: str,
    target_variable: str,
    cmip6_experiment: str,
    queue: str,
    overwrite: bool,
) -> None:
    cd_data = ClimateData(output_dir)

    years = clio.VALID_FORECAST_YEARS if year == clio.RUN_ALL else [year]
    draws = clio.VALID_DRAWS if draw == clio.RUN_ALL else [draw]
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
    for d, y, v, e in itertools.product(draws, years, variables, experiments):
        path = cd_data.daily_results_path(scenario=e, variable=v, year=y, draw=d)
        if not path.exists() or overwrite:
            yve.append((d, y, v, e))
        else:
            complete.append((d, y, v, e))

    print(f"{len(complete)} tasks already done. " f"Launching {len(yve)} tasks")
    jobmon.run_parallel(
        runner="cdtask",
        task_name="generate scenario_daily",
        flat_node_args=(
            ("draw", "year", "target-variable", "cmip6-experiment"),
            yve,
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
