from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click
import pandas as pd
from rra_population_pipelines.pipelines.climate import data
from rra_population_pipelines.shared.cli_tools import options as clio
from rra_population_pipelines.shared.data import RRA_POP
from rra_tools import jobmon

if TYPE_CHECKING:
    import xarray as xr

_ENSEMBLE_MEMBERS = [
    ("NCAR", "CESM2"),
    ("MOHC", "UKESM1-0-LL"),
    ("IPSL", "IPSL-CM6A-LR"),
    ("MPI-M", "MPI-ESM1-2-LR"),
    ("MIROC", "MIROC6"),
    ("NOAA-GFDL", "GFDL-ESM4"),
]

_VALID_YEARS = tuple([str(y) for y in range(2015, 2101)])


def compute_common_lat_lon(
    run_metadata: pd.DataFrame,
) -> tuple[pd.Index[float], pd.Index[float]]:
    lat = pd.Index([], name="lat", dtype=float)
    lon = pd.Index([], name="lon", dtype=float)

    for key in run_metadata.index.tolist():
        historical = data.load_cmip_historical_data(run_metadata.loc[key, "historical"])
        lat = lat.union(historical["lat"])  # type: ignore[arg-type]
        lon = lon.union(historical["lon"])  # type: ignore[arg-type]
    return lat, lon


def compute_single_model_anomaly(
    historical: xr.Dataset,
    experiment: xr.Dataset,
    variable: str,
) -> xr.Dataset:
    if variable == "tas":
        anomaly = experiment.groupby("time.month") - historical
    else:
        historical = 86400 * historical + 1
        experiment = 86400 * experiment + 1
        anomaly = (1 / historical) * experiment.groupby("time.month")
    return anomaly


def interp_common_lat_lon(
    ds: xr.Dataset, lat: pd.Index[float], lon: pd.Index[float]
) -> xr.Dataset:
    return (
        ds.pad(lon=1, mode="wrap")
        .assign_coords(lon=ds.lon.pad(lon=1, mode="reflect", reflect_type="odd"))
        .interp(lat=lat, lon=lon)
    )


def project_anomaly_main(variable: str, experiment: str, year: str) -> xr.Dataset:
    run_meta = get_run_metadata(variable, experiment)
    lat, lon = compute_common_lat_lon(run_meta)

    anomalies: list[xr.Dataset] = []
    for key in run_meta.index.tolist():
        historical = data.load_cmip_historical_data(run_meta.loc[key, "historical"])
        scenario = data.load_cmip_experiment_data(
            run_meta.loc[key, "experiment"], year=year
        )
        anomaly = compute_single_model_anomaly(historical, scenario, variable=variable)
        anomaly = interp_common_lat_lon(anomaly, lat, lon)
        anomalies.append(anomaly)

    mean_anomaly = 1 / len(anomalies) * sum(anomalies)
    return mean_anomaly  # type: ignore[return-value]


@click.command()  # type: ignore[arg-type]
@click.option(
    "--variable",
    type=click.Choice(["tas", "pr"]),
)
@clio.with_climate_scenario(allow_all=False)
@clio.with_year(allow_all=False, choices=_VALID_YEARS)
@clio.with_output_directory(RRA_POP.projected_climate_anomaly_data)
def project_anomaly_task(
    variable: str,
    climate_scenario: str,
    year: str,
    output_dir: str,
) -> None:
    projected_anomaly = project_anomaly_main(variable, climate_scenario, year)
    out_path = Path(output_dir) / "{variable}_{experiment}_{year}.nc"
    projected_anomaly.to_netcdf(out_path)


@click.command()  # type:  ignore[arg-type]
@clio.with_output_directory(RRA_POP.projected_climate_anomaly_data)
@clio.with_queue()
def project_anomaly(output_dir: str, queue: str) -> None:
    jobmon.run_parallel(
        task_name="project_anomaly",
        node_args={
            "variable": [
                "tas",
                "pr",
            ],
            "experiment": list(clio.VALID_CLIMATE_SCENARIOS),
            "year": list(_VALID_YEARS),
        },
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 2,
            "memory": "70G",
            "runtime": "120m",
            "project": "proj_rapidresponse",
        },
        runner="rptask",
    )
