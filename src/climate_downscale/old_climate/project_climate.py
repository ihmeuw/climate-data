import click
import pandas as pd
import xarray as xr
from rra_population_pipelines.shared.cli_tools import options as clio
from rra_population_pipelines.shared.data import (
    RRA_DATA_ROOT,
    RRA_POP,
    RRAPopulationData,
)
from rra_tools import jobmon


def get_chelsa(variable: str, lat: slice, lon: slice) -> xr.Dataset:
    ds_paths = [
        RRA_POP.get_downscaled_reference_map_path(variable, month)
        for month in range(1, 13)
    ]
    ds = (
        xr.open_mfdataset(
            ds_paths,
            chunks={"lat": -1, "lon": -1},
            concat_dim=[pd.Index(range(1, 13), name="month")],  # type: ignore[arg-type]
            combine="nested",
        )
        .sel(lat=lat, lon=lon)
        .rename({"Band1": variable})
        .drop_vars("crs")
    )
    if variable == "tas":  # noqa: SIM108
        ds = 0.1 * ds - 273.15
    else:
        ds = 0.1 * ds
    return ds


def load_and_downscale_anomaly(
    variable: str,
    scenario: str,
    year: int,
    lat: xr.DataArray,
    lon: xr.DataArray,
) -> xr.Dataset:
    in_root = (
        RRA_POP.human_niche_data
        / "chelsa-downscaled-projections"
        / "_anomalies"
        / "GLOBAL"
    )
    path = in_root / f"{variable}_{scenario}_{year}.nc"
    ds = xr.open_dataset(
        path,
        # Load the whole thing, but use a dask array
        chunks={"lat": -1, "lon": -1, "time": -1},
    ).interp(lat=lat, lon=lon)
    return ds


def apply_anomaly(data: xr.Dataset, anomaly: xr.Dataset) -> xr.Dataset:
    if "tas" in anomaly.keys():  # noqa: SIM118
        result = anomaly.groupby("time.month") + data
    else:
        result = anomaly.groupby("time.month") * data * (1 / 30)
    return result


def compute_measure(data: xr.Dataset, measure: str) -> xr.Dataset:
    if measure == "temperature":
        result = data.mean("time")
    elif measure == "precipitation":
        result = data.sum("time")
    else:
        threshold = 30
        result = (data > threshold).sum("time")
    return result


def project_climate_main(
    iso3: str,
    measure: str,
    scenario: str,
    pop_data_dir: str,
) -> None:
    pop_data = RRAPopulationData(pop_data_dir)
    admin0 = pop_data.load_shapefile(
        admin_level=0,
        iso3=iso3,
        year=2022,
    )
    minx, miny, maxx, maxy = admin0.total_bounds
    lat, lon = slice(miny, maxy), slice(minx, maxx)

    variable = {
        "temperature": "tas",
        "precipitation": "pr",
        "days_over_thirty": "tas",
    }[measure]

    print("Working on", scenario, measure)
    ds = get_chelsa(variable, lat, lon)

    results = []
    for year in range(2015, 2101):
        anom = load_and_downscale_anomaly(
            variable, scenario, year, ds["lat"], ds["lon"]
        )
        result = apply_anomaly(ds, anom)
        result = compute_measure(result, measure)
        results.append(result)
    result = xr.concat(results, dim=pd.Index(range(2015, 2101), name="year"))

    print("Writing results")
    pop_data.save_climate_data(
        result,
        measure=measure,
        iso3=iso3,
        scenario=scenario,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_iso3(allow_all=False)
@click.option(
    "--measure",
    type=click.Choice(["temperature", "precipitation", "days_over_thirty"]),
)
@clio.with_climate_scenario(allow_all=False)
@clio.with_input_directory("pop-data", RRA_DATA_ROOT)
def project_climate_task(
    iso3: str,
    measure: str,
    climate_scenario: str,
    pop_data_dir: str,
) -> None:
    project_climate_main(iso3, measure, climate_scenario, pop_data_dir)


@click.command()  # type: ignore[arg-type]
@clio.with_iso3(allow_all=False)
@clio.with_input_directory("pop-data", RRA_DATA_ROOT)
@clio.with_queue()
def project_climate(
    iso3: str,
    pop_data_dir: str,
    queue: str,
) -> None:
    pop_data = RRAPopulationData(pop_data_dir)
    jobmon.run_parallel(
        task_name="project_climate",
        node_args={
            "iso3": [
                iso3,
            ],
            "measure": [
                "temperature",
                "precipitation",
                "days_over_thirty",
            ],
            "scenario": list(clio.VALID_CLIMATE_SCENARIOS),
        },
        task_args={
            "pop-data-dir": pop_data_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 2,
            "memory": "70G",
            "runtime": "120m",
            "project": "proj_rapidresponse",
        },
        runner="rptask",
        log_root=pop_data.climate_data,
    )
