import itertools
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData
from climate_data.generate import utils
from climate_data.jobmon_utils import run_parallel_maybe_dry_run

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


def _monthly_means_by_reference_year(reference: xr.Dataset) -> xr.Dataset:
    """Monthly means of the reference period, one slice per reference year.

    Dims ``(reference_year, month, latitude, longitude)``, on the GCM's own grid. The years
    come from the data rather than from ``cdc.REFERENCE_YEARS`` so that the ``(n-1)/n`` rescale
    below stays consistent with whatever window was actually loaded. Selection is by boolean
    mask rather than by date string so it works for the 360-day cftime calendars some GCMs use.
    """
    years = [int(y) for y in np.unique(reference["date"].dt.year.values)]
    by_year = []
    for year in years:
        one_year = reference.sel(date=reference["date"].dt.year == year)
        by_year.append(one_year.groupby("date.month").mean("date"))
    return xr.concat(by_year, dim="reference_year").assign_coords(
        reference_year=years,
    )


def jensen_debias_factor(
    reference: xr.Dataset,
    reference_monthly: xr.Dataset,
    debias_method: str,
) -> xr.Dataset:
    """The factor to divide a multiplicative anomaly by, per month and per GCM cell.

    The anomaly is ``(T + 1) / (R + 1)`` with ``R`` a monthly mean over only five reference
    years. ``1/(R + 1)`` is convex, so by Jensen's inequality the anomaly averages above 1 even
    when the target year is drawn from the same distribution as the reference period -- a level
    bias on every forecast year. This returns an estimate of that inflation.

    ``loo`` -- leave-one-out. For each held-out reference year, form the multiplier of that
    year against the mean of the *other* years and average over the folds. Between reference
    years there is no climate signal, so an unbiased estimator would return 1; the excess is
    the bias, measured from the data with no series expansion. The held-out denominator
    averages ``n-1`` years while the pipeline averages ``n``, and the bias goes as ``1/n``, so
    the excess is rescaled by ``(n-1)/n``.

    This is provably ``>= 1``: with ``u_y = T_y + 1`` and ``S = sum_y u_y`` the held-out
    denominator is ``(S - u_y)/(n-1)``, so each fold is ``(n-1)*u_y/(S - u_y)``, convex in
    ``u_y``, and Jensen gives ``mean_y >= f(S/n) = 1`` with equality iff every reference year
    is identical. So dividing by it can only shrink the anomaly, never inflate it -- which is
    what makes the effect on a threshold count such as ``precipitation_days`` sign-definite.

    ``analytic`` -- the second-order expansion ``1 + Var(Rbar)/(R + 1)^2``. Cheaper to reason
    about but it is a truncated series, and the neglected terms matter exactly where the
    correction is largest (near-zero ``R``, where ``eps`` dominates the denominator). Kept for
    comparison; ``loo`` is the estimator this was built for.

    ``reference_monthly`` is the pipeline's own denominator, passed in rather than recomputed so
    the analytic form squares precisely the value the anomaly divides by.
    """
    by_year = _monthly_means_by_reference_year(reference)
    n_years = by_year.sizes["reference_year"]

    if debias_method == "loo":
        mean_year = by_year.mean("reference_year")
        folds = []
        for i in range(n_years):
            held_out = by_year.isel(reference_year=i)
            others = (n_years * mean_year - held_out) / (n_years - 1)
            folds.append((held_out + 1) / (others + 1))
        raw = xr.concat(folds, dim="reference_year").mean("reference_year")
        factor = 1.0 + ((n_years - 1) / n_years) * (raw - 1.0)
    elif debias_method == "analytic":
        variance = by_year.var("reference_year", ddof=1)
        factor = 1.0 + (variance / n_years) / (reference_monthly + 1) ** 2
    else:
        msg = f"Unknown debias method: {debias_method}"
        raise ValueError(msg)

    factor = factor.drop_vars("reference_year", errors="ignore")
    if not bool(np.isfinite(factor.to_dataarray()).all()):
        msg = (
            f"Non-finite value in the {debias_method} de-bias factor. Interpolation would "
            "silently fill it from a neighbour rather than surface it, so refusing to proceed."
        )
        raise ValueError(msg)
    return factor


def compute_anomaly(
    reference: xr.Dataset,
    target: xr.Dataset,
    anomaly_type: str,
    *,
    debias_method: str,
) -> xr.Dataset:
    reference_monthly = reference.groupby("date.month").mean("date")
    if anomaly_type == "additive":
        if debias_method != "none":
            msg = (
                f"debias_method={debias_method!r} was requested for an additive anomaly. The "
                "Jensen bias comes from the convexity of 1/(R + eps) and has no additive "
                "counterpart, so there is nothing to correct."
            )
            raise ValueError(msg)
        anomaly = target.groupby("date.month") - reference_monthly
    elif anomaly_type == "multiplicative":
        denominator = reference_monthly + 1
        if debias_method != "none":
            # Fold the factor into the denominator rather than dividing the anomaly by it.
            # `anomaly / factor` silently OUTER-BROADCASTS to (date, latitude, longitude,
            # month) -- no error raised -- which is a 12x blow-up of an eager multi-GB daily
            # array. Folding costs one 12-month temporary and is numerically identical.
            denominator = denominator * jensen_debias_factor(
                reference, reference_monthly, debias_method
            )
        anomaly = (target + 1).groupby("date.month") / denominator
    else:
        msg = f"Unknown anomaly type: {anomaly_type}"
        raise ValueError(msg)
    anomaly = anomaly.drop_vars("month")
    return anomaly


def check_debias_variable(target_variable: str, debias_method: str) -> None:
    """Refuse a de-bias for a variable it has not been validated against.

    Called from the launchers as well as from the worker, so that ``--target-variable all``
    fails in a second rather than after submitting thousands of doomed jobs.
    """
    if debias_method != "none" and target_variable not in cdc.DEBIAS_VARIABLES:
        msg = (
            f"debias_method={debias_method!r} is not validated for {target_variable!r}. "
            f"Allowed: {list(cdc.DEBIAS_VARIABLES)}. Name the variable explicitly rather "
            "than using 'all'."
        )
        raise ValueError(msg)


def generate_scenario_daily_main(
    target_variable: str,
    cmip6_experiment: str,
    year: str | int,
    gcm_member: str,
    output_dir: str | Path,
    write_output: bool = True,
    *,
    debias_method: str,
) -> xr.Dataset:
    # NOTE: debias_method is deliberately keyword-only with NO default. A default of "none"
    # here would mean that forgetting to thread it through generate_scenario_annual_main
    # produces a silently undebiased run that reports success. Let mypy catch the call site.
    cdata = ClimateData(output_dir)
    check_debias_variable(target_variable, debias_method)

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
    v_anomaly = compute_anomaly(sref, target, anomaly_type, debias_method=debias_method)

    print(f"{gcm_member}: resampling anomaly")
    resampled_anomaly = utils.interpolate_to_target_latlon(v_anomaly, method="linear")
    print(f"{gcm_member}: computing scenario data")
    if anomaly_type == "additive":
        scenario_data = historical_reference + resampled_anomaly.groupby("date.month")
    else:
        scenario_data = historical_reference * resampled_anomaly.groupby("date.month")
    scenario_data.attrs["debias_method"] = debias_method

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


@click.command()
@clio.with_target_variable(list(TRANSFORM_MAP))
@clio.with_cmip6_experiment()
@clio.with_year(cdc.FORECAST_YEARS)
@clio.with_gcm_member()
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_debias_method()
def generate_scenario_daily_task(
    target_variable: str,
    cmip6_experiment: str,
    year: str,
    gcm_member: str,
    output_dir: str,
    debias_method: str,
) -> None:
    generate_scenario_daily_main(
        target_variable,
        cmip6_experiment,
        year,
        gcm_member,
        output_dir,
        write_output=True,
        debias_method=debias_method,
    )


@click.command()
@clio.with_target_variable(TRANSFORM_MAP, allow_all=True)
@clio.with_cmip6_experiment(allow_all=True)
@clio.with_year(cdc.FORECAST_YEARS, allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_debias_method()
@clio.with_queue()
@clio.with_overwrite()
@clio.with_dry_run()
def generate_scenario_daily(
    target_variable: list[str],
    cmip6_experiment: list[str],
    year: list[str],
    output_dir: str,
    debias_method: str,
    queue: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    # Fail before submitting anything: with `-t all` a de-bias request would otherwise die
    # one job at a time, after the whole fan-out is already queued.
    for variable in target_variable:
        check_debias_variable(variable, debias_method)
    cdata = ClimateData(output_dir)
    veyg = []
    complete = []
    for v, e, y in itertools.product(target_variable, cmip6_experiment, year):
        source_variables = TRANSFORM_MAP[v][0].source_variables
        gcms = cdata.get_gcms(source_variables)
        for g in gcms:
            path = cdata.raw_daily_results_path(e, v, y, g)
            if not path.exists() or overwrite:
                veyg.append((v, e, y, g))
            else:
                complete.append((v, e, y, g))
    if not veyg:
        print("All tasks already done.")
        return

    print(f"{len(complete)} tasks already done. Launching {len(veyg)} tasks")
    run_parallel_maybe_dry_run(
        runner="cdtask",
        task_name="generate scenario_daily",
        flat_node_args=(
            ("target-variable", "cmip6-experiment", "year", "gcm-member"),
            veyg,
        ),
        task_args={
            "output-dir": output_dir,
            "debias-method": debias_method,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "90G",
            "runtime": "20m",
            "project": "proj_rapidresponse",
        },
        max_attempts=2,
        dry_run=dry_run,
    )
