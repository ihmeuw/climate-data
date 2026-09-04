import itertools
from pathlib import Path
from typing import Any

import click
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData
from climate_data.generate import utils
from climate_data.generate.scenario_daily import (
    TRANSFORM_MAP as DAILY_TRANSFORM_MAP,
)
from climate_data.generate.scenario_daily import (
    check_debias_variable,
    generate_scenario_daily_main,
)
from climate_data.jobmon_utils import run_parallel_maybe_dry_run

TEMP_THRESHOLDS = [30]


TRANSFORM_MAP = {
    "mean_temperature": utils.Transform(
        source_variables=["mean_temperature"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    "mean_high_temperature": utils.Transform(
        source_variables=["max_temperature"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    "mean_low_temperature": utils.Transform(
        source_variables=["min_temperature"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    **{
        f"days_over_{temp}C": utils.Transform(
            source_variables=["mean_temperature"],
            transform_funcs=[utils.count_threshold(temp), utils.annual_sum],
        )
        for temp in TEMP_THRESHOLDS
    },
    **{
        f"{disease}_suitability": utils.Transform(
            source_variables=["mean_temperature"],
            transform_funcs=[
                utils.map_suitability(disease),
                utils.annual_sum,
            ],
        )
        for disease in ["malaria", "dengue"]
    },
    "wind_speed": utils.Transform(
        source_variables=["wind_speed"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    "relative_humidity": utils.Transform(
        source_variables=["relative_humidity"],
        transform_funcs=[utils.annual_mean],
        encoding_scale=0.01,
    ),
    "total_precipitation": utils.Transform(
        source_variables=["total_precipitation"],
        transform_funcs=[utils.annual_sum],
        encoding_scale=10,
    ),
    "precipitation_days": utils.Transform(
        source_variables=["total_precipitation"],
        transform_funcs=[utils.count_threshold(0.1), utils.annual_sum],
    ),
}

# Notes about what to do:
# We want to leave the interface for this function/entry point essentially the same.  We'll add in
# a `draw` argument to the task function, but otherwise we'll keep the same interface.
# The idea here is to take a target variable in annual space, get all the source variables,
# compute the daily source variables in memory, then collapse them to the annual target variable.


ANOMALY_TYPES = {}
for _variable, _transform in TRANSFORM_MAP.items():
    _types = set()
    for _source_variable in _transform.source_variables:
        _types.add(DAILY_TRANSFORM_MAP[_source_variable][1])
    ANOMALY_TYPES[_variable] = (
        "multiplicative" if _types == {"multiplicative"} else "additive"
    )


def forecast_jobs_for_anomaly_scheme(
    to_run: list[tuple[str, str, str, str]],
    anomaly_scheme: str,
) -> list[tuple[str, str, str, str]]:
    """Drop forecast jobs whose variable the anomaly scheme cannot be applied to.

    Filtered per job rather than per variable because the `historical` scenario never
    reaches `generate_scenario_daily_main` -- it reads the daily results off disk -- so
    the scheme does not constrain it, and an additive variable is perfectly runnable
    there. Only the forecast scenarios pass through `compute_anomaly`.
    """
    if anomaly_scheme == cdc.ANOMALY_SCHEME_MONTHLY:
        return to_run

    keep = []
    skipped_variables = set()
    for job in to_run:
        variable, scenario = job[0], job[1]
        blocked = (
            scenario != "historical" and ANOMALY_TYPES[variable] != "multiplicative"
        )
        if blocked:
            skipped_variables.add(variable)
        else:
            keep.append(job)

    dropped = len(to_run) - len(keep)
    if dropped:
        print(
            f"Anomaly scheme '{anomaly_scheme}' applies to multiplicative variables only;"
            f" skipping {dropped} forecast tasks for:"
            f" {', '.join(sorted(skipped_variables))}."
        )
    return keep


def generate_scenario_annual_main(
    target_variable: str,
    scenario: str,
    year: str,
    gcm_member: str,
    output_dir: str | Path,
    progress_bar: bool = False,
    *,
    debias_method: str,
    dry_day_rule: str,
    anomaly_scheme: str = cdc.ANOMALY_SCHEME_MONTHLY,
    reference_years: str = cdc.REFERENCE_YEARS_ARG,
    eps_floor: float = cdc.DEFAULT_EPS_FLOOR,
    anomaly_cap: float | None = cdc.DEFAULT_ANOMALY_CAP,
) -> None:
    # NOTE: keyword-only with NO default, on purpose -- see the note in
    # generate_scenario_daily_main. A default here would let a missed hand-off produce a
    # silently undebiased product that reports success.
    cdata = ClimateData(output_dir)
    transform = TRANSFORM_MAP[target_variable]

    print("Loading files")
    if scenario == "historical":
        ds = transform(
            *[
                xr.open_dataset(
                    cdata.daily_results_path(scenario, source_variable, year)
                )
                for source_variable in transform.source_variables
            ]
        )
    else:
        ds = transform(
            *[
                generate_scenario_daily_main(
                    output_dir=output_dir,
                    year=year,
                    gcm_member=gcm_member,
                    target_variable=source_variable,
                    cmip6_experiment=scenario,
                    write_output=False,
                    debias_method=debias_method,
                    dry_day_rule=dry_day_rule,
                    anomaly_scheme=anomaly_scheme,
                    reference_years=reference_years,
                    eps_floor=eps_floor,
                    anomaly_cap=anomaly_cap,
                )
                for source_variable in transform.source_variables
            ]
        )
    if progress_bar:
        with ProgressBar():  # type: ignore[no-untyped-call]
            ds = ds.compute()
    else:
        ds = ds.compute()

    ds.attrs["debias_method"] = debias_method
    ds.attrs["dry_day_rule"] = dry_day_rule
    ds.attrs["anomaly_scheme"] = anomaly_scheme
    ds.attrs["reference_years"] = reference_years

    print("Saving files")
    cdata.save_raw_annual_results(
        ds,
        scenario=scenario,
        variable=target_variable,
        year=year,
        gcm_member=gcm_member,
        encoding_kwargs=transform.encoding_kwargs,
    )


@click.command()
@clio.with_target_variable(list(TRANSFORM_MAP))
@clio.with_scenario()
@clio.with_year(cdc.HISTORY_YEARS + cdc.FORECAST_YEARS)
@clio.with_gcm_member()
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_debias_method()
@clio.with_dry_day_rule()
@clio.with_anomaly_scheme()
@clio.with_reference_years()
@clio.with_eps_floor()
@clio.with_anomaly_cap()
def generate_scenario_annual_task(
    target_variable: str,
    scenario: str,
    year: str,
    gcm_member: str,
    output_dir: str,
    debias_method: str,
    dry_day_rule: str,
    anomaly_scheme: str,
    reference_years: str,
    eps_floor: float,
    anomaly_cap: float | None,
) -> None:
    history_flags = [
        year in cdc.HISTORY_YEARS,
        scenario == "historical",
        gcm_member == "era5",
    ]
    if any(history_flags) and not all(history_flags):
        msg = f"Historical years must use the 'historical' experiment and era5 GCM member. {year} {scenario} {gcm_member}"
        raise ValueError(msg)
    if all(history_flags) and (debias_method != "none" or dry_day_rule != "none"):
        msg = (
            "debias_method / dry_day_rule have no effect on the historical branch, which "
            "reads ERA5 dailies straight from disk and never computes an anomaly. Refusing "
            "rather than writing output whose provenance implies a correction was applied. "
            f"Got debias_method={debias_method!r}, dry_day_rule={dry_day_rule!r}."
        )
        raise ValueError(msg)

    generate_scenario_annual_main(
        target_variable,
        scenario,
        year,
        gcm_member,
        output_dir,
        progress_bar=False,
        debias_method=debias_method,
        dry_day_rule=dry_day_rule,
        anomaly_scheme=anomaly_scheme,
        reference_years=reference_years,
        eps_floor=eps_floor,
        anomaly_cap=anomaly_cap,
    )


def build_arg_list(
    target_variables: list[str],
    scenarios: list[str],
    output_dir: str,
    overwrite: bool,
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    cdata = ClimateData(output_dir)
    to_run, complete = [], []
    trc, cc = 0, 0

    print_template = "{v:<30} {e:<12} {tra:>10} {ca:>10}"
    print(
        print_template.format(v="VARIABLE", e="EXPERIMENT", tra="TO_RUN", ca="COMPLETE")
    )

    for v, s in itertools.product(target_variables, scenarios):
        if s == "historical":
            years = cdc.HISTORY_YEARS
            gcm_members = ["era5"]
        else:
            years = cdc.FORECAST_YEARS
            annual_source_variables = TRANSFORM_MAP[v].source_variables
            daily_source_variables = itertools.chain(
                *[
                    DAILY_TRANSFORM_MAP[source_variable][0].source_variables
                    for source_variable in annual_source_variables
                ]
            )
            gcm_members = cdata.get_gcms(list(daily_source_variables))

        for y, g in itertools.product(years, gcm_members):
            path = cdata.raw_annual_results_path(
                scenario=s, variable=v, year=y, gcm_member=g
            )
            if not path.exists():
                to_run.append((v, s, y, g))
            else:
                complete.append((v, s, y, g))

        tra, ca = len(to_run) - trc, len(complete) - cc
        trc, cc = len(to_run), len(complete)
        print(print_template.format(v=v, e=s, tra=tra, ca=ca))

    if overwrite:
        to_run += complete
        complete = []

    return to_run, complete


@click.command()
@clio.with_target_variable(TRANSFORM_MAP, allow_all=True)
@clio.with_scenario(allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_debias_method()
@clio.with_dry_day_rule()
@clio.with_anomaly_scheme()
@clio.with_reference_years()
@clio.with_eps_floor()
@clio.with_anomaly_cap()
@clio.with_queue()
@clio.with_concurrency_limit(default=500)
@clio.with_overwrite()
@clio.with_dry_run()
def generate_scenario_annual(
    target_variable: list[str],
    scenario: list[str],
    output_dir: str,
    debias_method: str,
    dry_day_rule: str,
    anomaly_scheme: str,
    reference_years: str,
    eps_floor: float,
    anomaly_cap: float | None,
    queue: str,
    concurrency_limit: int | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    # Fail before submitting: `-t all` spans additive variables, which a de-bias request
    # must reject, and finding that out one job at a time wastes the whole fan-out.
    for variable in target_variable:
        for source_variable in TRANSFORM_MAP[variable].source_variables:
            check_debias_variable(source_variable, debias_method, dry_day_rule)

    to_run, complete = build_arg_list(
        target_variable,
        scenario,
        output_dir,
        overwrite,
    )
    to_run = forecast_jobs_for_anomaly_scheme(to_run, anomaly_scheme)

    print(f"{len(complete)} tasks already done. {len(to_run)} tasks to do.")

    if not to_run:
        return
    # `anomaly_cap` is None whenever no ceiling was requested, which is the default.
    # Passing the key through with a None value renders a bare `--anomaly-cap` onto the
    # task command line with nothing after it, and click rejects it -- so EVERY task of
    # an uncapped run fails with "Option '--anomaly-cap' requires an argument." while the
    # controller still exits 0. Omit the key instead, and let the task's own default
    # stand. Every run between b25dc84 and this fix passed an explicit cap, which is why
    # the default path went unexercised.
    task_args: dict[str, Any] = {
        "output-dir": output_dir,
        "debias-method": debias_method,
        "dry-day-rule": dry_day_rule,
        "anomaly-scheme": anomaly_scheme,
        "reference-years": reference_years,
        "eps-floor": eps_floor,
    }
    if anomaly_cap is not None:
        task_args["anomaly-cap"] = anomaly_cap
    run_parallel_maybe_dry_run(
        runner="cdtask",
        task_name="generate scenario_annual",
        flat_node_args=(
            ("target-variable", "scenario", "year", "gcm-member"),
            to_run,
        ),
        task_args=task_args,
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "100G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        concurrency_limit=concurrency_limit,
        dry_run=dry_run,
    )
