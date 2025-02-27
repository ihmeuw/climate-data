import itertools
from collections import defaultdict

import click
import numpy as np
import tqdm
import xarray as xr
from rra_tools import jobmon

from climate_data import cli_options as clio
from climate_data import constants as cdc
from climate_data.data import ClimateData
from climate_data.generate.scenario_annual import TRANSFORM_MAP


def compile_gcm_main(
    target_variable: str,
    cmip6_experiment: str,
    gcm_member: str,
    ouptut_dir: str,
) -> None:
    cdata = ClimateData(ouptut_dir)
    print("Compiling", target_variable, cmip6_experiment, gcm_member)
    historical_paths = list(
        (cdata.raw_annual_results / "historical" / target_variable).glob("*.nc")
    )
    scenario_paths = list(
        (cdata.raw_annual_results / cmip6_experiment / target_variable).glob(
            f"*{gcm_member}.nc"
        )
    )
    print("Opening datasets")
    ds = (
        xr.open_mfdataset(historical_paths + scenario_paths, combine="by_coords")
        .sortby("year")
        .compute()
    )
    print("Saving compiled dataset")
    cdata.save_compiled_annual_results(
        ds,
        scenario=cmip6_experiment,
        variable=target_variable,
        gcm_member=gcm_member,
    )


def draws_main(
    target_variable: str,
    output_dir: str,
) -> None:
    cdata = ClimateData(output_dir)

    scenario_gcm_members = {}
    for scenario in cdc.CMIP6_EXPERIMENTS:
        paths = (cdata.compiled_annual_results / scenario / target_variable).glob(
            "*.nc"
        )
        scenario_gcm_members[scenario] = [p.stem for p in paths]

    # Ensure that all scenarios have the same members
    all_members = set().union(*scenario_gcm_members.values())
    differences = {}
    for scenario, members in scenario_gcm_members.items():
        differences[scenario] = all_members - set(members)

    if any(differences.values()):
        msg = "The following members are missing from some scenarios:\n"
        for scenario, missing_members in differences.items():
            if missing_members:
                msg += f"{scenario}: {missing_members}\n"
        raise ValueError(msg)

    # Sort for stability
    all_members = sorted(all_members)  # type: ignore[assignment]
    source_member_map = defaultdict(list)
    for gcm_member in all_members:
        source, member = gcm_member.split("_")
        source_member_map[source].append(member)

    num_draws = 100
    rs = np.random.RandomState(42)
    for draw in tqdm.trange(num_draws):
        gcm = rs.choice(list(source_member_map))
        member = rs.choice(source_member_map[gcm])
        gcm_member = f"{gcm}_{member}"
        for scenario in cdc.CMIP6_EXPERIMENTS:
            cdata.link_annual_draw(
                draw=draw,
                variable=target_variable,
                scenario=scenario,
                gcm_member=gcm_member,
            )


@click.command()
@clio.with_target_variable(TRANSFORM_MAP)
@clio.with_cmip6_experiment()
@clio.with_gcm_member()
@clio.with_output_directory(cdc.MODEL_ROOT)
def compile_gcm_task(
    target_variable: str,
    cmip6_experiment: str,
    gcm_member: str,
    output_dir: str,
) -> None:
    compile_gcm_main(target_variable, cmip6_experiment, gcm_member, output_dir)


@click.command()
@clio.with_target_variable(TRANSFORM_MAP, allow_all=True)
@clio.with_cmip6_experiment(allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_overwrite()
@clio.with_queue()
def draws(
    target_variable: list[str],
    cmip6_experiment: list[str],
    output_dir: str,
    overwrite: bool,
    queue: str,
) -> None:
    cdata = ClimateData(output_dir)

    to_run = []
    complete = []
    for variable, scenario in itertools.product(target_variable, cmip6_experiment):
        root = cdata.raw_annual_results / scenario / variable
        gcm_members = list({p.stem[5:] for p in root.glob("*.nc")})
        for gcm_member in gcm_members:
            out_path = cdata.compiled_annual_results_path(
                scenario, variable, gcm_member
            )
            if out_path.exists() and not overwrite:
                complete.append((variable, scenario, gcm_member))
            else:
                to_run.append((variable, scenario, gcm_member))

    print(f"{len(to_run)} GCM-members to compile.")

    status = jobmon.run_parallel(
        runner="cdtask",
        task_name="generate compile_gcm",
        flat_node_args=(
            ("target-variable", "cmip6-experiment", "gcm-member"),
            to_run,
        ),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "50G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
    )

    if status != "D":
        msg = f"GCM compilation failed with status {status}"
        raise RuntimeError(msg)

    for variable in target_variable:
        print("Generating draws for", variable)
        draws_main(variable, output_dir)
