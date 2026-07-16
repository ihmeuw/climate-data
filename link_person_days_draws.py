#!/usr/bin/env python
"""Standalone "step 4": link 100 draws to the compiled person-days files.

Run this AFTER `cdrun special compile_person_days` has finished. It creates, for
each subset hierarchy (gbd_2023, fhs_2023) and scenario:

    {AGGREGATE_ROOT}/{version}/results/{hierarchy}/temperature_person_days_{scenario}/{draw}.parquet

as a symlink to the compiled source:

    {AGGREGATE_ROOT}/erf-scratch/compiled-person-days/{hierarchy}/{scenario}_{gcm_member}.parquet

The draw -> gcm_member mapping is read PER SCENARIO from the annual-results draw
symlinks (results/annual/{scenario}/mean_temperature/{draw}.nc), so the person-days
draws line up with the temperature draws. (Empirically the mapping is identical
across scenarios, but reading per-scenario is robust if that ever changes.)

Design notes / fixes vs. the old inlined block:
  * Standalone -- does NOT depend on the parameterized branch APIs; uses only the
    deploy/main data layer.
  * ADDITIVE-ONLY on shared storage (/mnt/team): it only CREATES new symlinks and
    never deletes, overwrites, or renames. Pre-existing outputs are SKIPPED and
    reported, per IHME shared-storage policy. To replace an existing link, remove
    it yourself and re-run.
  * Verifies each compiled source exists before linking -- no dangling symlinks.
    Missing sources are reported, not linked, so it is safe to run before compile
    fully finishes; just re-run to fill the gaps.
  * --dry-run previews the plan without touching the filesystem.

Run with the deploy env, e.g.:
    /ihme/homes/billg/miniconda3_new/envs/dep-clim/bin/python \
        ~/deploy/climate-data/link_person_days_draws.py --dry-run
"""

from __future__ import annotations

import argparse

from rra_tools.shell_tools import mkdir

from climate_data import constants as cdc
from climate_data.data import ClimateAggregateData, ClimateData

DEFAULT_SCENARIOS = ["ssp126", "ssp245", "ssp585"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results-version",
        default="2026_05_28",
        help="Version dir under AGGREGATE_ROOT to write results into "
        "(default: %(default)s). CONFIRM this is the intended version.",
    )
    ap.add_argument(
        "--pixel-hierarchy",
        default="gbd_2023",
        choices=list(cdc.HIERARCHY_MAP),
        help="Pixel hierarchy whose subset hierarchies get linked "
        "(default: %(default)s -> %(default)s produces its HIERARCHY_MAP subsets).",
    )
    ap.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Scenario(s); repeatable. Default: ssp126 ssp245 ssp585.",
    )
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument("--variable", default="mean_temperature")
    ap.add_argument("--output-dir", default=str(cdc.AGGREGATE_ROOT))
    ap.add_argument("--climate-data-dir", default=str(cdc.MODEL_ROOT))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only; create nothing.",
    )
    args = ap.parse_args()

    scenarios = args.scenarios or DEFAULT_SCENARIOS
    cd_data = ClimateData(args.climate_data_dir, read_only=True)
    ca_data = ClimateAggregateData(args.output_dir)
    subset_hierarchies = cdc.HIERARCHY_MAP[args.pixel_hierarchy]
    results_root = ca_data.results_root(args.results_version)

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"[{mode}] link person-days draws")
    print(f"  results_root      = {results_root}")
    print(f"  compiled source   = {ca_data.root / 'erf-scratch' / 'compiled-person-days'}")
    print(f"  subset hierarchies= {subset_hierarchies}")
    print(f"  scenarios         = {scenarios}")
    print(f"  draws             = {args.n_draws}\n")

    created = skipped = missing = 0
    missing_examples: list[str] = []

    for scenario in scenarios:
        # per-scenario draw -> gcm_member, from the annual draw symlinks
        draw_map: dict[str, str] = {}
        for d in range(args.n_draws):
            draw = f"{d:0>3}"
            target = cd_data.annual_results_path(
                scenario, args.variable, draw
            ).resolve()
            draw_map[draw] = target.stem

        for subset_hierarchy in subset_hierarchies:
            out_root = (
                results_root
                / subset_hierarchy
                / f"temperature_person_days_{scenario}"
            )
            made_dir = False
            for draw, gcm_variant in draw_map.items():
                raw_path = (
                    ca_data.root
                    / "erf-scratch"
                    / "compiled-person-days"
                    / subset_hierarchy
                    / f"{scenario}_{gcm_variant}.parquet"
                )
                out_path = out_root / f"{draw}.parquet"

                if not raw_path.exists():
                    missing += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(str(raw_path))
                    continue
                if out_path.is_symlink() or out_path.exists():
                    skipped += 1
                    continue
                if not args.dry_run:
                    if not made_dir:
                        mkdir(out_root, parents=True, exist_ok=True)
                        made_dir = True
                    out_path.symlink_to(raw_path)
                created += 1

    print(
        f"Done. created={created} skipped_existing={skipped} "
        f"missing_source={missing}"
    )
    if missing:
        print(
            f"  {missing} draws had no compiled source yet "
            "(compile still running, or a gap). Examples:"
        )
        for m in missing_examples:
            print(f"    {m}")
        print("  Re-run after compile finishes to fill these.")
    if args.dry_run:
        print("\n(dry-run: nothing was created)")


if __name__ == "__main__":
    main()
