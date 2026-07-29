"""Standalone "step 4": link 100 draws to the compiled person-days files.

Run this AFTER `cdrun special compile_person_days` has finished. It creates, for
each subset hierarchy (gbd_2023, fhs_2023) and scenario:

    {AGGREGATE_ROOT}/{version}/results/{hierarchy}/temperature_person_days_{scenario}/{draw}.parquet

as a symlink to the compiled source, which `compile_person_days` writes under the
PIXEL hierarchy (note the extra path segment -- results versions do not have it):

    {AGGREGATE_ROOT}/{pixel_hierarchy}/erf-scratch/compiled-person-days/{hierarchy}/{scenario}_{gcm_member}.parquet

The draw -> gcm_member mapping is read PER SCENARIO from the annual-results draw
symlinks (results/annual/{scenario}/mean_temperature/{draw}.nc), so the person-days
draws line up with the temperature draws. (Empirically the mapping is identical
across scenarios, but reading per-scenario is robust if that ever changes.)

Design notes / fixes vs. the old inlined block:
  * Standalone -- uses only the data-access layer in `climate_data.data`, so it
    does not depend on any parameterized-branch APIs.
  * ADDITIVE-ONLY on shared storage (/mnt/team): it only CREATES new symlinks and
    never deletes, overwrites, or renames. Pre-existing outputs are SKIPPED and
    reported, per IHME shared-storage policy. To replace an existing link, remove
    it yourself and re-run.
  * Verifies each compiled source exists before linking -- no dangling symlinks.
    Missing sources are reported, not linked, so it is safe to run before compile
    fully finishes; just re-run to fill the gaps.
  * --dry-run previews the plan without touching the filesystem.

Run from a climate-data checkout, using an env where this package is installed
(`poetry install`) -- a bare `python` from an unrelated env will fail on
`import climate_data`:
    python scripts/link_person_days_draws.py --results-version 2026_05_28 --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from rra_tools.shell_tools import mkdir

from climate_data import constants as cdc
from climate_data.data import ClimateAggregateData, ClimateData

DEFAULT_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
MAX_MISSING_EXAMPLES = 5


@dataclass
class LinkCounts:
    """Running tallies over the whole link pass, shared across scenarios."""

    created: int = 0
    skipped: int = 0
    missing: int = 0
    missing_examples: list[str] = field(default_factory=list)

    def record_missing(self, raw_path: Path) -> None:
        """Count a compiled source that isn't there yet, keeping a few examples."""
        self.missing += 1
        if len(self.missing_examples) < MAX_MISSING_EXAMPLES:
            self.missing_examples.append(str(raw_path))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results-version",
        required=True,
        help="Version dir under AGGREGATE_ROOT to write results into, "
        "e.g. 2026_05_28. Required: there is no safe default, since writing into "
        "the wrong version silently scatters links across an unrelated release.",
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
    return ap.parse_args()


def _build_draw_map(
    cd_data: ClimateData, scenario: str, variable: str, n_draws: int
) -> dict[str, str]:
    """Map draw id -> gcm_member by resolving the annual-results draw symlinks."""
    draw_map: dict[str, str] = {}
    for d in range(n_draws):
        draw = f"{d:0>3}"
        target = cd_data.annual_results_path(scenario, variable, draw).resolve()
        draw_map[draw] = target.stem
    return draw_map


def _link_draws(
    *,
    src_data: ClimateAggregateData,
    out_root: Path,
    subset_hierarchy: str,
    scenario: str,
    draw_map: dict[str, str],
    counts: LinkCounts,
    dry_run: bool,
) -> None:
    """Link every draw for one (scenario, subset hierarchy), updating ``counts``.

    Additive-only: an existing output is left alone and counted as skipped, and a
    missing compiled source is reported rather than linked.
    """
    made_dir = False
    for draw, gcm_variant in draw_map.items():
        raw_path = src_data.compiled_person_days_path(
            subset_hierarchy, scenario, gcm_variant
        )
        out_path = out_root / f"{draw}.parquet"

        if not raw_path.exists():
            counts.record_missing(raw_path)
            continue
        if out_path.is_symlink() or out_path.exists():
            counts.skipped += 1
            continue
        if not dry_run:
            if not made_dir:
                mkdir(out_root, parents=True, exist_ok=True)
                made_dir = True
            out_path.symlink_to(raw_path)
        counts.created += 1


def _print_summary(counts: LinkCounts, *, dry_run: bool) -> None:
    print(
        f"Done. created={counts.created} skipped_existing={counts.skipped} "
        f"missing_source={counts.missing}"
    )
    if counts.missing:
        print(
            f"  {counts.missing} draws had no compiled source yet "
            "(compile still running, or a gap). Examples:"
        )
        for m in counts.missing_examples:
            print(f"    {m}")
        print("  Re-run after compile finishes to fill these.")
    if dry_run:
        print("\n(dry-run: nothing was created)")


def main() -> None:
    args = _parse_args()

    scenarios = args.scenarios or DEFAULT_SCENARIOS
    cd_data = ClimateData(args.climate_data_dir, read_only=True)
    # compile_person_days roots its manager at <output_dir>/<pixel_hierarchy>, so the
    # source manager must match it. Results versions live at <output_dir> itself, so
    # the two roots differ by exactly that segment and cannot share one manager.
    src_data = ClimateAggregateData(
        Path(args.output_dir) / args.pixel_hierarchy, read_only=True
    )
    out_data = ClimateAggregateData(args.output_dir, read_only=True)
    subset_hierarchies = cdc.HIERARCHY_MAP[args.pixel_hierarchy]
    results_root = out_data.results_root(args.results_version)

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"[{mode}] link person-days draws")
    print(f"  results_root      = {results_root}")
    print(
        f"  compiled source   = {src_data.root / 'erf-scratch' / 'compiled-person-days'}"
    )
    print(f"  subset hierarchies= {subset_hierarchies}")
    print(f"  scenarios         = {scenarios}")
    print(f"  draws             = {args.n_draws}\n")

    counts = LinkCounts()
    for scenario in scenarios:
        # per-scenario draw -> gcm_member, from the annual draw symlinks
        draw_map = _build_draw_map(cd_data, scenario, args.variable, args.n_draws)
        for subset_hierarchy in subset_hierarchies:
            out_root = (
                results_root / subset_hierarchy / f"temperature_person_days_{scenario}"
            )
            _link_draws(
                src_data=src_data,
                out_root=out_root,
                subset_hierarchy=subset_hierarchy,
                scenario=scenario,
                draw_map=draw_map,
                counts=counts,
                dry_run=args.dry_run,
            )

    _print_summary(counts, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
