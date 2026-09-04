"""Standalone "step 4": link draws to the compiled person-days files.

Run this AFTER `cdrun special compile_person_days` has finished. It creates, for
each subset hierarchy (gbd_2023, fhs_2023) and scenario:

    {AGGREGATE_ROOT}/{version}/results/{hierarchy}/temperature_person_days_{scenario}/{draw}.parquet

as a symlink to the compiled source, which `compile_person_days` writes under the
PIXEL hierarchy (note the extra path segment -- results versions do not have it):

    {AGGREGATE_ROOT}/{pixel_hierarchy}/erf-scratch/compiled-person-days/{hierarchy}/{scenario}_{gcm_member}.parquet

The draw -> gcm_member mapping is read PER SCENARIO from the annual-results draw
symlinks (results/annual/{scenario}/mean_temperature/{draw}.nc), so the person-days
draws line up with the temperature draws. The mapping must agree across scenarios
(`generate/draws.py` enforces that upstream) -- otherwise a scenario delta at a
fixed draw index would difference two different climate models, so a divergence
aborts the run rather than publishing a contaminated warming signal.

Design notes:
  * Standalone -- uses only the data-access layer in `climate_data.data`, so it
    does not depend on any parameterized-branch APIs.
  * ADDITIVE-ONLY on shared storage (/mnt/team): it only CREATES new symlinks and
    never deletes, overwrites, or renames. To replace an existing link, remove it
    yourself and re-run.
  * FAILS LOUDLY rather than plausibly. Every degenerate case that used to print a
    healthy-looking summary and exit 0 now exits non-zero:
      - the results version must already exist (a typo cannot mkdir a new release)
      - annual draw paths must be symlinks that resolve (no fabricating a
        gcm_member from the draw id)
      - an existing output whose target differs from the intended source is a
        CONFLICT, not a skip -- it would otherwise silently serve the wrong GCM
      - any missing source or conflict exits 1, so `&& next-step` chaining is safe
  * --dry-run previews the plan without creating anything.

Run from a climate-data checkout, using an env where this package is installed
(`poetry install`) -- a bare `python` from an unrelated env will fail on
`import climate_data`:
    python scripts/link_person_days_draws.py --results-version 2026_05_28 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rra_tools.shell_tools import mkdir

from climate_data import constants as cdc
from climate_data.data import ClimateAggregateData, ClimateData

MAX_EXAMPLES = 5


@dataclass
class LinkCounts:
    """Running tallies over the whole link pass, shared across scenarios."""

    created: int = 0
    skipped: int = 0
    missing: int = 0
    conflicts: int = 0
    missing_examples: list[str] = field(default_factory=list)
    conflict_examples: list[str] = field(default_factory=list)

    def record_missing(self, raw_path: Path) -> None:
        """Count a compiled source that isn't there yet, keeping a few examples."""
        self.missing += 1
        if len(self.missing_examples) < MAX_EXAMPLES:
            self.missing_examples.append(str(raw_path))

    def record_conflict(self, out_path: Path, want: Path) -> None:
        """Count an existing output that points somewhere other than ``want``."""
        self.conflicts += 1
        if len(self.conflict_examples) < MAX_EXAMPLES:
            have = out_path.resolve() if out_path.is_symlink() else out_path
            self.conflict_examples.append(
                f"{out_path}\n      -> {have}\n      want {want}"
            )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results-version",
        required=True,
        help="Existing version dir under AGGREGATE_ROOT to write results into, "
        "e.g. 2026_05_28. Required, and must already exist: there is no safe "
        "default, and writing into the wrong version scatters links across an "
        "unrelated release.",
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
        choices=list(cdc.AGGREGATION_SCENARIOS),
        help="Scenario(s); repeatable. Default: all of "
        f"{' '.join(cdc.AGGREGATION_SCENARIOS)}.",
    )
    ap.add_argument(
        "--n-draws",
        type=int,
        default=len(cdc.DRAWS),
        help="Link only the first N of the %(default)s draws in constants.DRAWS "
        "(default: %(default)s). Mainly for debugging.",
    )
    ap.add_argument("--variable", default="mean_temperature")
    ap.add_argument("--output-dir", default=str(cdc.AGGREGATE_ROOT))
    ap.add_argument("--climate-data-dir", default=str(cdc.MODEL_ROOT))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only; create nothing.",
    )
    return ap.parse_args()


def _resolve_results_root(out_data: ClimateAggregateData, version: str) -> Path:
    """Return the results root for ``version``, refusing to invent a new release."""
    results_root = out_data.results_root(version)
    if results_root.exists():
        return results_root
    available = []
    for path in sorted(out_data.root.glob("*/results")):
        available.append(path.parent.name)
    sys.exit(
        f"error: results version {version!r} does not exist:\n"
        f"  {results_root}\n"
        f"  available versions: {', '.join(available) or '(none found)'}\n"
        "  refusing to create a new version tree; pass an existing version."
    )


def _build_draw_map(
    cd_data: ClimateData, scenario: str, variable: str, draws: list[str]
) -> dict[str, str]:
    """Map draw id -> gcm_member by resolving the annual-results draw symlinks.

    Aborts rather than guessing. ``Path.resolve()`` is non-strict, so a missing or
    plain-file draw path would quietly yield the draw id itself as the gcm_member
    and send every source lookup to a fabricated filename.
    """
    draw_map: dict[str, str] = {}
    for draw in draws:
        path = cd_data.annual_results_path(scenario, variable, draw)
        if not path.is_symlink():
            sys.exit(
                f"error: annual draw is not a symlink to a gcm member:\n  {path}\n"
                f"  link the {scenario}/{variable} annual draws first "
                "(cdrun generate draws)."
            )
        try:
            target = path.resolve(strict=True)
        except OSError as exc:
            sys.exit(f"error: annual draw symlink is dangling:\n  {path}\n  {exc}")
        draw_map[draw] = target.stem
    return draw_map


def _draw_maps_by_scenario(
    cd_data: ClimateData, scenarios: list[str], variable: str, draws: list[str]
) -> dict[str, dict[str, str]]:
    """Build every scenario's draw map and require them to agree.

    Scenario deltas at a fixed draw index are only meaningful if draw N is the same
    climate model in every scenario. That invariant holds upstream today; verify it
    instead of assuming it.
    """
    draw_maps = {}
    for scenario in scenarios:
        draw_maps[scenario] = _build_draw_map(cd_data, scenario, variable, draws)

    reference_scenario, reference = next(iter(draw_maps.items()))
    for scenario, draw_map in draw_maps.items():
        if draw_map == reference:
            continue
        differing = []
        for draw, gcm_member in draw_map.items():
            if reference.get(draw) != gcm_member:
                differing.append(
                    f"    draw {draw}: {reference_scenario}={reference.get(draw)} "
                    f"vs {scenario}={gcm_member}"
                )
        detail = "\n".join(differing[:MAX_EXAMPLES])
        sys.exit(
            "error: the draw -> gcm_member mapping differs between scenarios, so a "
            "fixed draw index would compare different climate models:\n"
            f"{detail}"
        )
    return draw_maps


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

    Additive-only: an existing output is never replaced. One that already points at
    the intended source is a skip; one pointing elsewhere is a conflict.
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
            if out_path.resolve() == raw_path.resolve():
                counts.skipped += 1
            else:
                counts.record_conflict(out_path, raw_path)
            continue
        if not dry_run:
            if not made_dir:
                mkdir(out_root, parents=True, exist_ok=True)
                made_dir = True
            out_path.symlink_to(raw_path)
        counts.created += 1


def _print_summary(counts: LinkCounts, *, dry_run: bool) -> None:
    label = "would_create" if dry_run else "created"
    print(
        f"Done. {label}={counts.created} skipped_existing={counts.skipped} "
        f"missing_source={counts.missing} conflicts={counts.conflicts}"
    )
    if counts.missing:
        print(f"  {counts.missing} draws had no compiled source. Examples:")
        for m in counts.missing_examples:
            print(f"    {m}")
        print("  Re-run once compile_person_days has produced them.")
    if counts.conflicts:
        print(
            f"  {counts.conflicts} existing outputs point somewhere other than the "
            "intended source. Examples:"
        )
        for c in counts.conflict_examples:
            print(f"    {c}")
        print(
            "  These were NOT modified. Inspect them, remove the stale links "
            "yourself, then re-run."
        )
    if dry_run:
        print("\n(dry-run: nothing was created)")


def main() -> None:
    args = _parse_args()

    if args.n_draws > len(cdc.DRAWS):
        sys.exit(
            f"error: --n-draws {args.n_draws} exceeds the {len(cdc.DRAWS)} draws "
            "defined in constants.DRAWS."
        )
    draws = cdc.DRAWS[: args.n_draws]
    scenarios = args.scenarios or list(cdc.AGGREGATION_SCENARIOS)

    cd_data = ClimateData(args.climate_data_dir, read_only=True)
    # compile_person_days roots its manager at <output_dir>/<pixel_hierarchy>, so the
    # source manager must match it. Results versions live at <output_dir> itself, so
    # the two roots differ by exactly that segment and cannot share one manager.
    src_data = ClimateAggregateData(
        Path(args.output_dir) / args.pixel_hierarchy, read_only=True
    )
    out_data = ClimateAggregateData(args.output_dir, read_only=True)
    subset_hierarchies = cdc.HIERARCHY_MAP[args.pixel_hierarchy]
    results_root = _resolve_results_root(out_data, args.results_version)

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"[{mode}] link person-days draws")
    print(f"  results_root      = {results_root}")
    print(
        f"  compiled source   = {src_data.root / 'erf-scratch' / 'compiled-person-days'}"
    )
    print(f"  subset hierarchies= {subset_hierarchies}")
    print(f"  scenarios         = {scenarios}")
    print(f"  draws             = {len(draws)}\n")

    draw_maps = _draw_maps_by_scenario(cd_data, scenarios, args.variable, draws)

    counts = LinkCounts()
    for scenario in scenarios:
        for subset_hierarchy in subset_hierarchies:
            out_root = (
                results_root / subset_hierarchy / f"temperature_person_days_{scenario}"
            )
            _link_draws(
                src_data=src_data,
                out_root=out_root,
                subset_hierarchy=subset_hierarchy,
                scenario=scenario,
                draw_map=draw_maps[scenario],
                counts=counts,
                dry_run=args.dry_run,
            )

    _print_summary(counts, dry_run=args.dry_run)
    if counts.conflicts or counts.missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
