# Running the pipeline

Operational notes for actually executing the stages: what depends on what, where each step
writes, how big the outputs are, and which behaviours are surprising enough to cost you a
run. For the science — sources, bias-correction and downscaling maths, ensembling — see
[Methodology](methods.md).

Figures here were measured on the IHME cluster against the production archive. They are
order-of-magnitude guides for capacity planning, not guarantees.

## Order of operations

The forecast chain, for one target variable:

```
extract cmip6  ->  generate scenario_inclusion  ->  generate scenario_annual  ->  generate draws  ->  aggregate
```

`generate historical_daily` and `generate historical_reference` feed the same chain from the
ERA5 side and are independent of the CMIP6 extracts.

Two dependencies are easy to miss:

- **`scenario_inclusion` must be re-run whenever the CMIP6 extracts change.** It surveys the
  extracted files and writes `results/metadata/scenario_inclusion_metadata.parquet`, which is
  what `ClimateData.get_gcms` reads to decide which ensemble members exist. Stale inclusion
  metadata silently produces the wrong member list.
- **`scenario_annual` needs `historical_reference` for its target variable.** The scenario
  field is built by applying a CMIP6 anomaly to the ERA5 reference climatology, so a missing
  or stale `reference.nc` propagates straight into every forecast year.

## `scenario_inclusion` has three non-obvious behaviours

**It surveys every CMIP6 file in the root at once.** `generate_scenario_inclusion_main`
globs `extracted_cmip6 / "*.nc"` across all variables, and the inclusion table it writes has
one column per variable. A root holding only one variable's extracts therefore produces
inclusion metadata missing every other variable, and `get_gcms` — which selects by column —
fails for those. If you extract one variable into a fresh root, make the other variables
reachable there too (symlinks are enough) before running this step.

**A member with an incomplete extract silently drops out.** Inclusion requires
`year_start <= 2020 and year_end >= 2099`, that the file opens, and that it has no duplicate
lat/lon/time coordinates — and then that the member is valid in *all three* experiments. A
member whose extract failed partway is not an error here; it just disappears from the member
list, and every downstream count shrinks with no warning. Verify an extract before trusting
the inclusion table built from it.

**It is not a fan-out.** The same function is registered in both `RUNNERS` and
`TASK_RUNNERS`, so `cdrun generate scenario_inclusion` runs inline rather than submitting
jobs. It parallelises within the process via `--num-cores` (default 10).

**Filenames are parsed, not just matched.** Metadata comes from
`path.stem.split("_")` unpacked as `(variable, scenario, source, variant)`, so a CMIP6 file
whose name does not have those four fields will corrupt or break the survey.

## Daily scenario fields are transient by default

`generate_scenario_annual_main` calls
`generate_scenario_daily_main(..., write_output=False)` and pipes the result straight into
the annual transform, computing it in memory. **In the normal chain, daily scenario data
never touches disk.**

Running `cdrun generate scenario_daily` standalone is the only path where `write_output`
keeps its default of `True`. That writes one file per (experiment, year, member) at roughly
823 MB each — for one variable across three experiments, about **6.5 TB**. Only do it if you
specifically need daily fields on disk.

The tradeoff: because daily is recomputed per annual variable rather than cached, an annual
job that derives from several source variables recomputes each of their daily fields. That
is CPU spent to avoid terabytes of intermediate storage.

## Where each step writes, and whether `-o` is honoured

`--output-directory` / `-o` sets the root for most outputs, but **not all**.

| output | path | honours `-o`? |
|---|---|---|
| daily historical results | `<root>/results/daily/historical/...` | yes |
| historical reference | `<root>/results/daily/historical/<var>/reference.nc` | yes |
| CMIP6 / ERA5 extracts | `<root>/extracted_data/...` | yes |
| inclusion metadata | `<root>/results/metadata/...` | yes |
| raw **annual** results | `<root>/results/annual/raw/...` | yes |
| raw **daily** scenario results | `AGGREGATE_ROOT / "erf-scratch"` | **no — hardcoded** |

`ClimateData.raw_daily_results` returns `cdc.AGGREGATE_ROOT / "erf-scratch"` regardless of
the root, carrying a `NOTE:` marking it temporary. Consequences worth knowing before you
rely on `-o`:

- A sandboxed `scenario_daily` is not currently possible — its output always lands in the
  shared aggregate area.
- That directory has previously held per-scenario **symlinks** pointing at other
  filesystems, to spread the load. They can be dangling, in which case a run with
  `mkdir(parents=True)` will recreate the target and quietly write terabytes somewhere you
  did not choose. Check `readlink` on each scenario directory before a large run.

Treat this row as a known wart rather than intended design. The other paths are all rooted
on `-o` as you would expect.

## A measure-filtered `aggregate pixel` run needs its own `--agg-version`

`--agg-measure` lets you aggregate one measure instead of all ten. The saving is large
(~9x per task for a single measure), but it is **safe only under a convention the code does
not enforce**, and getting it wrong produces silently incomplete aggregates rather than an
error.

Two facts combine badly:

1. `ClimateAggregateData.raw_results_path(version, hierarchy, block_key, draw)` has **no
   measure component**, so a filtered run and an all-measure run write to the *same* path.
2. The `pixel` runner skips work whose output already exists:

   ```python
   if not ca_data.raw_results_path(agg_version, h, b, d).exists():
       jobs.append((h, b, d))
   ```

So running `--agg-measure total_precipitation` and later re-running the **same
`agg_version`** without the flag skips every block already written, and the missing nine
measures are never computed. Nothing errors, the controller exits 0, and
`aggregate hierarchy` then loads raw results filtered to a measure that has no rows.

**The rule: a version is either all-measure or filtered, never a mixture.** Give each
filtered run its own `agg_version`, following the existing `YYYY_MM_DD_<PURPOSE>` naming
(e.g. `2026_08_21_PRECIP_TEST`). The skip check stays correct within any single version.

**Hierarchy, by contrast, *is* part of the key**, so hierarchies can safely be added to an
existing version later — a run of `gbd_2023` today and `lsae_1209` next week write to
different paths and will not skip each other.

## Measured output sizes

| product | per file | notes |
|---|---|---|
| daily, one year, one variable | ~823 MB | 365 x 1800 x 3600, `int16`, 0.1 mm precision |
| annual, one year/member | 3.2–3.4 MB | 1800 x 3600, `int16`, 10 mm precision |
| CMIP6 `pr` extract, one member | ~0.4–3 GB | varies with each model's native grid |

For one variable across the three forecast scenarios, annual output is about **25 GB** in
total — roughly 250x smaller per file than the daily fields it is derived from, which is why
the daily step is transient.

## Job shapes

| step | jobs | cores / memory / runtime | attempts | concurrency |
|---|---|---|---|---|
| `extract cmip6` (one variable) | one per ensemble member | 1 / 10 G / 3000m | 1 | 50 |
| `generate historical_daily` | one per (variable, year) | 5 / 150 G / 240m | 2 | none set |
| `generate historical_reference` | one per variable | 1 / 80 G / 30m | 1 | none set |
| `generate scenario_annual` | one per (variable, scenario, year, member) | 1 / 100 G / 60m | 1 | none set |

`extract cmip6` fans out per member rather than per (source, experiment) because member
counts are very uneven — one model may publish 50 members where most publish one — and
because a failure aborts only the member that caused it. `scenario_annual` enumerates
`FORECAST_YEARS x get_gcms(...)` per (variable, scenario), so for one variable with 35
included members it is roughly 8,000 jobs; the historical scenario instead uses
`HISTORY_YEARS` with the single member `era5`.

Both of these runners **filter out work already on disk** unless `--overwrite` is passed, and
print a `to_run` / `complete` summary first, so an interrupted run resumes cheaply. Neither
sets a concurrency limit for `scenario_annual`, so the full fan-out hits the scheduler at
once.

## Preview before submitting

Runners built on `jobmon_utils.run_parallel_maybe_dry_run` take `--dry-run`, which prints
sbatch-like previews and the job count instead of submitting. Use it before any large
fan-out — it is the cheapest way to catch a wrong `-o`, an unintended `ALL`, or a job count
an order of magnitude off what you expected.

## Reading from Google Cloud Storage

CMIP6 extracts stream from GCS-backed zarr stores. Measured throughput from the cluster is
roughly **55 MB/s** single-threaded. The extract's encoding guard reduces each array's min
and max before writing, so budget **two** passes over each member's source data — the guard's
traversal and `to_netcdf`'s.

The guard refuses to write values the declared 16-bit encoding cannot represent, rather than
letting them wrap silently. A rejected member fails loudly with an actionable message naming
the offending value, and leaves any previous file intact. Treat an occasional rejection on an
unusually extreme model as designed behaviour rather than a bug.

There are two levers when it fires, and they are not equivalent:

- **`encoding_dtype`.** A variable that cannot be negative should be `uint16`, which puts the
  whole 65535-code range above zero — double the ceiling of `int16` at *identical*
  resolution. `pr` uses this. `_FillValue` sits at the top of the unsigned range, so zero
  encodes as `0` and cannot be confused with missing data.
- **`encoding_scale`.** Coarsens the quantum to buy ceiling. Check what it costs before
  reaching for it: the daily product stores precipitation at 0.1 mm/day, so a `pr` input
  quantum above that would make the input the limiting factor rather than the output.

Because the encoding is written into every file, settle any change **before** a bulk
re-extract, not after — the files carry whatever was declared when they were written, and
the only way to change it is to extract them again.
