# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
- A `yearly` anomaly scheme for multiplicative variables in `generate scenario_daily`
  (`--anomaly-scheme`, default `monthly` = historical behavior): daily values are
  divided by the reference-period annual-mean rate, which is equivalent to raking each
  year's total to the reference level and distributing it over days by the GCM's own
  daily shape. Avoids the Jensen inflation of near-zero dry-season monthly denominators
  in the `(target+1)/(reference+1)` construction. Also a `--reference-years` option for
  the GCM reference window (default `2019-2023`, the current behavior). (CLIMATE-30)
- A `yearly-delta` variant of the yearly scheme that additionally removes the Jensen
  bias of the noisy window-mean denominator by dividing by `1 + Var(mean)/mean^2`;
  corrects the mean and leaves the CV unchanged. (CLIMATE-30)
- The yearly schemes guard the zero-denominator edge: a cell whose reference window
  has no rain at all forecasts zero rain rather than inf/NaN; a missing (NaN)
  reference still propagates NaN. (CLIMATE-30)
- Dry-run preview for cluster runners: `jobmon_utils.run_parallel_maybe_dry_run`
  and a `--dry-run/--no-dry-run` option (`clio.with_dry_run`) threaded through the
  runners; prints sbatch-like job previews instead of submitting. (CLIMATE-21)
- GBD-hierarchy versioning for the special stage: `gbd_2021`/`gbd_2025` hierarchies,
  `GBD_HIERARCHIES`, and a `--hierarchy` option (default `gbd_2023`) on the
  person-days runners. (CLIMATE-22)
- Historical run mode for `temperature_zone` and the person-days steps. (CLIMATE-22)
- `special download_era5_uncertainty` step: downloads whole-year ERA5
  `reanalysis` + `ensemble_spread` 2m-temperature files (2024/2025) from CDS,
  matching Katrin Burkart's `era5_{product_type}_{variable}_{year}.nc` layout, to
  fill the GBD temperature-uncertainty gap. (CLIMATE-22)
- docs: a `Running the pipeline` page (`docs/running.md`) for the operational side the
  methodology page does not cover — stage order and the two dependencies that are easy to
  miss, the three non-obvious `scenario_inclusion` behaviours (it surveys *all* CMIP6
  variables at once, so a root holding one variable drops the rest from the inclusion
  metadata; a member with an incomplete extract silently drops out rather than erroring; it
  runs inline rather than fanning out), that daily scenario fields are transient by default
  because `scenario_annual` calls `generate_scenario_daily_main(..., write_output=False)`, a
  table of which outputs honour `-o` and the one that does not, measured output sizes
  (~823 MB per daily year against 3.2–3.4 MB per annual file), job shapes per step, and
  measured GCS throughput. (CLIMATE-29)
- docs: an `Output schema` section for the temperature person-days product, covering the
  three output tiers and their path templates, the index levels and the `year` /
  `year_id` difference between the block and compiled tiers, that the 800 columns are a
  daily-temperature axis labelled by bin lower edge, that the first and last bin of both
  the temperature and zone axes are **unbounded** catch-alls, the person-days units and
  what can and cannot be inverted from them, that scenario/member/hierarchy/draw are
  path-encoded only, that the compiled tier interleaves aggregate with most-detailed
  locations so naive summation multiply-counts, and that the 100-draw axis resolves onto
  fewer distinct model members than draws. (CLIMATE-17)
### Changed
- `extract cmip6` fans out one job per ensemble member instead of one per
  (source, experiment). Member counts are very uneven — MIROC6 publishes 50 `pr` members
  where most sources publish one — so three jobs carried fifty times the work of a typical
  one. More importantly `extract_cmip6_main` re-raises inside the member loop, so a member
  the encoding guard rejects abandoned every member behind it in the same job, with
  `max_attempts=1` and no resume; per-member jobs contain a rejection to the member that
  caused it. The runner now also skips members already on disk unless `--overwrite`, so a
  resumed run does not redo whole groups. A `pr` re-extract is 295 jobs, up from 66. The
  member space is not a cartesian product, so it is enumerated from the metadata and passed
  as `flat_node_args`, the pattern `generate scenario_daily` already uses. (CLIMATE-29)
- `draws` now prefers historical ERA5 over scenario data for overlapping years.
  (CLIMATE-22)
- Moved the storage root (`MODEL_ROOT`) to `/mnt/share/geospatial/climate/`. (CLIMATE-22)
- Moved CI workflows off a departed maintainer's personal `GH_TOKEN` to the built-in
  `GITHUB_TOKEN`. Auto dependency-bump PRs no longer trigger CI, and the cookiecutter
  template-sync workflow is retired (daily schedule removed); reviving it needs a
  non-personal token. (CLIMATE-24)
### Fixed
- `total_precipitation` was inflated ~1.5-1.6x across the whole historical record.
  ERA5 stamps an accumulation window by its **end**, so day D's window (forecast steps
  01-24) has its closing sample timestamped `00:00` of day D+1 — these are interval
  labels, not instants. The `groupby("time.date")` buckets therefore straddled two
  windows: each opened with the *previous* day's completed total and never saw its own.
  Collapsing ERA5-Land with `daily_max` consequently returned `max(yesterday's total,
  today's 23-hour partial)`. Both ERA5 datasets now collapse with an interval-aware
  `resample(closed="right", label="left")`, which bins on `(D 00:00, D+1 00:00]` labelled
  `D` — one whole window per day — via the new `utils.daily_accumulation_last` (ERA5-Land,
  cumulative) and `utils.daily_accumulation_sum` (single-levels, hourly increments).
  `last` rather than `max` because the two agree only while the window rises
  monotonically, which int16 packing does not guarantee. Reported by Anna Rutherford
  (EOD/WASH). (CLIMATE-29)
- The encoding guard rejected values the writer would have stored correctly. It compared
  the **unrounded** `(value - offset) / scale` against the dtype's limits, but `to_netcdf`
  packs with round-half-to-even, so anything within half a quantum of a limit rounds back
  into range. Floating-point noise in the source GCM fields — CMCC-CM2-SR5, CMCC-ESM2 and
  GISS-E2-1-G all report a `pr` minimum a few times 1e-17 kg m-2 s-1 below zero, which is
  -6.19e-11 stored and rounds to `0` — tripped it, and eight `pr` members failed to extract
  with the guard reporting a stored value of `-0` as out of range. The guard now rounds
  before comparing, so its cut sits where the writer's behaviour actually changes: half a
  quantum below zero, where the code rounds to `-1` and casts to 65535, which is
  `_FillValue`. That is the wrap worth refusing, and it is still refused, as is the
  symmetric case half a quantum above the ceiling. Non-finite bounds bypass the rounding so
  an all-NaN field still fails with the guard's own message rather than a bare
  `cannot convert float NaN to integer`. (CLIMATE-29)
- CMIP6 `pr` is now encoded **unsigned**. At `int16`/`1e-6` the ceiling was 2831 mm/day, and
  a real member exceeded it: `ACCESS-CM2 ssp585 r1i1p1f1` peaks at 3108 mm/day, needing code
  35978 against 32767, and failed the guard partway through a re-extract. Precipitation is
  never negative, so half the signed range was never usable; `uint16` reaches **5662 mm/day
  at the same 0.0864 mm/day quantum**. Coarsening `encoding_scale` instead was rejected —
  `2e-6` gives a 0.173 mm/day quantum, worse than the 0.1 mm/day the daily product
  downstream stores, which would make the input the limiting factor. `_FillValue` moves to
  the top of the unsigned range (65535), so zero rainfall encodes as `0` and cannot be
  decoded as missing — the trap in the alternative of shifting `add_offset` on a signed
  dtype, where zero would land adjacent to `-32767`. `CMIP6Variable` carries an
  `encoding_dtype` (default `int16`; only `pr` overrides it) and the guard checks the
  bounds of the declared dtype, so an unsigned variable now rejects negatives outright
  rather than wrapping them. The other seven variables are unchanged — temperatures need
  the sign. Both readers of these files (`scenario_daily`, `scenario_inclusion`) go through
  xarray, which decodes native unsigned types transparently. (CLIMATE-29)
- CMIP6 `pr` was extracted as int16 with `scale_factor=1e-9`. `pr` is a flux in
  kg m-2 s-1, so that put the representable ceiling at **2.83 mm/day**: anything wetter
  wrapped modulo 65536 and decoded as garbage, including negative precipitation. Measured
  on a sampled year, 26.4% of cells were wrong and 12.4% were negative, across all 295
  `pr_*.nc` files. `scenario_daily` reads these files directly and precipitation is
  multiplicative, so the anomaly was a ratio of two corrupted quantities. The scale is now
  `1e-6` (ceiling 2831 mm/day) and the extract refuses to write values its encoding cannot
  represent instead of wrapping them silently. **The existing extracted files are not
  repaired by this change** — they stay corrupt until re-extracted. (CLIMATE-29)
- `extract cmip6` wrote its output path with the experiment and variable transposed, so a
  re-extract would have produced `ssp126_pr_<member>.nc` while the generate stage looks up
  `pr_ssp126_<member>.nc` — new files invisible to the pipeline consuming them. Present
  since the `source` parameter was dropped in `0de5beb0`. (CLIMATE-29)
- The ERA5-Land collapse silently accepted a day whose closing sample was missing.
  `Resample.last` defaults to `skipna=True`, so it stepped back to hour 23 and returned
  the 23-hour partial — the incomplete window the collapse exists to remove — one pixel at
  a time. It now passes `skipna=False`. **This is not detectable in the output:**
  ERA5-Land NaNs are filled from the interpolated single-level field, which is the
  mechanism that supplies ocean pixels, so such a pixel ends up carrying a complete 0.25°
  value rather than a truncated 0.1° one and never reaches `validate_output`. Preferring
  the coarse-but-whole value is the intent; detecting the substitution would need a
  separate check against the sea mask. A survey of all 1778 `total_precipitation` extracts
  found every day 1950–2023 in possession of its closing *timestamp*, so the remaining
  exposure is a transient NaN at a land pixel. (CLIMATE-29)
- `extract cmip6` dropped the source from its output filename. When `0de5beb0` collapsed
  `extracted_cmip6_path` from `(variable, experiment, source, member)` to
  `(variable, experiment, gcm_member)`, the reader was updated to match —
  `ClimateData.get_gcms` returns `<source>_<variant>` — but the writer kept passing the bare
  `member_id`. A CMIP6 `member_id` is unique only *within* a source, so `r1i1p1f1` is shared
  by 24 of the extracted sources in ssp126 and 30 in ssp585: a `pr` re-extract's 295
  extractions would have collapsed onto **171 filenames, 124 of them overwriting each
  other**, concurrently at `concurrency_limit=50` — and not one of those names is what
  `scenario_daily` looks up, so every file would have been invisible to the stage consuming
  it. `88b3dc7` fixed that commit's transposed arguments but not its dropped source. The key
  is now built by a shared `data.gcm_member_id`, called by both the writer and
  `get_gcms`. The 295 files on disk predate `0de5beb0` and are named correctly. Caught by a
  one-member validation extract, before the full re-extract ran. (CLIMATE-29)
- `extract cmip6` deleted the file it had failed to replace. The failure handler unlinked
  `out_path` unconditionally, but the encoding guard raises *before* the write begins, so
  a rejected extract destroyed the previous file and wrote nothing in its place — leaving
  neither a corrected file nor the one it was meant to supersede. It now clears only a
  file the invocation actually started writing. A live risk for the 295-file `pr`
  re-extract with `--overwrite`, where one too-wet GCM would have taken its old file with
  it. (CLIMATE-29)
- Because an accumulation window is closed by the following hour, generating a month now
  also reads the first sample of the *next* month — for December, of the next year's
  January — and trims the out-of-month bins the collapse produces at each end. A missing
  look-ahead extract raises rather than silently shortening the final day, so
  regenerating **2023 requires an ERA5 January 2024 extract**. One exists in the GBD-2025
  pull at `/mnt/share/geospatial/climate/extracted_data/era5/` (both datasets, all of 2024
  and 2025, `expver='0001'` final ERA5). Files pulled through the newer CDS API carry
  `number` and `expver` coordinates and are stored float32 rather than packed int16, so the
  look-ahead now normalises coordinates before concatenating — `xr.concat` refuses to join
  datasets whose coordinates differ. That format split is **not** chronological, as this
  entry first described it: it tracks download date. `valid_time` covers 1950–1989 and
  2024, `time` covers 1990–2023, so there are two seams — 1989→1990 and 2023→2024 — and the
  full 1950–2023 regeneration crossed both, in opposite directions. (CLIMATE-29)
- The look-ahead sample was accepted on trust. `.isel(time=[0])` assumed the next month
  opens on midnight — ERA5-Land `1950_01` opens at 01:00, so a file of that shape exists in
  the archive, and accepting one leaves the month's final day on its 23-hour partial. And
  `xr.concat` defaulted to `join="outer"`, so a look-ahead on a different grid would have
  widened both months onto a union grid and NaN-filled the difference, which the
  single-level fill then hides from `validate_output`. A non-midnight stamp now raises and
  the concat is `join="exact"`. That guards the single-level datasets; `load_variable`
  overwrites ERA5-Land's coordinates from `cdc.ERA5_LAND_*`, so a genuine land-grid change
  would still be relabelled first — the real land files differ by ~1e-5° across the format
  change, which is why that overwrite exists. (CLIMATE-29)
- That look-ahead could not be produced through either CLI. Both ERA5 extract tasks bound
  `--year` to `HISTORY_YEARS`, which stops at the last history year, so the year the
  look-ahead needs was rejected by click and the failure message named a command nothing
  would accept; the 2024 files that let the first full run succeed came from a separate
  GBD-2025 pull, not from this repo. The single-job tasks now span a new
  `cdc.EXTRACT_YEARS` — the history range plus the following year — and the message prints
  the two commands to run. `cdrun extract era5` deliberately keeps the narrower range:
  `-y ALL` resolves to every choice and the runner decides what to fetch by file
  existence, so widening it would add a year of downloads to a step that already
  re-downloads terabytes when run with its defaults. (CLIMATE-29)
- docs: the ERA5 spatial-harmonization section claimed the 0.25° single-level data is
  upsampled with nearest-neighbor interpolation. It is bilinear — the two
  `interpolate_to_target_latlon(..., method="linear")` calls in
  `generate_historical_daily_main`; nearest is used only for sea-surface temperature, on
  the `method="nearest"` branch of the same function, which has no ERA5-Land counterpart.
  The stale wording appears to come from `interpolate_to_target_latlon`'s default, which
  both call sites override. Also documented that ocean pixels are therefore supplied entirely by the
  upsampled 0.25° field, so the product's effective resolution is 0.1° only over land.
- person-days: zero-fill gridded-population nodata (NaN) before `compute_person_days`,
  so NaN pixels no longer poison output cells and zero out small/coastal locations
  (e.g. American Samoa). Latent bug exposed by a population-model nodata-encoding
  change. (CLIMATE-22)
- Retyped for the newer `pandas-stubs`/`numpy`/`numba` pulled in by the monthly
  dependency bump: dropped three `type: ignore`s the improved stubs made redundant,
  cast the `rioxarray` `write_crs` result, the `unstack` result, and the unstacked
  `MultiIndex` columns (passing `droplevel(0)` explicitly, 0 being its existing
  default), and coerced the CMIP6 `member_id` key to `str`. No runtime behavior
  change. The three dropped ignores are still required by the *previous* lock, so
  these fixes only type-check alongside the updated `poetry.lock`.
- Cleared accumulated mypy/ruff/formatting debt so the `pre-commit` CI job passes
  again. Also removed a dead post-submit serial loop in `grid_plots` whose
  `Path.exists()` misuse (positional arg → `TypeError`) had masked that it was a
  no-op: the parallel workers already write every page, and its one live branch
  regenerated figures with `write=False` and discarded them.
- Docs deployment: `build_docs` now authenticates with the built-in `GITHUB_TOKEN`
  (+ `contents: write`) instead of a dead personal token, so `mkdocs gh-deploy` can push
  `gh-pages` again; also fixed a malformed job `if:` expression. (CLIMATE-24)
