# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
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
- Extended `HISTORY_YEARS` through 2025; `draws` now prefers historical ERA5 over
  scenario data for overlapping years. (CLIMATE-22)
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
- Because an accumulation window is closed by the following hour, generating a month now
  also reads the first sample of the *next* month — for December, of the next year's
  January — and trims the out-of-month bins the collapse produces at each end. A missing
  look-ahead extract raises rather than silently shortening the final day, so
  regenerating **2023 requires an ERA5 January 2024 extract**, which does not currently
  exist. (CLIMATE-29)
- docs: the ERA5 spatial-harmonization section claimed the 0.25° single-level data is
  upsampled with nearest-neighbor interpolation. It is bilinear
  (`generate/historical_daily.py:219`, `:228`); nearest is used only for sea-surface
  temperature (`:195`), which has no ERA5-Land counterpart. The stale wording appears
  to come from `interpolate_to_target_latlon`'s default, which both call sites
  override. Also documented that ocean pixels are therefore supplied entirely by the
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
