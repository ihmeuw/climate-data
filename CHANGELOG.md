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
- `scripts/link_person_days_draws.py`: the person-days "step 4" draw symlinker, which
  links the compiled per-draw parquets into the results layout. It previously lived only
  in a personal `~/deploy` clone, so the forecast person-days product could not be
  reproduced from a clean checkout. `--results-version` is required and must already
  exist, `--dry-run` previews without writing, and every degenerate case (missing
  compiled source, an existing output pointing at a different GCM, an unresolvable
  annual draw, a draw map that disagrees between scenarios) aborts or exits non-zero
  rather than reporting a healthy-looking run. (CLIMATE-25)
- `--concurrency-limit` option (`clio.with_concurrency_limit`) on the
  `temperature_person_days` runner, capping how many tasks jobmon runs at once to keep
  write latency on shared storage manageable. (CLIMATE-25)
- `ClimateAggregateData(..., read_only=True)`, mirroring `ClimateData`, so building a
  manager purely to construct paths no longer creates directories. (CLIMATE-25)
### Changed
- Extended `HISTORY_YEARS` through 2025; `draws` now prefers historical ERA5 over
  scenario data for overlapping years. (CLIMATE-22)
- Moved the storage root (`MODEL_ROOT`) to `/mnt/share/geospatial/climate/`. (CLIMATE-22)
- Moved CI workflows off a departed maintainer's personal `GH_TOKEN` to the built-in
  `GITHUB_TOKEN`. Auto dependency-bump PRs no longer trigger CI, and the cookiecutter
  template-sync workflow is retired (daily schedule removed); reviving it needs a
  non-personal token. (CLIMATE-24)
- `cdrun special temperature_person_days` now caps concurrency at 1500 by default
  instead of inheriting jobmon's 10000 (effectively unthrottled), matching what
  production runs actually used. Pass `--concurrency-limit` to override. (CLIMATE-25)
### Fixed
- `cdrun special temperature_person_days --dry-run` no longer creates
  `<output-dir>/<hierarchy>/` and a `logs/` child on shared storage: the aggregate-data
  manager was constructed before `dry_run` was consulted, so a preview wrote. (CLIMATE-25)
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
