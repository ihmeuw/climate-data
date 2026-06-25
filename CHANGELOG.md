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
### Changed
- Extended `HISTORY_YEARS` through 2025; `draws` now prefers historical ERA5 over
  scenario data for overlapping years. (CLIMATE-22)
- Moved the storage root (`MODEL_ROOT`) to `/mnt/share/geospatial/climate/`. (CLIMATE-22)
