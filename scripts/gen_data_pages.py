"""Generates a comprehensive report of the data stored in the Climate Database."""

from pathlib import Path

import mkdocs_gen_files
import pandas as pd

from climate_data.data import ClimateData

nav = mkdocs_gen_files.Nav()  # type: ignore[attr-defined, no-untyped-call]
doc_root = Path()

cdata = ClimateData()

# Extracted data

main_header_content = f"""# Climate Database

The Climate Database contains a variety of data extracted from different sources. This page provides an overview of the
data layout, the sources we've extracted data from, the data harmonization and downscaling process, and an overview of
the available climate variables.

## Data Organization

The root of the climate data is located at `{cdata.root}`. There are several
subdirectories in the root directory, but only the extracted data directory `{cdata.extracted_data.stem}` and the results
directory `{cdata.results.stem}` are relevant to users of the database.

The file tree with subdirectories is as follows:

```
{cdata.root}/
├── {cdata.extracted_data.stem}/
│   ├── {cdata.extracted_era5.stem}/
│   │   └── {{ERA5_DATASET}}_{{ERA5_VARIABLE}}_{{YEAR}}_{{MONTH}}.nc
│   ├── {cdata.extracted_cmip6.stem}/
│   │   └── {{CMIP6_VARIABLE}}_{{CMIP6_EXPERIMENT}}_{{CMIP6_SOURCE}}_{{VARIANT}}.nc
│   └── _Other Data Sources_/
└── {cdata.results.stem}/
    ├── {cdata.annual_results.stem}/
    │   └── {{SCENARIO}}/
    │       └── {{ANNUAL_VARIABLE}}/
    │           └── {{YEAR}}_{{DRAW}}.nc
    ├── {cdata.daily_results.stem}/
    │   └── {{SCENARIO}}/
    │       └── {{DAILY_VARIABLE}}/
    │           └── {{YEAR}}.nc
    └── {cdata.results_metadata.stem}/

```

The file patterns will be explained in more detail in the following sections.
"""

extraction_header_content = """## Extracted Data

The two primary data sources are historical climate data from the European Centre for Medium-Range Weather Forecasts (ECMWF)
ERA5 dataset and climate forecast data from the Climate Model Intercomparison Project Phase 6 (CMIP6). There are also some
additional data sources that have been extracted to serve as covariates in a forthcoming downscaling model.
"""

era5_page_content = f"""### ERA5 Data

The [ECMWF Reanalysis v5 (ERA5)](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) is the
fifth generation ECMWF atmospheric reanalysis of the global climate covering the period from January 1950 to present. ERA5 is produced
by the Copernicus Climate Change Service (C3S) at ECMWF. There are three datasets of note:

  - [The Complete ERA5 global atmospheric reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-complete?tab=overview):
    This dataset contains a wide range of atmospheric, land, and oceanic climate variables on a regular latitude/longitude grid at a roughly 31km
    resolution. Additionally it splits the atmosphere into 137 pressure levels starting at the Earth's surface and extending to a height of 80km.
    We do not typically extract from this source as we generally have no need of the oceanic data or the data above the Earth's surface.
  - [ERA5-Land hourly data from 1950 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview):
    This dataset contains land surface variables at the native model resolution of 9km. Variables not defined over the land surface (such as
    sea surface temperature) are not present in this dataset. Additionally, this dataset sometimes misses some land area, especially islands and
    regions. Because of its detailed resolution, this is the model we prefer to extract from wherever possible.
  - [ERA5 hourly data on single levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview):
    This dataset contains a wide range of atmospheric, land, and oceanic climate variables on a regular latitude/longitude grid at a roughly 31km.
    This dataset is similar to the complete ERA5 dataset, but it only contains data at the Earth's surface and at a few fixed pressure levels, making
    it significantly smaller and faster to work with. This is the dataset we use to supplement the ERA5-Land data over regions where the land data is
    missing or incomplete. We also use this dataset for variables that are not available in the ERA5-Land dataset.

#### Storage and Naming Conventions

  - Storage Root: `{cdata.extracted_era5}`
  - Naming Convention: `{{ERA5_DATASET}}_{{ERA5_VARIABLE}}_{{YEAR}}_{{MONTH}}.nc`
    * `{{ERA5_DATASET}}`: One of `reanalysis-era5-land`, or `reanalysis-era5-single-levels`.
    * `{{ERA5_VARIABLE}}`: The variable being extracted (variable names can be found on the pages linked above).
    * `{{YEAR}}` and `{{MONTH}}`: The year and month of the data being extracted.

"""

inclusion_metadata = pd.read_parquet(
    Path(__file__).parent / "scenario_inclusion_metadata.parquet"
)
source_table = inclusion_metadata.groupby("source").sum()

total_counts = pd.concat(
    [
        (source_table.T > 0).sum(axis=1).rename("Source Count"),
        source_table.T.sum(axis=1).rename("Variant Count"),
    ],
    axis=1,
)


cmip_page_content = f"""### CMIP6 Data

The [Climate Model Intercomparison Project Phase 6 (CMIP6)](https://en.wikipedia.org/wiki/Coupled_Model_Intercomparison_Project)
is a collaborative effort to compare climate models across the globe. The data is organized into
different variables, scenarios, and sources.

#### Model Inclusion

We use a subset of the CMIP6 data in our analysis following a [model evaluation published in Nature](https://www.nature.com/articles/s41597-023-02549-6)
that determined which models to include based on their performance. Our inclusion criteria are as follows:

  1. The model must be flagged as "Yes" in the `Included in 'Model Subset'?` column in the [evaluation table](https://www.nature.com/articles/s41597-023-02549-6/tables/3).
  2. The model must have daily results (which concretely means there is either a `day` or `Oday` table_id associated with the model).
  3. The model must make estimates for the three scenarios we are interested in: `ssp126`, `ssp245`, and `ssp585`.
  4. The model must cover the time range 2019-2099. We project out to 2100, and many models run a 2100 year, but some stop at 2099. They are incorporated here with their last year repeated.

!!! warning "Model Inclusion Caveats"

    The extraction criteria does not completely capture model inclusion criteria as it does not account for the year range available in the data.
    This determination is made when we process the data in later steps. See [the scenario inclusion stage](#pipeline-stages) of the processing
    pipeline for more detail.

#### Data Availability

The following tables show the number of unique models available for each variable.

=== "Model Count"

    {"\n\t".join(total_counts.to_markdown().split("\n"))}

=== "Source Breakdown"

    {"\n\t".join(source_table.to_markdown().split("\n"))}

#### Storage and Naming Conventions

  - Storage Root: `{cdata.extracted_cmip6}`
  - Naming Convention: `{{CMIP6_VARIABLE}}_{{CMIP6_EXPERIMENT}}_{{CMIP6_SOURCE}}_{{VARIANT}}.nc`
    * `{{CMIP6_VARIABLE}}`: The variable being extracted (variable names can be found in the [CMIP6 database](https://airtable.com/appYNLuWqAgzLbhSq/shrKcLEdssxb8Yvcp/tblL7dJkC3vl5zQLb)).
    * `{{CMIP6_EXPERIMENT}}`: The scenario being extracted (one of `ssp126`, `ssp245`, or `ssp585`).
    * `{{CMIP6_SOURCE}}`: The source model for the data. A source model is a particular model from a particular institution, e.g. `BCC-CSM2-MR`.
    * `{{VARIANT}}`: The variant of the model, which is a particular run of the model with specific initial and boundary conditions and forcing scenarios.

??? example "Variant Labels"

    For a given experiment, the realization_index, initialization_index, physics_index, and forcing_index are used to uniquely identify each simulation
    of an ensemble of runs contributed by a single model. These indices are defined as follows:

      - **realization_index** = an integer (≥1) distinguishing among members of an ensemble of simulations that differ only in their initial conditions (e.g.,
        initialized from different points in a control run). Note that if two different simulations were started from the same initial conditions, the
        same realization number should be used for both simulations. For example if a historical run with “natural forcing” only and another historical
        run that includes anthropogenic forcing were both spawned at the same point in a control run, both should be assigned the same realization. Also,
        each so-called RCP (future scenario) simulation should normally be assigned the same realization integer as the historical run from which it was
        initiated. This will allow users to easily splice together the appropriate historical and future runs.
      - **initialization_index** = an integer (≥1), which should be assigned a value of 1 except to distinguish simulations performed under the same conditions
        but with different initialization procedures. In CMIP6 this index should invariably be assigned the value “1” except for some hindcast and forecast
        experiments called for by the DCPP activity. The initialization_index can be used either to distinguish between different algorithms used to impose
        initial conditions on a forecast or to distinguish between different observational datasets used to initialize a forecast.
      - **physics_index** = an integer (≥1) identifying the physics version used by the model. In the usual case of a single physics version of a model, this
        argument should normally be assigned the value 1, but it is essential that a consistent assignment of physics_index be used across all simulations
        performed by a particular model. Use of “physics_index” is reserved for closely-related model versions (e.g., as in a “perturbed physics” ensemble)
        or for the same model run with slightly different parameterizations (e.g., of cloud physics). Model versions that are substantially different from
        one another should be given a different source_id” (rather than simply assigning a different value of the physics_index).
      - **forcing_index** = an integer (≥1) used to distinguish runs conforming to the protocol of a single CMIP6 experiment, but with different variants of
        forcing applied. One can, for example, distinguish between two historical simulations, one forced with the CMIP6-recommended forcing data sets
        and another forced by a different dataset, which might yield information about how forcing uncertainty affects the simulation.

"""

extraction_content = (
    f"{extraction_header_content}\n{era5_page_content}\n{cmip_page_content}"
)

processed_data_content = f"""## Processed Data

The processed data is stored in the `{cdata.results}` directory, organized by scenario, variable, and year.
We generally only generate annual results, as storing daily results for all models and all variables would be
prohibitively expensive.

### Storage and Naming Conventions

    - Daily Storage Root: `{cdata.daily_results}`
    - Naming Convention: `{{SCENARIO}}/{{DAILY_VARIABLE}}/{{YEAR}}.nc` (historical data only)
        - `{{SCENARIO}}`: Generally, only historical data is available at the daily level, so this will be `historical`.
        - `{{DAILY_VARIABLE}}`: The name of the variable being stored.
        - `{{YEAR}}`: The year of the data being stored.
    - Annual Storage Root: `{cdata.results}`
    - Naming Convention: `{{SCENARIO}}/{{ANNUAL_VARIABLE}}/{{YEAR}}.nc` or `{{SCENARIO}}/{{ANNUAL_VARIABLE}}/{{YEAR}}_{{DRAW}}.nc`

### Pipeline Stages

The processing pipelines turn the extracted [ERA5](#era5-data) and [CMIP6](#cmip6-data) data into a coherent set of
climate variables with a consistent resolution, time scale, and data storage format. The pipeline is run
with the `cdrun` command (see [Installation](../installation.md) for installation instructions). The pipeline
has the following steps:

  1.  **Historical Daily** (`cdrun generate historical_daily`): This processes the hourly ERA5-Land and ERA5-Single-Level
      data into a unified daily format, pulling the higher-resolution ERA5-Land data where available and filling in
      with ERA5-Single-Level data. Historical daily data is produced for all core variables (those not derived by
      computing annual summary statistics from the daily data).
  2.  **Historical Reference** (`cdrun generate historical_reference`): This produces a set of reference climatologies
      from the historical daily data. These reference climatologies are built by averaging the last 5 years of
      historical data grouped by month. For example, the reference climatology for January mean temperature is the
      average of all Januare mean daily temperatures from 2019 to 2023. These climatologies are used to bias-correct
      the scenario data by serving as a seasonally-aware reference point we can intercept shift to.
  3.  **Scenario Inclusion** (`cdrun generate scenario_inclusion`): This produces a set of metadata that determines
      which CMIP sources and variants are used to generate scenario draws. This is the second stage scenario determination.
      When we [extract CMIP6 data](####model-inclusion), we cannot determine the year range of the data until it is extracted.
      This stage determines which models are included based on the year range of the data and writes this information to a file
      in {cdata.results_metadata}.
  4.  **Scenario Daily** (`cdrun generate scenario_daily`): This produces scenario projections from the CMIP6 data by dynamical
      downscaling the daily data to the same resolution as the ERA5 data. Our downscaling process computes the absolute or
      relative anomaly of a forecast day from a CMIP6 model relative to what that model's average prediction in the month
      over the reference period (2019-2023). This anomaly is then downscaled using linear interpolation to the ERA5 grid.
      The downscaled anomaly is then applied back to the reference climatology, which adds in fine-scale detail and provides
      a seasonally-aware bias correction. This produces draw level estimates using a two-stage sampling method, first selecting
      a CMIP `source` model, and then selecting a `variant` from that model.  This sampling method ensures each model of the ensemble
      is equally represented despite the large difference in the number of variants produced by each source model.
      **NOTE**: This stage is not typically invoked on its own, as storing draw-level daily data for all models and all
      variables is prohibitively expensive. It is instead invoked indirectly as part of the scenario annual stage.
  5.  **Scenario Annual** (`cdrun generate scenario_annual`): This produces annual estimates of the climate variables.
      It invokes the scenario daily stage to produce daily data, then computes annual summaries of the daily data, saving
      only the annual summary.


###"""


content = f"{main_header_content}\n{extraction_content}\n{processed_data_content}"
with mkdocs_gen_files.open(doc_root / "index.md", "a") as fd:
    fd.write(content)
