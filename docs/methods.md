# Constructing a comprehensive climate database from ERA5 estimates and CMIP6 forecasts

Understanding the health impacts of climate change requires high-resolution, temporally consistent climate data that spans both historical observations and future projections. This section describes the methodology used to construct a comprehensive climate database that combines historical reanalysis data from the European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 products with future climate projections from the Coupled Model Intercomparison Project Phase 6 (CMIP6). The resulting database provides daily and annual climate variables at a spatial resolution of 0.1° × 0.1° from 1950 to 2100, covering multiple climate scenarios that represent different socioeconomic and policy futures.

Our approach integrates two primary data sources: ERA5 reanalysis products for historical climate data and CMIP6 model outputs for future projections. The ERA5 data provides high-quality, observationally constrained estimates of past climate conditions, while CMIP6 offers a range of plausible future climate trajectories under different emission scenarios. We employ a downscaling and bias correction methodology to ensure spatial and temporal consistency between these datasets, while preserving the fine-scale climate patterns captured by the higher-resolution ERA5 data.

The database construction process involves several key steps: (1) harmonization of historical ERA5 data to create a consistent daily time series, (2) development of a reference climatology for bias correction, (3) downscaling and bias correction of CMIP6 projections, and (4) aggregation to annual time scales. The final database includes multiple climate variables relevant to health impacts, such as temperature, humidity, precipitation, and wind speed, all processed to ensure consistency across the entire temporal range.

To enable epidemiological analyses at administrative levels, we further process the gridded climate data through population-weighted aggregation. This process combines the climate variables with high-resolution population estimates derived from internal historical population data and projections, enhanced with built environment information from the Global Human Settlement Layer. The resulting population-weighted aggregates are calculated for the most detailed administrative boundaries available, ensuring that climate exposure estimates reflect the actual population distribution within each administrative unit. These aggregated estimates can then be directly linked to health outcome data for epidemiological analyses.

## ERA5 data sources

We utilized two ERA5 reanalysis products to construct our historical climate database: ERA5-Land and ERA5 on single levels. ERA5-Land provides hourly estimates at 0.1° × 0.1° resolution over land areas, while ERA5 on single levels offers global coverage at 0.25° × 0.25° resolution. This combination ensures both high spatial resolution over land and complete global coverage.

From these datasets, we extracted six key variables relevant to health impacts:
- 10-meter u-component of wind (eastward wind velocity)
- 10-meter v-component of wind (northward wind velocity)
- 2-meter dewpoint temperature
- 2-meter temperature
- Surface pressure
- Total precipitation

## CMIP6 data sources

For future climate projections, we analyzed output from CMIP6 models under three Shared Socioeconomic Pathway (SSP) scenarios: SSP1-2.6, SSP2-4.5, and SSP5-8.5. These scenarios represent a range of possible future emissions trajectories, from ambitious mitigation to high emissions.

We selected 42 CMIP6 models based on their performance in reproducing historical air temperature trends, transient climate response, and equilibrium climate sensitivity [https://www.nature.com/articles/s41597-023-02549-6#Sec2]. From these, we included models that met the following criteria:
- Daily temporal resolution
- Coverage of all three SSP scenarios
- Complete timespan from 2019-2099 (with 2099 values extended to 2100 where needed)

The final analysis incorporated 21 models, with multiple ensemble members from some models. The included variables were:
- uas (near-surface eastward wind)
- vas (near-surface northward wind)
- hurs (near-surface relative humidity)
- tas (near-surface air temperature)
- tasmin (daily minimum temperature)
- tasmax (daily maximum temperature)
- pr (total precipitation)

| source        |   hurs |   pr |   tas |   tasmax |   tasmin |   uas |   vas |
|:--------------|-------:|-----:|------:|---------:|---------:|------:|------:|
| ACCESS-CM2    |      3 |    2 |     2 |        3 |        2 |     1 |     1 |
| AWI-CM-1-1-MR |      1 |    1 |     1 |        1 |        1 |     1 |     1 |
| BCC-CSM2-MR   |      0 |    1 |     1 |        1 |        1 |     1 |     1 |
| CAMS-CSM1-0   |      0 |    1 |     1 |        1 |        0 |     0 |     0 |
| CMCC-CM2-SR5  |      1 |    1 |     1 |        1 |        1 |     0 |     1 |
| CMCC-ESM2     |      1 |    1 |     1 |        1 |        1 |     1 |     1 |
| CNRM-CM6-1    |      6 |    1 |     6 |        1 |        1 |     6 |     6 |
| CNRM-CM6-1-HR |      1 |    0 |     0 |        0 |        0 |     1 |     1 |
| CNRM-ESM2-1   |      3 |    1 |     4 |        1 |        1 |     3 |     3 |
| FGOALS-g3     |      0 |    0 |     4 |        4 |        4 |     0 |     0 |
| GFDL-ESM4     |      1 |    1 |     1 |        1 |        1 |     1 |     1 |
| GISS-E2-1-G   |      0 |    0 |     1 |        0 |        0 |     0 |     0 |
| IITM-ESM      |      1 |    1 |     1 |        0 |        0 |     1 |     1 |
| INM-CM4-8     |      1 |    1 |     1 |        1 |        1 |     1 |     1 |
| INM-CM5-0     |      1 |    1 |     1 |        1 |        1 |     1 |     1 |
| MIROC-ES2L    |      0 |    1 |     1 |        1 |        1 |     1 |     1 |
| MIROC6        |      0 |    3 |     3 |        3 |        3 |     3 |     3 |
| MPI-ESM1-2-HR |      2 |    2 |     2 |        2 |        2 |     2 |     1 |
| MPI-ESM1-2-LR |     10 |   10 |    10 |       10 |       10 |    10 |    10 |
| MRI-ESM2-0    |      1 |    5 |     1 |        5 |        5 |     1 |     1 |
| NorESM2-MM    |      1 |    1 |     1 |        1 |        1 |     0 |     0 |

## Model harmonization, forecast bias correction, and downscaling

To produce a consistent climate database spanning 1950-2100, we implemented a multi-stage process that harmonizes historical ERA5 data with future CMIP6 projections. The methodology ensures spatial and temporal consistency while preserving the fine-scale climate patterns captured by the higher-resolution ERA5 data.

The process begins with the creation of a historical daily database from ERA5 data, which serves as the foundation for subsequent steps. We then develop a reference climatology using the most recent five years of historical data (2019-2023) from the ERA5 daily database. For each climate variable, we:

1. **Load daily data** for each year in the reference period
2. **Compute monthly means** by averaging all days within each month
3. **Average across years** to produce a single monthly climatology that represents typical conditions for each month of the year

The resulting reference climatology captures the seasonal cycle of each variable while smoothing out interannual variability. This monthly climatology is particularly important for:

- Providing a consistent baseline for bias correction
- Preserving the seasonal patterns in the historical data
- Enabling seasonally-aware downscaling of CMIP6 projections

The reference climatology is stored in the same format as the daily data, with appropriate encoding scales to optimize storage. This ensures consistency in the data processing pipeline and facilitates the subsequent downscaling and bias correction steps.

The CMIP6 data is processed through a dynamical downscaling approach that preserves the relative changes in climate variables while adjusting for systematic biases. Finally, we aggregate the daily data to annual time scales, producing a comprehensive database that maintains consistency across the entire temporal range.

This methodology addresses several key challenges in climate data processing:
1. Spatial resolution differences between ERA5 and CMIP6 data
2. Systematic biases in CMIP6 model outputs
3. Temporal consistency between historical and future projections
4. Preservation of fine-scale climate patterns important for health impact assessment

The following sections detail each stage of this process, including the specific transformations, bias correction methods, and aggregation procedures employed.

### Historical daily variables

The historical daily database is constructed from ERA5 data through a series of transformations and harmonization steps. We process six key variables from both ERA5-Land and ERA5 single-level datasets:

1. **Unit conversions and height adjustments**:
   - Wind components (u and v) are scaled to 10-meter height
   - Temperature variables are converted from Kelvin to Celsius
   - Precipitation is converted from meters to millimeters
   - Surface pressure remains in its original units

2. **Daily aggregations**:
   - Temperature variables are aggregated to daily mean, maximum, and minimum values
   - Wind speed is calculated as the vector magnitude of u and v components
   - Relative humidity is derived from temperature and dewpoint temperature
   - Precipitation is processed differently for land and single-level datasets:
     - ERA5-Land: daily maximum (this is a cumulative variable in the land dataset)
     - ERA5 single-level: daily sum

3. **Spatial harmonization**:
   - ERA5-Land data (0.1° × 0.1°) is left in its native resolution
   - ERA5 single-level data (0.25° × 0.25°) is interpolated using nearest-neighbor interpolation
   - The two datasets are combined, with ERA5-Land data taking precedence over land areas

4. **Temporal processing**:
   - Data is processed year by year from 1950 to 2023
   - Each year's data is validated to ensure completeness and consistency
   - The final output is stored in NetCDF format with appropriate encoding scales to optimize storage

The resulting daily database serves as the foundation for both the reference climatology and the bias correction of CMIP6 projections. The high spatial resolution (0.1° × 0.1°) and consistent temporal coverage make it particularly suitable for health impact assessments.

### Reference climatology

The reference climatology serves as the baseline for bias correction and downscaling of CMIP6 projections. It is constructed using the most recent five years of historical data (2019-2023) from the ERA5 daily database. For each climate variable, we:

1. **Load daily data** for each year in the reference period
2. **Compute monthly means** by averaging all days within each month
3. **Average across years** to produce a single monthly climatology that represents typical conditions for each month of the year

The resulting reference climatology captures the seasonal cycle of each variable while smoothing out interannual variability. This monthly climatology is particularly important for:

- Providing a consistent baseline for bias correction
- Preserving the seasonal patterns in the historical data
- Enabling seasonally-aware downscaling of CMIP6 projections

The reference climatology is stored in the same format as the daily data, with appropriate encoding scales to optimize storage. This ensures consistency in the data processing pipeline and facilitates the subsequent downscaling and bias correction steps.

### Forecast daily variables

The forecast daily variables are produced through a dynamical downscaling approach that combines CMIP6 model outputs with the reference climatology. For each climate variable and CMIP6 model, we:

1. **Compute anomalies**:
   - Calculate the difference between daily CMIP6 values and the model's monthly mean during the reference period (2019-2023)
   - For additive variables (e.g., temperature), compute absolute differences
   - For multiplicative variables (e.g., precipitation), compute relative differences

2. **Downscale anomalies**:
   - Interpolate the anomalies to the ERA5 grid (0.1° × 0.1°) using linear interpolation
   - This preserves the large-scale climate change signal while enabling fine-scale detail

3. **Apply bias correction**:
   - Add (for additive variables) or multiply (for multiplicative variables) the downscaled anomalies to the reference climatology
   - This step ensures that the forecast values maintain the same statistical properties as the historical data

The resulting daily forecasts preserve both the climate change signal from the CMIP6 models and the fine-scale spatial patterns from the ERA5 data. This approach is particularly important for health impact assessment as it:

- Maintains the temporal consistency between historical and future projections
- Preserves the fine-scale climate patterns that influence local health outcomes
- Ensures that extreme events are properly represented in the downscaled projections

### Annual variables

The annual variables are produced by aggregating the daily data to annual time scales, with different transformations applied depending on the variable type:

1. **Basic climate metrics**:
   - Mean temperature: Annual average of daily mean temperatures
   - Mean high temperature: Annual average of daily maximum temperatures
   - Mean low temperature: Annual average of daily minimum temperatures
   - Wind speed: Annual average of daily mean wind speeds
   - Relative humidity: Annual average of daily mean relative humidity
   - Total precipitation: Annual sum of daily precipitation

2. **Threshold-based metrics**:
   - Days over 30°C: Annual count of days where mean temperature exceeds 30°C
   - Precipitation days: Annual count of days with precipitation exceeding 0.1mm

3. **Disease suitability metrics**:
   - Malaria suitability: Annual sum of daily temperature-based suitability scores
   - Dengue suitability: Annual sum of daily temperature-based suitability scores

Each variable is processed with appropriate encoding scales to optimize storage:
- Temperature variables: 0.01°C precision
- Precipitation: 10mm precision
- Count-based metrics: Integer values

These annual variables provide a comprehensive set of climate indicators relevant to health impact assessment, capturing both average conditions and extreme events that may influence health outcomes.

### Model ensembling strategy

To quantify uncertainty in future climate projections, we implement a random sampling approach that creates 100 equally-weighted model variants. For each draw:

1. **Model selection**: A CMIP6 model is randomly selected from the pool of 21 models, with each model having an equal probability of selection regardless of the number of ensemble members it provides.

2. **Variant selection**: For the selected model, a specific variant (ensemble member) is randomly chosen from its available members. This ensures that models with multiple ensemble members are not overrepresented in the final ensemble.

3. **Consistency across scenarios**: The same model-variant combination is used for all three SSP scenarios (SSP1-2.6, SSP2-4.5, and SSP5-8.5) to maintain internal consistency in the climate change signal.

This approach provides several advantages:
- Equal representation of each model's climate response, regardless of the number of ensemble members
- Preservation of model-specific climate patterns and feedbacks
- Quantification of structural uncertainty in climate projections
- Consistent representation of climate change signals across scenarios

The resulting 100 draws provide a robust sample of possible future climate trajectories, enabling uncertainty quantification in health impact assessments.

## Population-weighted climate variable aggregation

To enable epidemiological analyses at administrative levels, we implement a multi-stage process that combines high-resolution climate data with population estimates:

### Population dataset production

The population dataset is constructed at 100-meter resolution using a combination of census data and built environment information:

1. **Base population data**: We use 2020 census data from the United States and Mexico, disaggregated to the most detailed administrative level (admin 5) in these countries.

2. **Built environment integration**: The census population counts are divided by the built residential area from the Global Human Settlement Layer to calculate an occupancy rate.

3. **Spatial modeling**: The log occupancy rate is modeled as a linear function of:
   - Geospatial averages of built residential area
   - Log of nighttime lights

4. **Administrative-level adjustment (raking)**: The gridded population estimates are adjusted to match administrative-level totals:
   - For historical years (1975-2023): Adjusted to match Global Burden of Disease 2021 demographic estimates at the most detailed administrative level (admin 0, 1, or 2)
   - For future years (2024-2100): Adjusted to match Future Health Scenarios estimates
   - For years before 1975: The 1975 spatial pattern is held constant and adjusted to match administrative totals
   - For years after 2023: The 2023 spatial pattern is held constant and adjusted to match administrative totals

This approach produces a high-resolution population dataset that:
- Captures fine-scale population distribution patterns
- Maintains consistency with administrative-level census data
- Preserves temporal consistency in population estimates
- Enables accurate population-weighted climate exposure assessment

### Aggregation of pixel-level estimates to most-detailed administrative boundaries

The climate variables are aggregated to administrative boundaries through a weighted averaging process:

1. **Spatial alignment**: The 0.1° × 0.1° climate data is resampled to match the 100-meter population grid using nearest-neighbor interpolation.

2. **Population weighting**: For each administrative unit, we calculate:
   - Weighted climate value = Σ(population × climate value) / Σ(population)
   - Total population = Σ(population)

3. **Administrative boundaries**: We use the most detailed administrative boundaries available, ensuring that climate exposure estimates reflect the actual population distribution within each unit.

### Aggregation of most-detailed administrative levels to GBD

The most-detailed administrative units are further aggregated to Global Burden of Disease (GBD) locations:

1. **Hierarchical aggregation**: We maintain a consistent hierarchy of locations, from most-detailed administrative units up to global level.

2. **Population-weighted sums**: For each higher-level location, we calculate:
   - Weighted climate value = Σ(population × weighted climate value) / Σ(population)
   - Total population = Σ(population)

3. **Multiple hierarchies**: We support different administrative hierarchies (e.g., GBD 2021, FHS 2021) to enable various analytical perspectives.

This multi-stage aggregation process ensures that climate exposure estimates are:
- Representative of actual population distribution
- Consistent across administrative levels
- Compatible with health outcome data for epidemiological analyses
