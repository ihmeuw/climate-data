# Climate Downscaling Data Sources

## The Climate Data

### [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview)

ERA5-Land is a reanalysis dataset providing a consistent view of the evolution of land
variables over several decades at an enhanced resolution compared to ERA5. ERA5-Land
has been produced by replaying the land component of the ECMWF ERA5 climate reanalysis.
Reanalysis combines model data with observations from across the world into a globally
complete and consistent dataset using the laws of physics. Reanalysis produces data
that goes several decades back in time, providing an accurate description of the
climate of the past.

ERA5-Land uses as input to control the simulated land fields ERA5 atmospheric
variables, such as air temperature and air humidity. This is called the atmospheric
forcing. Without the constraint of the atmospheric forcing, the model-based estimates
can rapidly deviate from reality. Therefore, while observations are not directly used
in the production of ERA5-Land, they have an indirect influence through the atmospheric
forcing used to run the simulation. In addition, the input air temperature, air
humidity and pressure used to run ERA5-Land are corrected to account for the altitude
difference between the grid of the forcing and the higher resolution grid of ERA5-Land.
This correction is called 'lapse rate correction'.

The ERA5-Land dataset, as any other simulation, provides estimates which have some
degree of uncertainty. Numerical models can only provide a more or less accurate
representation of the real physical processes governing different components of the
Earth System. In general, the uncertainty of model estimates grows as we go back in
time, because the number of observations available to create a good quality atmospheric
forcing is lower. ERA5-land parameter fields can currently be used in combination with
the uncertainty of the equivalent ERA5 fields.

The temporal and spatial resolutions of ERA5-Land makes this dataset very useful for
all kind of land surface applications such as flood or drought forecasting. The
temporal and spatial resolution of this dataset, the period covered in time, as well as
the fixed grid used for the data distribution at any period enables decisions makers,
businesses and individuals to access and use more accurate information on land states.

### [CMIP6](https://pcmdi.llnl.gov/CMIP6/)


## The Reference Data

### [NCEI Climate Station Summaries](https://www.ncei.noaa.gov/data/global-summary-of-the-day/)

The Global Summary of the Day (GSOD) is a daily weather dataset that provides
observations for a subset of weather stations around the world. The dataset
includes elements such as temperature, precipitation, wind, and pressure. The
GSOD dataset is produced by the National Centers for Environmental Information
(NCEI) and is available for download from the NCEI website.


## Downscaling Predictors

### [Local Climate Zones](https://lcz-generator.rub.de/global-lcz-map)
