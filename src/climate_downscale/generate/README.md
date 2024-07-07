# Climate Variable Pipeline

This set of scripts processes ERA5 and CMIP6 climate data into a database of
climate variables at a consistent resolution and format. The pipeline is
run in several stages:

1.  Historical Daily: This processes the hourly ERA5-Land and ERA5-Single-Level
    data into a unified daily format, pulling the higher-resolution ERA5-Land data
    where available and filling in with ERA5-Single-Level data.
2.  Historical Reference: This produces a set of reference climatologies from this
    historical daily results by averaging, by month and space, over a historical
    reference period. This is used to downscale and bias-correct the CMIP6 data.
3.  Scenario Daily: This produces scenario projections from the CMIP6 data, ensembling
    over a curated set of GCMs and using the historical reference climatologies to
    bias-correct the data.
4.  Derived Daily: This produces derived climate variables from the daily data, such as
    humidex and effective temperature. This writes results to the same directories
    as the daily data.
5.  Scenario Annual: This produces annualized summaries of the scenario data, such as
    annual averages and extremes.
