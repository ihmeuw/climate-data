Downscaling temperature

This repo contains the fundamental code used to downscale ERA5 daily average temperatures to 1-km resolution using machine learning. 00_launch_scripts.ipynb is the main script. It contains the launcher code for each worker script in the repo, as well as some important intermediate processing steps. Future to-do items include:

1. Moving the intermediate processing steps to their own worker scripts
2. Converting all outputs to a compressed file format. Ideally, using compression options found in the xarray library when outputting .nc files.
3. Creating an additional script that addresses problematic outliers in the prediction step. Ideally, this is done within the regional_predictions.py script or directly after it. 
4. Include diagnostic scripts used to visualize prediction data.