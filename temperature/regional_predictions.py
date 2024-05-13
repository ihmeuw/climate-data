import numpy as np
import pandas as pd
from joblib import load
import sys
from datetime import datetime
from joblib import load
import argparse
import os

# import pyproj
# pyproj.datadir.set_data_dir('/ihme/homes/nhashmeh/miniconda3/envs/earth-analytics-python/share/proj')
# os.environ['PROJ_LIB'] = '/ihme/homes/nhashmeh/miniconda3/envs/earth-analytics-python/share/proj'

# Output path (should make this an argument)
output_path = "/mnt/share/erf/ERA5/merged_era5_with_stations/grid_setup/predictions/"

task_id = int(sys.argv[1])
year = int(sys.argv[2])
path_to_chunk = sys.argv[3]
output_path = sys.argv[4]

print(task_id)
print(year)
print("Path to chunks: ", path_to_chunk)
print("Output path: ", output_path)

list_of_files = []

list_of_files = [
    os.path.join(path_to_chunk, file)
    for file in os.listdir(path_to_chunk)
    if file.split("_")[-1].split(".")[0] == str(year)
]

filename = list_of_files[task_id]
print("Filename: ", filename)

new_filename = output_path + "predictions_" + filename.split("/")[-1].split(".")[0]

# Load model

model = load(
    "/mnt/share/code/nhashmeh/downscaling/temperature/validation_model_one_dummy_v3.joblib"
)
scaler = load(
    "/mnt/share/code/nhashmeh/downscaling/temperature/validation_scaler_one_dummy_v3.joblib"
)

# Loop through each chunk, then loop through each day in that file, then process a little bit before predictions

loaded_data = pd.read_feather(filename)

num_unique_days = loaded_data["day_of_year"].nunique()
print(f"Number of unique days: {num_unique_days}")  # there should be 365 or 366
unique_days = loaded_data["day_of_year"].unique()

final_data = pd.DataFrame()  # this is where the output of each loop will be stored

for day in unique_days:
    loaded_data_day = loaded_data[loaded_data["day_of_year"] == day]

    # Convert longitude to -180 to 180 to match training data
    loaded_data_day.loc[loaded_data_day["longitude_st"] > 180, "longitude_st"] = (
        loaded_data_day.loc[loaded_data_day["longitude_st"] > 180, "longitude_st"] - 360
    )

    # Keeping a few columns to use later
    lcz_original = loaded_data_day["lcz_value"]

    print("converting lcz_value to dummies")
    data_expanded = pd.get_dummies(loaded_data_day, columns=["lcz_value"])

    print("adding missing dummies and filling them with zeros")
    # Get existing lcz_value columns
    existing_lcz_columns = [col for col in data_expanded.columns if "lcz_value_" in col]

    # Create a dataframe with dummy columns for missing lcz_values
    missing_lcz_values = pd.DataFrame(
        columns=[
            f"lcz_value_{i}"
            for i in range(18)
            if f"lcz_value_{i}" not in existing_lcz_columns
        ]
    )

    # Concatenate this with your original dataframe
    data_expanded = pd.concat([data_expanded, missing_lcz_values], axis=1)

    # Fill NaN values with 0 only in missing lcz_value columns
    data_expanded[missing_lcz_values.columns] = data_expanded[
        missing_lcz_values.columns
    ].fillna(0)

    # Create 'day_of_year_sin' and 'day_of_year_cos' features (these will replace day_of_year)
    data_expanded["day_of_year_sin"] = np.sin(
        2 * np.pi * data_expanded["day_of_year"] / data_expanded["total_days_in_year"]
    )
    data_expanded["day_of_year_cos"] = np.cos(
        2 * np.pi * data_expanded["day_of_year"] / data_expanded["total_days_in_year"]
    )

    # match column order of training data (this drops day_of_year)
    data_expanded = data_expanded[
        [
            "latitude_st",
            "longitude_st",
            "elevation",
            "t2m",
            "month",
            "year",
            "total_days_in_year",
            "day_of_year_sin",
            "day_of_year_cos",
            "lcz_value_1",
            "lcz_value_2",
            "lcz_value_3",
            "lcz_value_4",
            "lcz_value_5",
            "lcz_value_6",
            "lcz_value_7",
            "lcz_value_8",
            "lcz_value_9",
            "lcz_value_10",
            "lcz_value_11",
            "lcz_value_12",
            "lcz_value_13",
            "lcz_value_14",
            "lcz_value_15",
            "lcz_value_16",
            "lcz_value_17",
        ]
    ]

    # sanity check: make sure there is only one unique month value...
    month_store = data_expanded["month"].unique()
    if len(month_store) > 1:
        print(month_store)
        print("More than one month value exists in data, ending loop")
        break

    # Predictions start here
    print(
        "Processing data for day of year "
        + str(day)
        + " (month, year: "
        + str(month_store[0])
        + ", "
        + str(year)
        + ")"
    )

    X_test = scaler.transform(data_expanded)

    now1 = datetime.now()

    # Make predictions
    predictions = model.predict(X_test)

    now2 = datetime.now()
    time_elapsed = now2 - now1
    print("Predictions complete, time elapsed: ", time_elapsed)

    # Trimming things down to what we want/need...
    data_expanded_trim = data_expanded[
        ["latitude_st", "longitude_st", "elevation", "t2m", "month", "year"]
    ].copy()
    data_expanded_trim["day_of_year"] = day
    data_expanded_trim["lcz_value"] = lcz_original
    data_expanded_trim["predictions"] = predictions

    # Append the data for the current day to the larger DataFrame
    final_data = pd.concat([final_data, data_expanded_trim])

# Save the final data
final_data.reset_index(inplace=True)
final_data.to_feather(new_filename + ".feather")
