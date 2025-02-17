import numpy as np
import matplotlib.pyplot as plt 
from setup_plot import setup_global, setup_local, get_colors, get_linestyles, set_aspect_ratio, get_markers, get_hatches
import pandas as pd
import re
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from plot_eager_vs_rdvz import parse_to_dataframe

import warnings
warnings.filterwarnings("ignore")

def parse_all_files_in_folder(folder_path):
    # Initialize an empty list to hold DataFrames for each file
    dataframes = []
    
    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".out"):
            #print(f"Parsing file: {file_path}")
            df = parse_to_dataframe(file_path)
            # Add a column to identify the source file
            df["source_file"] = file_name
            dataframes.append(df)
    
    # Combine all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

#impl = sys.argv[1]
machine = sys.argv[1]
library = sys.argv[2]
#buffer_size = int(sys.argv[2])
if machine == "frontier":
    coll = "fr_all_gather"
elif machine == "perlmutter":
    coll = "pm_all_gather_2"

# setup_global()
# setup_local()
# colors = get_colors()
# linestyles = get_linestyles()
# markers=get_markers()
combined_df = parse_all_files_in_folder(folder_path=f"{coll}/")
data = combined_df[combined_df["method"] == library]
data["reg_1"] = data["gpu_count"] 
data["reg_2"] = (data["gpu_count"] - 1)/data["gpu_count"] * data["buffer_size_MB"]
data["target"] = data["time_ms"] 

# Define features and target
X = data[['reg_1', 'reg_2']]
y = data['time_ms']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)

# Initialize and train the model
model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print metrics
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

print(model.coef_, model.intercept_)
print(f"library = {library}")
print(f"latency term = {model.coef_[0]:.4f} ms/GPU")
print(f"bw term = {1/model.coef_[1]:.4f} GB/s")

print(np.abs((y_test-y_pred)/y_test*100).mean())
