import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from data.setup_plot import setup_global, setup_local, get_linestyles, get_markers, get_colors

# Directory containing CSV files

if False:
    csv_files = glob.glob("./data_10_runs/frontier/*.csv")  # Adjust the path if needed

    dataframes = []

    # Read all CSV files and store them in a list
    temp_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                temp_list.append(df)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")


    # Concatenate all dataframes
    full_df = pd.concat(temp_list, ignore_index=True)

    # Identify columns that contain time measurements
    time_columns = [col for col in full_df.columns if col.startswith("time_")]

    # Compute mean and standard deviation grouped by gpu_count and output_size
    result = full_df.groupby(["gpu_count", "output_size"])[time_columns].agg(['mean', 'std', 'count'])

    # Flatten multi-index columns
    result.columns = ['_'.join(col).strip() for col in result.columns.values]
    # Save the result to a CSV file
    result.to_csv("mean_std_results.csv")
else:
    csv_files = glob.glob("./data_10_runs/frontier_3/*.csv")
    # Dictionary to store separate DataFrames for each method
    method_dataframes = {}

    # Read and store data for each method separately
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if df.empty:
                print(f"Skipping empty file: {file}")
                continue

            # Identify time column (each file contains only one method)
            time_columns = [col for col in df.columns if col.startswith("time_")]
            
            if len(time_columns) != 1:
                print(f"Skipping file {file} due to unexpected format.")
                continue
            
            time_col = time_columns[0]  # Extract the time method name

            if "gpu_count" in df.columns and "output_size" in df.columns:
                df["bus_bw"] = ((df["gpu_count"] - 1) / df["gpu_count"]) * (df["output_size"] / (df[time_col] / 1000))  # Convert time from ms to s
                df["bus_bw"] = df["bus_bw"].fillna(0)  # Handle NaN values
                if time_col not in method_dataframes:
                    method_dataframes[time_col] = df[["gpu_count", "output_size", time_col]]
                else:
                    method_dataframes[time_col] = pd.concat([method_dataframes[time_col], df[["gpu_count", "output_size", time_col]]])
            
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")

    # Compute mean, std, and count for each method separately
    processed_data = {}
    for method, df in method_dataframes.items():
        result = df.groupby(["gpu_count", "output_size"])[method].agg(['mean', 'std', 'count']).reset_index()
        processed_data[method] = result
        result.to_csv(f"processed_{method}.csv", index=False)
        print(f"Processed results saved for {method} in processed_{method}.csv.")

print("Processing complete. Results saved in mean_std_results.csv.")
exit()

setup_global() 
selected_methods = ["mpi", "nccl", "inner_nccl_outer_mpi", "inner_nccl_outer_nccl", "inner_mpi_outer_mpi"]
selected_methods = ["time_" + x for x in selected_methods]

def plot_times_for_message_size(message_size):
    #setup_global()
    setup_local()
    

    for i, method in enumerate(selected_methods):
        df = processed_data[method]
        filtered_df = df[df["output_size"] == message_size]
        
        if not filtered_df.empty:
            plt.errorbar(filtered_df["gpu_count"], 
                         filtered_df["mean"], 
                         yerr=filtered_df["std"], 
                         marker=get_markers()[i], capsize=3, alpha=0.7, 
                         label=method, linestyle=get_linestyles()[i], color=get_colors()[i])

            # Annotate count values
            for j, txt in enumerate(filtered_df["count"].values):
                plt.text(filtered_df["gpu_count"].values[j], filtered_df["mean"].values[j], 
                         str(int(txt)), ha='right', va='bottom', fontsize=9, color='black')

    #plt.ylim(0)
    plt.xlabel("GPU Count")
    plt.ylabel("Time (s)")
    plt.yscale("log", base=2)
    plt.xscale("log", base=2)
    plt.xticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    plt.title(f"Time Performance for Message Size {message_size} MB")
    plt.legend()
    plt.savefig(f"performance_{message_size}_time.pdf")
    plt.show()

def plot_bws_for_message_size(message_size):
    setup_local()
    
    for i, method in enumerate(selected_methods):
        df = processed_data[method]
        filtered_df = df[df["output_size"] == message_size]
        eff_message_size = (filtered_df["gpu_count"] - 1) / (filtered_df["gpu_count"]) * message_size

        if not filtered_df.empty:
            bus_bw = eff_message_size / filtered_df["mean"]
            plt.plot(filtered_df["gpu_count"], bus_bw, marker=get_markers()[i], alpha=0.7, 
                          label=method, linestyle=get_linestyles()[i], color=get_colors()[i] ) 

            # Annotate count values
            for j, txt in enumerate(filtered_df["count"].values):
                plt.text(filtered_df["gpu_count"].values[j], bus_bw.values[j], 
                         str(int(txt)), ha='right', va='bottom', fontsize=9, color='black')

    plt.ylim(0)
    #plt.xlim(0)
    plt.xscale("log", base=2)
    plt.xlabel("GPU Count")
    plt.ylabel("Bus Bw (GB/s)")
    plt.title(f"Bus Bw Performance for Message Size {message_size} MB")
    plt.legend()
    plt.savefig(f"performance_{message_size}_bw.pdf")
    plt.show()

# Example usage (adjust message size as needed)
for msg in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    plot_times_for_message_size(msg)
    plot_bws_for_message_size(msg)