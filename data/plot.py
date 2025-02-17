import numpy as np
import matplotlib.pyplot as plt 
from setup_plot import setup_global, setup_local, get_colors, get_linestyles, set_aspect_ratio, get_markers, get_hatches
import pandas as pd
import re
import os
import sys

def parse_file():
    import re

import re
import pandas as pd

def parse_to_dataframe(file_path):
    # Initialize an empty list to hold rows for the DataFrame
    rows = []
    num_gpus = None
    
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    current_output_size = None
    method = None

    for line in lines:
        line = line.strip()
        
        if "output size =" in line:
            current_output_size = float(re.search(r"output size = ([\d.]+) MB", line).group(1))

        elif "input size =" in line:
            current_output_size = float(re.search(r"input size = ([\d.]+) MB", line).group(1))
        
        elif "Method =" in line:
            method = re.search(r"Method = ([\w-]+)", line).group(1).lower()
        
        elif "bus bw" in line:
            match = re.search(r"bus bw for (\d+) GPUs is ([\d.]+) GBPS for message output size ([\d.]+) MB", line)
            if match:
                num_gpus = int(match.group(1))
                bus_bw = float(match.group(2))
                msg_size = float(match.group(3))
        
        elif "time =" in line:
            time_ms = float(re.search(r"time = ([\d.]+) ms", line).group(1))
            # Add a new row for the DataFrame
            rows.append({
                "method": method,
                "buffer_size_MB": current_output_size,
                "gpu_count": num_gpus,
                "time_ms": time_ms,
                "bus_bw_GBPS": bus_bw
            })
    
    # Convert the rows into a Pandas DataFrame
    df = pd.DataFrame(rows)
    return df

def parse_all_files_in_folder(folder_path):
    # Initialize an empty list to hold DataFrames for each file
    dataframes = []
    
    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".out"):
            print(f"Parsing file: {file_path}")
            df = parse_to_dataframe(file_path)
            # Add a column to identify the source file
            df["source_file"] = file_name
            dataframes.append(df)
    
    # Combine all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

if __name__ == "__main__":
    #impl = sys.argv[1]
    machine = sys.argv[1]
    buffer_size = int(sys.argv[2])
    if machine == "frontier":
        coll = "all_gather_2"
    elif machine == "perlmutter":
        coll = "pm_all_gather_2"

    setup_global()
    setup_local()
    colors = get_colors()
    linestyles = get_linestyles()
    markers=get_markers()
    combined_df = parse_all_files_in_folder(folder_path=f"{machine}/{coll}/")
    key = "time_ms"
    #key = "bus_bw_GBPS"    

    for idx, impl in enumerate(["nccl", "hybrid", "mpi"]):
        #collective="all_gather_2"
        # Filter for the allowed buffer sizes
        filtered_df = combined_df[combined_df["buffer_size_MB"] == buffer_size]
        print(filtered_df)
        df = filtered_df[filtered_df["method"] == impl]
        if impl == "nccl" and machine == "frontier":
            impl = "rccl"
        buffer_sizes = sorted(filtered_df["buffer_size_MB"].unique())
        i=0
        for buffer_size in buffer_sizes:
            subset = df[df["buffer_size_MB"] == buffer_size].sort_values(by="gpu_count")
            #x_values = subset["gpu_count"].apply(lambda x: int(np.log2(x / 8)))  # log2 scale for 8, 16, 32, etc.
            x_values = subset["gpu_count"]
            
            plt.plot(x_values, subset[key], label=f"{impl}-{buffer_size} MB", 
                    linestyle=linestyles[idx], color=colors[idx], marker=markers[idx])
        
            i+=1
        
    # Add labels, title, and legend
    plt.xlabel("GPU Count", fontsize=12)
    if key == "bus_bw_GBPS":
        plt.ylabel("Bus Bandwidth (GBPS)", fontsize=12)
    elif key == "time_ms":
        plt.ylabel("Time (ms)", fontsize=12)

    #plt.title(f"Effect of (en/dis)abling eager for {impl} on Frontier", fontsize=14)
    plt.title(f"Latencies for {coll} on {machine}", fontsize=14)

    plt.legend(title="Message Size (MB)", fontsize=10)
    plt.ylim(0) # 100)
    set_aspect_ratio(3/5)
    if not os.path.isdir(f"plots/{machine}/{coll}/"):
        os.makedirs(f"plots/{machine}/{coll}/")
    plt.savefig(f"plots/{machine}/{coll}/plot_{buffer_size}_MB.pdf", bbox_inches='tight')

