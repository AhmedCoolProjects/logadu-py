#!/bin/bash

# DataSets: Win25ChAPT, Linux24APT, Fox, Russellmitchell
# window sizes: 5, 10, 20, 30, 60

# Example: logadu run deeplog Fox 5 --path /home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation

# Define datasets and window sizes
datasets=("Win25ChAPT" "Linux24APT" "Fox" "Russellmitchell")
window_sizes=(5 10 20 30 60)

# Base path for datasets
base_path="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation"

# Function to run a command on the combination of dataset and window size
run_logadu() {
    local dataset=$1
    local window_size=$2
    
    echo "Running logadu with dataset: $dataset, window size: $window_size"
    logadu run deeplog "$dataset" "$window_size" --path "$base_path"
    
    # Check if command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $dataset with window size $window_size"
    else
        echo "✗ Failed: $dataset with window size $window_size"
    fi
    echo "----------------------------------------"
}

# Main execution loop
echo "Starting logadu runs for all dataset and window size combinations..."
echo "========================================"

for dataset in "${datasets[@]}"; do
    for window_size in "${window_sizes[@]}"; do
        run_logadu "$dataset" "$window_size"
    done
done

echo "All runs completed!"