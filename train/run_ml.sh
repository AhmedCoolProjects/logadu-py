#!/bin/bash

# Define the arrays
models=("knn" "pca" "rf")
# datasets=("Fox" "Russellmitchell")
datasets=("Linux24APT" "Fox" "Russellmitchell")
# datasets=("Win25ChAPT" "Linux24APT" "Fox" "Russellmitchell")
window_sizes=(5 10 20 30 60)

# Define common path
PATH_DIR="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation"

# Counter for tracking progress
total_runs=$((${#models[@]} * ${#datasets[@]} * ${#window_sizes[@]}))
current_run=0

echo "Starting batch processing: $total_runs total runs"
echo "Models: ${models[*]}"
echo "Datasets: ${datasets[*]}"
echo "Window sizes: ${window_sizes[*]}"
echo "=================================="

# Triple nested loop for all combinations
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for window_size in "${window_sizes[@]}"; do
            current_run=$((current_run + 1))
            echo "[$current_run/$total_runs] Processing: $model | $dataset | window_size=$window_size"
            
            # Run the logadu command
            logadu run "$model" "$dataset" "$window_size" --path "$PATH_DIR" --k-neighbors 5
            
            # Check if the command was successful
            if [ $? -eq 0 ]; then
                echo "✓ Successfully processed: $model $dataset $window_size"
            else
                echo "✗ Error processing: $model $dataset $window_size"
            fi
            echo "---"
        done
    done
done

echo "All combinations processed! ($total_runs total runs)"