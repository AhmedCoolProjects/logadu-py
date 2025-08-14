#!/bin/bash

# Define the arrays
models=("knn" "rf")
datasets=("santos" "wardbeck" "shaw" "wilson")
window_sizes=(20)

# PATH_DIR="/home/gpuadmin/Desktop/ahmed.bargady/data"
PATH_DIR="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/temp/prod"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_LOG="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/experiments/logs/knn_rf_${TIMESTAMP}.log"
ERROR_LOG="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/experiments/logs/knn_rf_error_${TIMESTAMP}.log"

# Counter for tracking progress
total_runs=$((${#models[@]} * ${#datasets[@]} * ${#window_sizes[@]}))
current_run=0

echo "Starting batch processing: $total_runs total runs" | tee -a "$OUTPUT_LOG"
echo "Models: ${models[*]}" | tee -a "$OUTPUT_LOG"
echo "Datasets: ${datasets[*]}" | tee -a "$OUTPUT_LOG"
echo "Window sizes: ${window_sizes[*]}" | tee -a "$OUTPUT_LOG"
echo "Output log: $OUTPUT_LOG" | tee -a "$OUTPUT_LOG"
echo "Error log: $ERROR_LOG" | tee -a "$OUTPUT_LOG"
echo "==================================" | tee -a "$OUTPUT_LOG"

# Triple nested loop for all combinations
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for window_size in "${window_sizes[@]}"; do
            current_run=$((current_run + 1))
            
            # Run the logadu command with model-specific parameters
            if logadu run "$model" "$dataset" "$window_size" --path "$PATH_DIR"  >> "$OUTPUT_LOG" 2>> "$ERROR_LOG"; then
                echo "✓ Successfully processed: $model $dataset $window_size with " | tee -a "$OUTPUT_LOG"
            else
                echo "✗ Error processing: $model $dataset $window_size with " | tee -a "$OUTPUT_LOG" "$ERROR_LOG"
            fi
            echo "---" | tee -a "$OUTPUT_LOG"
        done
    done
done

echo "All combinations processed! ($total_runs total runs)" | tee -a "$OUTPUT_LOG"