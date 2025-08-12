#!/bin/bash

# Define the arrays
models=("neurallog")
datasets=("Fox")
# NEXT
# datasets=("Russellmitchell")
# datasets=("Linux24APT" "Fox" "Russellmitchell")
window_sizes=(30 60)
# NEXT
# window_sizes=(5 60)
# window_sizes=(5 10 20 30 60)

# Define common path
# PATH_DIR="/home/gpuadmin/Desktop/ahmed.bargady/data"
PATH_DIR="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation"

# Create log files with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_LOG="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/train/run_neurallog_output_${TIMESTAMP}.log"
ERROR_LOG="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/train/run_neurallog_error_${TIMESTAMP}.log"

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
            echo "[$current_run/$total_runs] Processing: $model | $dataset | window_size=$window_size" | tee -a "$OUTPUT_LOG"
            
            # Run the logadu command
            if logadu run "$model" "$dataset" "$window_size" --path "$PATH_DIR" >> "$OUTPUT_LOG" 2>> "$ERROR_LOG"; then
                echo "✓ Successfully processed: $model $dataset $window_size" | tee -a "$OUTPUT_LOG"
            else
                echo "✗ Error processing: $model $dataset $window_size" | tee -a "$OUTPUT_LOG" "$ERROR_LOG"
            fi
            echo "---" | tee -a "$OUTPUT_LOG"
        done
    done
done

echo "All combinations processed! ($total_runs total runs)" | tee -a "$OUTPUT_LOG"