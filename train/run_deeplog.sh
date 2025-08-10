#!/bin/bash

# Define the arrays
datasets=("Linux24APT" "Fox" "Russellmitchell")
window_sizes=(5 10 20 30 60)
topk_values=(3 9)

# Define common path
PATH_DIR="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation"

# Create log files with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_LOG="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/train/run_deeplog_output_${TIMESTAMP}.log"
ERROR_LOG="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/train/run_deeplog_error_${TIMESTAMP}.log"

# Counter for tracking progress
total_runs=$((${#datasets[@]} * ${#window_sizes[@]} * ${#topk_values[@]}))
current_run=0

echo "Starting batch processing: $total_runs total runs" | tee -a "$OUTPUT_LOG"
echo "Datasets: ${datasets[*]}" | tee -a "$OUTPUT_LOG"
echo "Window sizes: ${window_sizes[*]}" | tee -a "$OUTPUT_LOG"
echo "TopK values: ${topk_values[*]}" | tee -a "$OUTPUT_LOG"
echo "Output log: $OUTPUT_LOG" | tee -a "$OUTPUT_LOG"
echo "Error log: $ERROR_LOG" | tee -a "$OUTPUT_LOG"
echo "==================================" | tee -a "$OUTPUT_LOG"

# Triple nested loop for all combinations
for dataset in "${datasets[@]}"; do
    for window_size in "${window_sizes[@]}"; do
        for topk in "${topk_values[@]}"; do
            current_run=$((current_run + 1))
            echo "[$current_run/$total_runs] Processing: deeplog | $dataset | window_size=$window_size | topk=$topk" | tee -a "$OUTPUT_LOG"
            
            # Run the logadu command
            if logadu run "deeplog" "$dataset" "$window_size" --path "$PATH_DIR" --topk "$topk" >> "$OUTPUT_LOG" 2>> "$ERROR_LOG"; then
                echo "✓ Successfully processed: deeplog $dataset $window_size topk=$topk" | tee -a "$OUTPUT_LOG"
            else
                echo "✗ Error processing: deeplog $dataset $window_size topk=$topk" | tee -a "$OUTPUT_LOG" "$ERROR_LOG"
            fi
            echo "---" | tee -a "$OUTPUT_LOG"
        done
    done
done

echo "All combinations processed! ($total_runs total runs)" | tee -a "$OUTPUT_LOG"