#!/bin/bash

# Create output and logs directory structure
mkdir -p ./trained_models/deeplog
mkdir -p ./logs-train/deeplog
mkdir -p ./logs-train/deeplog/fox
mkdir -p ./logs-train/deeplog/linux24
mkdir -p ./logs-train/deeplog/russellmitchell
mkdir -p ./logs-train/deeplog/win25ch

# Function to run command and log output
run_and_log() {
    local log_file="$1"
    shift
    echo "Running: $@"
    echo "Logging to: $log_file"
    echo "Started at: $(date)" | tee "$log_file"
    echo "Command: $@" | tee -a "$log_file"
    echo "----------------------------------------" | tee -a "$log_file"
    
    # Run the command and capture both stdout and stderr
    "$@" 2>&1 | tee -a "$log_file"
    
    # Log completion time and exit status
    local exit_code=$?
    echo "----------------------------------------" | tee -a "$log_file"
    echo "Completed at: $(date)" | tee -a "$log_file"
    echo "Exit code: $exit_code" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    
    return $exit_code
}

# Define datasets and their configurations
declare -A datasets
datasets[fox]="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation/Fox/drain"
datasets[linux24]="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation/Linux24APT/drain"
datasets[russellmitchell]="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation/Russellmitchell/drain"
datasets[win25ch]="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation/Win25ChAPT/drain"

# Define window sizes
window_sizes=(10 20 50 100)

echo "Starting DeepLog training for all datasets and window sizes..."
echo "=========================================="

# Loop through each dataset
for dataset_name in "${!datasets[@]}"; do
    dataset_path="${datasets[$dataset_name]}"
    
    echo ""
    echo "============================================"
    echo "Training DeepLog on ${dataset_name^^} dataset..."
    echo "============================================"
    
    # Loop through each window size
    for window_size in "${window_sizes[@]}"; do
        echo ""
        echo "Training ${dataset_name} with window size ${window_size}..."
        echo "--------------------------------------------"
        
        # Construct the input file path
        if [ "$dataset_name" = "fox" ]; then
            input_file="${dataset_path}/Fox_${window_size}_1_seq_index.csv"
        elif [ "$dataset_name" = "linux24" ]; then
            input_file="${dataset_path}/Linux24APT_${window_size}_1_seq_index.csv"
        elif [ "$dataset_name" = "russellmitchell" ]; then
            input_file="${dataset_path}/Russellmitchell_${window_size}_1_seq_index.csv"
        elif [ "$dataset_name" = "win25ch" ]; then
            input_file="${dataset_path}/Win25ChAPT_${window_size}_1_seq_index.csv"
        fi
        
        # Construct log file path
        log_file="./logs-train/deeplog/${dataset_name}/deeplog_${dataset_name}_train_${window_size}.log"
        
        # Construct W&B run name and dataset name
        wandb_run_name="deeplog_${dataset_name}_index_${window_size}"
        dataset_full_name="${dataset_name}_${window_size}"
        
        # Check if input file exists
        if [ ! -f "$input_file" ]; then
            echo "WARNING: Input file not found: $input_file"
            echo "Skipping ${dataset_name} with window size ${window_size}"
            continue
        fi
        
        # Run the training command
        run_and_log "$log_file" \
            logadu train "$input_file" \
            --model deeplog \
            --epochs 100 \
            --output-dir ./trained_models/deeplog \
            --wandb-project "lad_in_apts" \
            --wandb-run-name "$wandb_run_name" \
            --dataset-name "$dataset_full_name"
        
        # Check if training was successful
        if [ $? -eq 0 ]; then
            echo "✅ Successfully completed training for ${dataset_name} (window size: ${window_size})"
        else
            echo "❌ Training failed for ${dataset_name} (window size: ${window_size})"
        fi
    done
    
    echo "Completed training for ${dataset_name^^} dataset!"
done

echo ""
echo "=========================================="
echo "🎉 All DeepLog training jobs completed!"
echo "=========================================="
echo ""
echo "📊 Training Summary:"
echo "  - Model: DeepLog"
echo "  - Datasets: Fox, Linux24APT, Russellmitchell, Win25ChAPT"
echo "  - Window sizes: 10, 20, 50, 100"
echo "  - Total jobs: 16 (4 datasets × 4 window sizes)"
echo ""
echo "📁 Check the following directories:"
echo "  - Models: ./trained_models/deeplog/"
echo "  - Logs: ./logs-train/deeplog/{dataset_name}/"
echo "=========================================="