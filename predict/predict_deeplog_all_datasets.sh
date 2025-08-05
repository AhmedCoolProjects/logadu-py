#!/bin/bash

# Create output and logs directory structure
mkdir -p ./logs-predict/deeplog
mkdir -p ./logs-predict/deeplog/fox
mkdir -p ./logs-predict/deeplog/linux24
mkdir -p ./logs-predict/deeplog/russellmitchell
mkdir -p ./logs-predict/deeplog/win25ch

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

# Base path for trained models
MODEL_BASE_PATH="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/trained_models/deeplog"

echo "Starting DeepLog prediction for all datasets and window sizes..."
echo "=========================================="

# Loop through each dataset
for dataset_name in "${!datasets[@]}"; do
    dataset_path="${datasets[$dataset_name]}"
    
    echo ""
    echo "============================================"
    echo "Running DeepLog predictions on ${dataset_name^^} dataset..."
    echo "============================================"
    
    # Loop through each window size
    for window_size in "${window_sizes[@]}"; do
        echo ""
        echo "Predicting ${dataset_name} with window size ${window_size}..."
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
        
        # Construct model checkpoint path
        model_checkpoint="${MODEL_BASE_PATH}/deeplog-${dataset_name}_${window_size}-best-checkpoint.ckpt"
        
        # Construct log file path
        log_file="./logs-predict/deeplog/${dataset_name}/deeplog_${dataset_name}_predict_${window_size}.log"
        
        # Check if input file exists
        if [ ! -f "$input_file" ]; then
            echo "WARNING: Input file not found: $input_file"
            echo "Skipping ${dataset_name} with window size ${window_size}"
            continue
        fi
        
        # Check if model checkpoint exists
        if [ ! -f "$model_checkpoint" ]; then
            echo "WARNING: Model checkpoint not found: $model_checkpoint"
            echo "Skipping ${dataset_name} with window size ${window_size}"
            continue
        fi
        
        # Run the prediction command
        run_and_log "$log_file" \
            logadu predict "$input_file" \
            --model-type deeplog \
            --model-checkpoint "$model_checkpoint" \
            --top-k 9
        
        # Check if prediction was successful
        if [ $? -eq 0 ]; then
            echo "✅ Successfully completed prediction for ${dataset_name} (window size: ${window_size})"
        else
            echo "❌ Prediction failed for ${dataset_name} (window size: ${window_size})"
        fi
    done
    
    echo "Completed predictions for ${dataset_name^^} dataset!"
done

echo ""
echo "=========================================="
echo "🎉 All DeepLog prediction jobs completed!"
echo "=========================================="
echo ""
echo "📊 Prediction Summary:"
echo "  - Model: DeepLog"
echo "  - Datasets: Fox, Linux24APT, Russellmitchell, Win25ChAPT"
echo "  - Window sizes: 10, 20, 50, 100"
echo "  - Total jobs: 16 (4 datasets × 4 window sizes)"
echo "  - Top-k parameter: 9"
echo ""
echo "📁 Check the following directories:"
echo "  - Prediction logs: ./logs-predict/deeplog/{dataset_name}/"
echo "  - Models used: ${MODEL_BASE_PATH}/"
echo "=========================================="