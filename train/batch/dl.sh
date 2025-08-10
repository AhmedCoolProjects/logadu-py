#!/bin/bash
#SBATCH --job-name=dl_training_batch
#SBATCH --output=/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/train/dl_training_%j.out
#SBATCH --error=/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/train/dl_training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --account=data_sec-6sevvl76uja-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ahmed.bargady@um6p.ma

# Load required modules
module load Anaconda3

# Source bashrc for environment setup
source ~/.bashrc

# Activate your conda environment
source activate logadu

# Change to the script directory
cd /home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/train

# Create the DL script if it doesn't exist
cat > run_dl.sh << 'EOF'
#!/bin/bash

# Define the arrays
models=("logcnn" "logrobust")
datasets=("Linux24APT" "Fox" "Russellmitchell")
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
echo "Running on node: $(hostname)"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "=================================="

# Triple nested loop for all combinations
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for window_size in "${window_sizes[@]}"; do
            current_run=$((current_run + 1))
            echo "[$current_run/$total_runs] Processing: $model | $dataset | window_size=$window_size"
            echo "Starting at: $(date)"
            
            # Run the logadu command
            logadu run "$model" "$dataset" "$window_size" --path "$PATH_DIR"
            
            # Check if the command was successful
            if [ $? -eq 0 ]; then
                echo "✓ Successfully processed: $model $dataset $window_size"
            else
                echo "✗ Error processing: $model $dataset $window_size"
            fi
            echo "Completed at: $(date)"
            echo "---"
        done
    done
done

echo "All combinations processed! ($total_runs total runs)"
EOF

# Make the script executable
chmod +x run_dl.sh

# Display system information
echo "Starting DL training batch job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "=================================="

# Run the script
./run_dl.sh

echo "=================================="
echo "DL training batch job completed at $(date)"