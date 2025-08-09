#!/bin/bash
#SBATCH --job-name=ml_training_batch
#SBATCH --output=/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/train/ml_training_%j.out
#SBATCH --error=/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/train/ml_training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=himem
#SBATCH --qos=himem-cpu
#SBATCH --mem=64G
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

# Make the script executable
chmod +x run_ml.sh

# Run the script
echo "Starting ML training batch job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "=================================="

./run_ml.sh

echo "=================================="
echo "ML training batch job completed at $(date)"