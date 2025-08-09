#!/bin/bash

# Define the datasets array
# datasets=("Win25ChAPT" "Linux24APT" "Fox" "Russellmitchell")
datasets=("Win25ChAPT" "Russellmitchell")

# Define common paths
FASTTEXT_MODEL="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/dataset/helpers/crawl-300d-2M.vec"
GPATH="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation"

# Loop through each dataset and run the command
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    logadu vectorize fasttext "$FASTTEXT_MODEL" --dataset "$dataset" --gpath "$GPATH"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed $dataset"
    else
        echo "Error processing $dataset"
    fi
    echo "---"
done

echo "All datasets processed!"