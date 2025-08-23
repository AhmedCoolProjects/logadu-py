#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_logadu_grid.sh <model> <paradigm> <PATH_DIR>
# Example: bash scripts/run_logadu_grid.sh logbert semi /data/logs
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <model> <paradigm> <PATH_DIR>"
  echo "  model: one of deeplog|logbert|logrobust|logcnn|plelog|neurallog|pca|knn|rf|ocsvm"
  echo "  paradigm: supervised|semi|unsupervised"
  echo "  PATH_DIR: dataset root directory"
  exit 1
fi

MODEL="$1"
PARADIGM="$2"
PATH_DIR="$3"

WIN_SIZES=(5 30)


DATASETS=(
  "Linux24APT" "santos" "wardbeck" "shaw" "wilson" "fox" "harrison" "russellmitchell" known_c1 known_c2 known_c3 known_c4 known_c5 known_c6 known_c7 known_c8
)

for win in "${WIN_SIZES[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo ">>> Running: model=${MODEL} paradigm=${PARADIGM} dataset=${dataset} win=${win}"
    logadu run "$MODEL" "$PARADIGM" "$dataset" "$win" --gpath "$PATH_DIR"
    echo
  done
done