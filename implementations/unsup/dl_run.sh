#!/usr/bin/env bash
set -euo pipefail

# Multiple models supported now
models=( "deeplog" )
datasets=( Linux24APT russellmitchell santos wardbeck shaw wilson fox harrison known_c1 known_c2 known_c3 known_c4 known_c5 known_c6 known_c7 known_c8 )
window_size=5

PATH_DIR="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/temp/prod"

LOG_DIR="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/implementations/unsup/logs"
RUN_TS=$(date +%Y%m%d_%H%M%S)
BASE_FOLDER="$LOG_DIR/run_${RUN_TS}"
mkdir -p "$BASE_FOLDER"

SUMMARY_LOG="$BASE_FOLDER/summary.log"
ERROR_LOG="$BASE_FOLDER/errors.log"

total_models=${#models[@]}
total_datasets=${#datasets[@]}
total_runs=$(( total_models * total_datasets ))

# Auto PARALLEL = models * datasets unless user exported PARALLEL
if [[ -z "${PARALLEL:-}" ]]; then
  PARALLEL=$total_runs
fi

echo "[+] Models: ${models[*]}" | tee -a "$SUMMARY_LOG"
echo "[+] Datasets: ${datasets[*]}" | tee -a "$SUMMARY_LOG"
echo "[+] Window size: $window_size" | tee -a "$SUMMARY_LOG"
echo "[+] Path dir: $PATH_DIR" | tee -a "$SUMMARY_LOG"
echo "[+] Total runs: $total_runs (parallel=$PARALLEL)" | tee -a "$SUMMARY_LOG"
echo "==================================" | tee -a "$SUMMARY_LOG"

# Semaphore
sem_init() {
  mkfifo "$SEM_FIFO"
  exec {SEM_FD}<> "$SEM_FIFO"
  rm -f "$SEM_FIFO"
  for ((i=0;i<PARALLEL;i++)); do printf '.' >&"$SEM_FD"; done
}
sem_acquire() { read -r -n1 _ <&"$SEM_FD"; }
sem_release() { printf '.' >&"$SEM_FD"; }

SEM_FIFO=$(mktemp -u)
sem_init

status_dir=$(mktemp -d)
cleanup() {
  rm -rf "$status_dir" || true
}
trap cleanup EXIT

run_job() {
  local model="$1"
  local dataset="$2"
  local run_id="${model}_${dataset}"
  local model_folder="$BASE_FOLDER/$model"
  mkdir -p "$model_folder"
  local out_log="$model_folder/${model}_${dataset}.out"
  local err_log="$model_folder/${model}_${dataset}.err"
  local start_ts end_ts
  start_ts=$(date +%s)
  {
    echo "[START] $run_id (win=$window_size)"
    if logadu run "$model" "$dataset" "$window_size" --path "$PATH_DIR" >>"$out_log" 2>>"$err_log"; then
      end_ts=$(date +%s)
      echo "[OK] $run_id ($((end_ts-start_ts))s)"
      touch "$status_dir/success_${run_id}"
    else
      end_ts=$(date +%s)
      echo "[FAIL] $run_id ($((end_ts-start_ts))s) (see $err_log)"
      touch "$status_dir/fail_${run_id}"
      {
        echo "----- $run_id -----"
        cat "$err_log"
        echo
      } >>"$ERROR_LOG"
    fi
  } >>"$SUMMARY_LOG" 2>&1
}

# Progress counter (atomic via subshell lock)
progress_file="$BASE_FOLDER/.progress"
echo 0 > "$progress_file"
progress_update() {
  local lock
  exec {lock}> "$progress_file.lock"
  flock "$lock"
  local cur
  cur=$(<"$progress_file")
  cur=$((cur+1))
  echo "$cur" > "$progress_file"
  echo "[PROGRESS] $cur / $total_runs" >>"$SUMMARY_LOG"
}

for model in "${models[@]}"; do
  for ds in "${datasets[@]}"; do
    sem_acquire
    {
      run_job "$model" "$ds"
      progress_update
      sem_release
    } &
  done
done

wait

success=$(ls "$status_dir"/success_* 2>/dev/null | wc -l || true)
fail=$(ls "$status_dir"/fail_* 2>/dev/null | wc -l || true)

echo "==================================" | tee -a "$SUMMARY_LOG"
echo "[=] Finished. Success: $success  Fail: $fail  Total: $total_runs" | tee -a "$SUMMARY_LOG"
[[ $fail -eq