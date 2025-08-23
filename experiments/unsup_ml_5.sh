#!/usr/bin/env bash
set -euo pipefail

model="pca"
datasets=( "santos" "wardbeck" "shaw" "wilson" )
window_size=5

PATH_DIR="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/temp/prod"
PARALLEL="${PARALLEL:-4}"

LOG_DIR="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/experiments/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="$LOG_DIR/pca_w${window_size}_${TS}.log"
ERROR_LOG="$LOG_DIR/pca_w${window_size}_error_${TS}.log"

total_runs=${#datasets[@]}
echo "[+] Model: $model" | tee -a "$SUMMARY_LOG"
echo "[+] Window size: $window_size" | tee -a "$SUMMARY_LOG"
echo "[+] Datasets: ${datasets[*]}" | tee -a "$SUMMARY_LOG"
echo "[+] Total runs: $total_runs (parallel=$PARALLEL)" | tee -a "$SUMMARY_LOG"
echo "[+] Path dir: $PATH_DIR" | tee -a "$SUMMARY_LOG"
echo "==================================" | tee -a "$SUMMARY_LOG"

# Semaphore init
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

run_job() {
  local dataset="$1"
  local run_id="${model}_${dataset}_${window_size}"
  local out_log="$LOG_DIR/${run_id}.out"
  local err_log="$LOG_DIR/${run_id}.err"
  local start_ts end_ts
  start_ts=$(date +%s)
  {
    echo "[START] $run_id"
    if logadu run "$model" "$dataset" "$window_size" --path "$PATH_DIR" >>"$out_log" 2>>"$err_log"; then
      end_ts=$(date +%s)
      echo "[OK] $run_id ($((end_ts-start_ts))s)"
      touch "$status_dir/success_${run_id}"
    else
      end_ts=$(date +%s)
      echo "[FAIL] $run_id ($((end_ts-start_ts))s) (see $err_log)"
      touch "$status_dir/fail_${run_id}"
      cat "$err_log" >>"$ERROR_LOG"
    fi
  } >>"$SUMMARY_LOG" 2>&1
}

for ds in "${datasets[@]}"; do
  sem_acquire
  {
    run_job "$ds"
    sem_release
  } &
done

wait

success=$(ls "$status_dir"/success_* 2>/dev/null | wc -l || true)
fail=$(ls "$status_dir"/fail_* 2>/dev/null | wc -l || true)

echo "==================================" | tee -a "$SUMMARY_LOG"
echo "[=] Finished. Success: $success  Fail: $fail  Total: $total_runs" | tee -a "$SUMMARY_LOG"
[[ $fail -eq 0 ]]