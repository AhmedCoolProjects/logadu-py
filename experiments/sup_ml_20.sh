#!/usr/bin/env bash
set -euo pipefail

# Models / datasets / window sizes
models=( "knn" "rf" )
datasets=( "santos" "wardbeck" "shaw" "wilson" )
window_sizes=(20)

PATH_DIR="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/temp/prod"
PARALLEL="${PARALLEL:-4}"

LOG_DIR="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/experiments/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
FOLDER_="$LOG_DIR/knn_rf"
mkdir -p "$FOLDER_"
SUMMARY_LOG="$FOLDER_/knn_rf_${TS}.log"
ERROR_LOG="$FOLDER_/knn_rf_error_${TS}.log"

total_runs=$(( ${#models[@]} * ${#datasets[@]} * ${#window_sizes[@]} ))
echo "[+] Total runs: $total_runs (parallel=$PARALLEL)" | tee -a "$SUMMARY_LOG"
echo "[+] Models: ${models[*]}" | tee -a "$SUMMARY_LOG"
echo "[+] Datasets: ${datasets[*]}" | tee -a "$SUMMARY_LOG"
echo "[+] Window sizes: ${window_sizes[*]}" | tee -a "$SUMMARY_LOG"
echo "[+] Path dir: $PATH_DIR" | tee -a "$SUMMARY_LOG"
echo "==================================" | tee -a "$SUMMARY_LOG"

# Build job list
declare -a JOBS=()
for m in "${models[@]}"; do
  for d in "${datasets[@]}"; do
    for w in "${window_sizes[@]}"; do
      JOBS+=( "$m;$d;$w" )
    done
  done
done

# Semaphore (token bucket)
sem_init() {
  mkfifo "$SEM_FIFO"
  exec {SEM_FD}<> "$SEM_FIFO"
  rm -f "$SEM_FIFO"
  for ((i=0;i<PARALLEL;i++)); do
    printf '.' >&"$SEM_FD"
  done
}
sem_acquire() { read -r -n1 _ <&"$SEM_FD"; }
sem_release() { printf '.' >&"$SEM_FD"; }

SEM_FIFO=$(mktemp -u)
sem_init

status_dir=$(mktemp -d)
run_counter_file=$(mktemp)
echo 0 > "$run_counter_file"

run_job() {
  local model="$1" dataset="$2" win="$3"
  local run_id="${model}_${dataset}_${win}"
  local out_log="$FOLDER_/${run_id}.out"
  local err_log="$FOLDER_/${run_id}.err"
  local start_ts end_ts
  start_ts=$(date +%s)

  {
    echo "[START] $run_id"
    if logadu run "$model" "$dataset" "$win" --gpath "$PATH_DIR" >>"$out_log" 2>>"$err_log"; then
      end_ts=$(date +%s)
      echo "[OK] $run_id (${end_ts-start_ts}s)"
      touch "$status_dir/success_${run_id}"
    else
      end_ts=$(date +%s)
      echo "[FAIL] $run_id (${end_ts-start_ts}s) (see $err_log)"
      touch "$status_dir/fail_${run_id}"
      cat "$err_log" >>"$ERROR_LOG"
    fi
  } >>"$SUMMARY_LOG" 2>&1
}

for job in "${JOBS[@]}"; do
  sem_acquire
  IFS=';' read -r M D W <<<"$job"
  {
    run_job "$M" "$D" "$W"
    sem_release
  } &
done

wait

success=$(ls "$status_dir"/success_* 2>/dev/null | wc -l || true)
fail=$(ls "$status_dir"/fail_* 2>/dev/null | wc -l || true)

echo "==================================" | tee -a "$SUMMARY_LOG"
echo "[=] Finished. Success: $success  Fail: $fail  Total: $total_runs" | tee -a "$SUMMARY_LOG"
[[ $fail -eq 0 ]] || exit 1