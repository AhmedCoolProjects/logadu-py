#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Models / datasets / window sizes
models=( "logrobust" ) # The only change is here: targeting only the logrobust model
datasets=( "Linux24APT" )
# datasets=( "Linux24APT" "santos" "wardbeck" "shaw" "wilson" "fox" "harrison" "russellmitchell" )
# datasets=( known_c1 known_c2 known_c3 known_c4 known_c5 known_c6 known_c7 known_c8 )
window_sizes=(20) 
paradigm="supervised"

PATH_DIR="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/temp/prod"
# Using PARALLEL=4 as a default, can be overridden by exporting the variable
PARALLEL="${PARALLEL:-4}"

LOG_DIR="/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/implementations/sup/logs3"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
# Creating a specific folder for this run
FOLDER_="$LOG_DIR/logrobust_run_${TS}"
mkdir -p "$FOLDER_"
SUMMARY_LOG="$FOLDER_/summary.log"
ERROR_LOG="$FOLDER_/errors.log"

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

# Semaphore (token bucket) for managing concurrency
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

run_job() {
  local model="$1" dataset="$2" win="$3"
  local run_id="${model}_${dataset}_${win}"
  local out_log="$FOLDER_/${run_id}.out"
  local err_log="$FOLDER_/${run_id}.err"
  local start_ts end_ts
  start_ts=$(date +%s)

  # This block is logged to the main summary log
  {
    echo "[START] $run_id"
    # The core command from your original script
    # NOTE: I noticed your original used --path, but the first script used --gpath. 
    # I've kept --path as it was in this template.
    if logadu run "$model" "$paradigm" "$dataset" "$win" --gpath "$PATH_DIR" >>"$out_log" 2>>"$err_log"; then
      end_ts=$(date +%s)
      echo "[OK] $run_id ($((${end_ts}-${start_ts}))s)"
      touch "$status_dir/success_${run_id}"
    else
      end_ts=$(date +%s)
      echo "[FAIL] $run_id ($((${end_ts}-${start_ts}))s) (see $err_log)"
      touch "$status_dir/fail_${run_id}"
      # Also append the specific error log to the main error log
      printf "\n----- START ERROR for %s -----\n" "$run_id" >> "$ERROR_LOG"
      cat "$err_log" >>"$ERROR_LOG"
      printf "----- END ERROR for %s -----\n" "$run_id" >> "$ERROR_LOG"
    fi
  } >>"$SUMMARY_LOG" 2>&1
}

# Main execution loop
for job in "${JOBS[@]}"; do
  sem_acquire
  # Run the job in the background
  (
    # Unpack job arguments
    IFS=';' read -r M D W <<<"$job"
    run_job "$M" "$D" "$W"
    sem_release # Release the token after the job is done
  ) &
done

# Wait for all background jobs to finish
wait

# Final summary based on status files
success=$(ls "$status_dir"/success_* 2>/dev/null | wc -l)
fail=$(ls "$status_dir"/fail_* 2>/dev/null | wc -l)

# Clean up temporary directory
rm -rf "$status_dir"

echo "==================================" | tee -a "$SUMMARY_LOG"
echo "[=] Finished. Success: $success  Fail: $fail  Total: $total_runs" | tee -a "$SUMMARY_LOG"
[[ $fail -eq 0 ]] || { echo "[!] Some jobs failed. Check $ERROR_LOG for details."; exit 1; }

exit 0