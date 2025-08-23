#!/usr/bin/env bash
set -euo pipefail

# FastText vectors file (can override via first arg)
VEC_FILE="${1:-/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation/crawl-300d-2M.vec}"

# Drain grouped path (override via GPATH env if desired)
GPATH="${GPATH:-/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/temp/prod/drain}"

# Datasets list (edit as needed)
datasets=( Linux24APT russellmitchell santos wardbeck shaw wilson fox harrison known_c1 known_c2 known_c3 known_c4 known_c5 known_c6 known_c7 known_c8 )

# Concurrency (default 4). Export PARALLEL to change.
PARALLEL="${PARALLEL:-4}"

# Logging setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/fasttext_tfidf_${RUN_TS}.log"

# Duplicate all stdout/stderr to master log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[+] Log file     : $LOG_FILE"
echo "[+] Per-dataset logs in: $LOG_DIR"
echo "[+] FastText file: $VEC_FILE"
echo "[+] Drain path   : $GPATH"
echo "[+] Datasets     : ${datasets[*]}"
echo "[+] Parallel     : $PARALLEL"
echo "=================================="

# Simple semaphore
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

STATUS_DIR=$(mktemp -d)
trap 'rm -rf "$STATUS_DIR"' EXIT

run_one() {
  local ds="$1"
  local start end
  local ds_log="$LOG_DIR/${ds}.log"
  start=$(date +%s)
  echo "[START] $ds  (log: $ds_log)"
  if logadu vectorize fasttext "$VEC_FILE" --dataset "$ds" --gpath "$GPATH" >"$ds_log" 2>&1; then
    end=$(date +%s)
    echo "[OK] $ds ($((end-start))s)"
    touch "$STATUS_DIR/success_$ds"
  else
    end=$(date +%s)
    echo "[FAIL] $ds ($((end-start))s)  See $ds_log"
    touch "$STATUS_DIR/fail_$ds"
  fi
}

for ds in "${datasets[@]}"; do
  sem_acquire
  {
    run_one "$ds"
    sem_release
  } &
done

wait

ok=$(ls "$STATUS_DIR"/success_* 2>/dev/null | wc -l || true)
fail=$(ls "$STATUS_DIR"/fail_* 2>/dev/null | wc -l || true)
echo "=================================="
echo "[=] Done. Success: $ok  Fail: $fail  Total: ${#datasets[@]}"
echo "[=] Master log: $LOG_FILE"

if (( fail > 0 )); then
  echo
  echo "[!] Failed datasets (last 15 lines each):"
  for f in "$STATUS_DIR"/fail_*; do
    [[ -e "$f" ]] || continue
    ds=${f##*fail_}
    ds_log="$LOG_DIR/${ds}.log"
    echo "----- $ds -----"
    tail -n 15 "$ds_log" || true
  done
  exit 1
fi