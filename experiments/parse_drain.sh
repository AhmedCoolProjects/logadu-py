#!/usr/bin/env bash
set -euo pipefail

# Datasets list
# DATASETS=(
#   "Linux24APT"
#   "russellmitchell"
#   "santos"
#   "wardbeck"
#   "shaw"
#   "wilson"
#   "fox"
# )
DATASETS=(
  "harrison"
  "known_c1"
  "known_c2"
  "known_c3"
  "known_c4"
  "known_c5"
  "known_c6"
  "known_c7"
  "known_c8"
)

# Base directory containing the CSVs (change if needed)
BASE_DIR="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/temp/prod"

# Parser to use
PARSER="drain"

# Optional: export LOGADU_BIN if logadu is not on PATH
LOGADU_BIN="${LOGADU_BIN:-logadu}"

# Allow parallel execution: set PARALLEL>1 (export PARALLEL=4 for example)
PARALLEL="${PARALLEL:-1}"

# Internal
total=${#DATASETS[@]}
success=0
fail=0
start_all=$(date +%s)

echo "[+] Starting parsing of $total datasets with parser '$PARSER'"
echo "[+] Base dir: $BASE_DIR"
echo "[+] Parallel jobs: $PARALLEL"
echo

# A function to process one dataset
process_one() {
  local ds="$1"
  local file_candidate1="${BASE_DIR}/${ds}_all.csv"
  local lc="$(echo "$ds" | tr 'A-Z' 'a-z')"
  local file_candidate2="${BASE_DIR}/${lc}_all.csv"
  local csv=""

  if [[ -f "$file_candidate1" ]]; then
    csv="$file_candidate1"
  elif [[ -f "$file_candidate2" ]]; then
    csv="$file_candidate2"
  else
    echo "[!] $ds: CSV not found (tried: $file_candidate1 , $file_candidate2)" >&2
    return 2
  fi

  local start_ts end_ts
  start_ts=$(date +%s)
  echo "[>] Parsing ($ds) file: $csv"
  if "$LOGADU_BIN" parse "$csv" --parser "$PARSER"; then
    end_ts=$(date +%s)
    echo "[✓] $ds done in $((end_ts-start_ts))s"
    return 0
  else
    end_ts=$(date +%s)
    echo "[x] $ds failed after $((end_ts-start_ts))s" >&2
    return 1
  fi
}

# Export for subshells if parallel
export -f process_one
export BASE_DIR PARSER LOGADU_BIN

if [[ "$PARALLEL" -gt 1 ]]; then
  # Parallel branch
  printf "%s\n" "${DATASETS[@]}" | xargs -I{} -P "$PARALLEL" bash -c 'process_one "$@"' _ {}
  # Count successes by grepping logs (simple heuristic)
  # (If needed, you can improve by capturing statuses explicitly.)
else
  idx=0
  for ds in "${DATASETS[@]}"; do
    idx=$((idx+1))
    echo "---- ($idx/$total) $ds ----"
    if process_one "$ds"; then
      success=$((success+1))
    else
      rc=$?
      if [[ $rc -eq 2 ]]; then
        echo "[!] Skipped (missing file) $ds"
      fi
      fail=$((fail+1))
    fi
    echo
  done
fi

end_all=$(date +%s)
echo "[=] All done in $((end_all-start_all))s"

# If sequential we have counters
if [[ "$PARALLEL" -le 1 ]]; then
  echo "[=] Success: $success  Fail: $fail  Total: $total"
fi