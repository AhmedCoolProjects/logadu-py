#!/usr/bin/env bash
set -euo pipefail

# Datasets list (same as parse script)
DATASETS=(
  "Linux24APT"
  "russellmitchell"
  "santos"
  "wardbeck"
  "shaw"
  "wilson"
  "fox"
)

EMBEDDINGS="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation/crawl-300d-2M.vec"
GPATH="/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/temp/prod/drain"
LOGADU_BIN="${LOGADU_BIN:-logadu}"
PARALLEL="${PARALLEL:-1}"

total=${#DATASETS[@]}
start_all=$(date +%s)
echo "[+] Vectorizing $total datasets (fasttext)"
echo "[+] Embeddings: $EMBEDDINGS"
echo "[+] gpath: $GPATH"
echo "[+] Parallel jobs: $PARALLEL"
echo

process_one() {
  local dataset="$1"

  if [[ ! -d "$GPATH/$dataset" && ! -f "$GPATH/${dataset}.parquet" && ! -f "$GPATH/${dataset}.csv" ]]; then
    echo "[!] $dataset: no obvious drain artifacts under $GPATH (continuing anyway)" >&2
  fi

  local start_ts end_ts
  start_ts=$(date +%s)
  echo "[>] Vectorizing ($dataset) --dataset $dataset"
  if "$LOGADU_BIN" vectorize fasttext "$EMBEDDINGS" --dataset "$dataset" --gpath "$GPATH"; then
    end_ts=$(date +%s)
    echo "[✓] $dataset done in $((end_ts-start_ts))s"
    return 0
  else
    end_ts=$(date +%s)
    echo "[x] $dataset failed after $((end_ts-start_ts))s" >&2
    return 1
  fi
}

export -f process_one
export LOGADU_BIN EMBEDDINGS GPATH

success=0
fail=0

if [[ "$PARALLEL" -gt 1 ]]; then
  tmp_fifo="$(mktemp -u)"; mkfifo "$tmp_fifo"
  exec 3<>"$tmp_fifo"
  rm -f "$tmp_fifo"
  printf "%s\n" "${DATASETS[@]}" | xargs -I{} -P "$PARALLEL" bash -c 'if process_one "$1"; then echo "OK" >&3; else echo "FAIL" >&3; fi' _ {}
  while read -r line <&3; do
    if [[ "$line" == "OK" ]]; then
      success=$((success+1))
    else
      fail=$((fail+1))
    fi
  done
  exec 3>&-
else
  idx=0
  for ds in "${DATASETS[@]}"; do
    idx=$((idx+1))
    echo "---- ($idx/$total) $ds ----"
    if process_one "$ds"; then
      success=$((success+1))
    else
      fail=$((fail+1))
    fi
    echo
  done
fi

end_all=$(date +%s)
echo "[=] All vectorizations done in $((end_all-start_all))s"
echo "[=] Success: $success  Fail: $fail  Total: