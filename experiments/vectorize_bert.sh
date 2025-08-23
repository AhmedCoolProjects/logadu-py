#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
  "Linux24APT"
  "russellmitchell"
  "santos"
  "wardbeck"
  "shaw"
  "wilson"
  "fox"
)

# Drain output directory (gpath)
GPATH="${GPATH:-/home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/temp/prod/drain}"
LOGADU_BIN="${LOGADU_BIN:-logadu}"
PARALLEL="${PARALLEL:-1}"

total=${#DATASETS[@]}
success=0
fail=0
start_all=$(date +%s)

echo "[+] BERT vectorization for $total datasets"
echo "[+] gpath: $GPATH"
echo "[+] Parallel jobs: $PARALLEL"
echo

process_one() {
  local dataset="$1"
  local start_ts end_ts
  start_ts=$(date +%s)

  if [[ ! -d "$GPATH" ]]; then
    echo "[x] gpath not found: $GPATH" >&2
    return 3
  fi

  echo "[>] Vectorizing ($dataset)"
  if "$LOGADU_BIN" vectorizebert "$GPATH" "$dataset"; then
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
export LOGADU_BIN GPATH

if [[ "$PARALLEL" -gt 1 ]]; then
  tmp_fifo="$(mktemp -u)"; mkfifo "$tmp_fifo"
  exec 3<>"$tmp_fifo"
  rm -f "$tmp_fifo"
  printf "%s\n" "${DATASETS[@]}" | xargs -I{} -P "$PARALLEL" bash -c 'if process_one "$1"; then echo OK >&3; else echo FAIL >&3; fi' _ {}
  while read -r r <&3; do
    [[ "$r" == OK ]] && success=$((success+1)) || fail=$((fail+1))
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
echo "[=] Finished in $((end_all-start_all))s"
echo "[=] Success: $success  Fail: $fail  Total: $total"
exit $(( fail > 0 ? 1 : 0 ))