#!/usr/bin/env bash
set -euo pipefail

# ---- Settings (edit if your paths differ) ----
BASE="/Scratch/sensing/proof_of_concept"
OPTUNA_DB="$BASE/optuna.db"
MLFLOW_DIR="$BASE/mlruns"
TB_DIR="$BASE/results/tb"
RESULTS_DIR="$BASE/results"

# ---- CLI flags ----
ARCHIVE=false
FORCE=false
KEEP_RESULTS=false

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--archive] [--force] [--keep-results]

Options:
  --archive       Move data into _archive/<timestamp>/ instead of deleting.
  --force         Skip confirmation prompt.
  --keep-results  Do NOT touch results/ (figures/CSVs). Only clear dashboards (Optuna/MLflow/TB).

Examples:
  $(basename "$0")                 # interactive delete
  $(basename "$0") --archive       # archive instead of delete
  $(basename "$0") --force         # delete without prompt
  $(basename "$0") --archive --keep-results
USAGE
}

# Parse args
while [[ ${1:-} ]]; do
  case "$1" in
    --archive) ARCHIVE=true ;;
    --force) FORCE=true ;;
    --keep-results) KEEP_RESULTS=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

# ---- Check running services and stop them (best-effort) ----
echo "[i] Stopping any running dashboards (best-effort)…"
pkill -f "optuna-dashboard" 2>/dev/null || true
pkill -f "mlflow ui"        2>/dev/null || true
pkill -f "tensorboard"      2>/dev/null || true

# ---- Plan what to remove/move ----
declare -a TARGETS=()
[[ -e "$OPTUNA_DB" ]] && TARGETS+=("$OPTUNA_DB")
[[ -d "$MLFLOW_DIR" ]] && TARGETS+=("$MLFLOW_DIR")
[[ -d "$TB_DIR" ]] && TARGETS+=("$TB_DIR")

if [[ "$KEEP_RESULTS" == "false" ]]; then
  [[ -d "$RESULTS_DIR" ]] && TARGETS+=("$RESULTS_DIR")
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "[i] Nothing to clean. Exiting."
  exit 0
fi

echo "[i] Targets:"
for t in "${TARGETS[@]}"; do echo "   - $t"; done

# ---- Confirmation ----
if [[ "$FORCE" == "false" ]]; then
  read -r -p "Proceed with $( $ARCHIVE && echo 'ARCHIVE' || echo 'DELETE' )? [y/N] " ans
  [[ "${ans,,}" == "y" ]] || { echo "Aborted."; exit 1; }
fi

# ---- Archive or Delete ----
if $ARCHIVE; then
  TS=$(date +%Y%m%d-%H%M%S)
  DEST="$BASE/_archive/$TS"
  echo "[i] Archiving to: $DEST"
  mkdir -p "$DEST"
  for t in "${TARGETS[@]}"; do
    bn=$(basename "$t")
    mv "$t" "$DEST/$bn" 2>/dev/null || true
  done
else
  echo "[i] Deleting targets…"
  for t in "${TARGETS[@]}"; do
    rm -rf "$t" 2>/dev/null || true
  done
fi

# ---- Recreate fresh dirs for next run ----
mkdir -p "$MLFLOW_DIR"
mkdir -p "$TB_DIR"
mkdir -p "$RESULTS_DIR/figures"

echo "[✓] Cleanup complete."
echo "[i] To restart dashboards (they’ll be empty until you run training):"
echo "    optuna-dashboard \"sqlite:////$BASE/optuna.db\" --host 127.0.0.1 --port 8081"
echo "    mlflow ui --backend-store-uri file:$MLFLOW_DIR --host 127.0.0.1 --port 5000"
echo "    tensorboard --logdir $TB_DIR --host 127.0.0.1 --port 6006"
