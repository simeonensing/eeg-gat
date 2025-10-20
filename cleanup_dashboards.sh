#!/usr/bin/env bash
set -euo pipefail

# =========================
# Settings (edit if needed)
# =========================
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # script dir as base
OPTUNA_DB="$BASE/tracking/optuna.db"
MLFLOW_DIR="$BASE/tracking/mlruns"
TB_DIR="$BASE/tracking/tb"
RESULTS_DIR="$BASE/results"

# Default ports (can be overridden via env)
OPTUNA_PORT="${OPTUNA_PORT:-8080}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
TB_PORT="${TB_PORT:-6006}"

# =========================
# CLI flags
# =========================
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
  $(basename "$0")
  $(basename "$0") --archive
  $(basename "$0") --force
  $(basename "$0") --archive --keep-results
USAGE
}

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

# =========================
# Kill helpers
# =========================
kill_by_port() {
  local port="$1"
  # lsof (macOS/Linux)
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids=$(lsof -ti tcp:"$port" -sTCP:LISTEN || true)
    if [[ -n "$pids" ]]; then
      echo "[i] Killing PIDs on port $port: $pids"
      kill $pids 2>/dev/null || true
      sleep 0.3
      pids=$(lsof -ti tcp:"$port" -sTCP:LISTEN || true)
      [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
    fi
    return
  fi
  # ss (Linux)
  if command -v ss >/dev/null 2>&1; then
    local pids
    pids=$(ss -lptn "sport = :$port" 2>/dev/null | awk -F',' '/users/ {print $2}' | sed -E 's/ pid=([0-9]+).*/\1/' | sort -u)
    if [[ -n "$pids" ]]; then
      echo "[i] Killing PIDs on port $port: $pids"
      kill $pids 2>/dev/null || true
      sleep 0.3
      for pid in $pids; do kill -9 "$pid" 2>/dev/null || true; done
    fi
    return
  fi
  # fuser (Linux)
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "$port"/tcp 2>/dev/null || true
    return
  fi
  echo "[!] Could not kill by port $port (no lsof/ss/fuser)."
}

kill_dashboards() {
  echo "[i] Stopping any running dashboards (best-effort)…"
  # Optuna: binary or module
  pkill -f "[o]ptuna-dashboard"         2>/dev/null || true
  pkill -f "[o]ptuna_dashboard"         2>/dev/null || true
  pkill -f "python.*optuna_dashboard"   2>/dev/null || true

  # MLflow: binary or python -m
  pkill -f "[m]lflow ui"                2>/dev/null || true
  pkill -f "[m]lflow server"            2>/dev/null || true
  pkill -f "python.*mlflow.*(ui|server)" 2>/dev/null || true

  # TensorBoard: binary or module
  pkill -f "[t]ensorboard"              2>/dev/null || true
  pkill -f "tensorboard.main"           2>/dev/null || true
  pkill -f "python.*tensorboard"        2>/dev/null || true

  # Also free the well-known ports
  kill_by_port "$OPTUNA_PORT"
  kill_by_port "$MLFLOW_PORT"
  kill_by_port "$TB_PORT"
}

# =========================
# Plan deletion/archive
# =========================
declare -a TARGETS=()
[[ -e "$OPTUNA_DB" ]] && TARGETS+=("$OPTUNA_DB")
[[ -d "$MLFLOW_DIR" ]] && TARGETS+=("$MLFLOW_DIR")
[[ -d "$TB_DIR" ]] && TARGETS+=("$TB_DIR")

if [[ "$KEEP_RESULTS" == "false" ]]; then
  [[ -d "$RESULTS_DIR" ]] && TARGETS+=("$RESULTS_DIR")
fi

# =========================
# Execute
# =========================
kill_dashboards

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "[i] Nothing to clean. Ensuring fresh dirs exist…"
else
  echo "[i] Targets:"
  for t in "${TARGETS[@]}"; do echo "   - $t"; done

  if [[ "$FORCE" == "false" ]]; then
    read -r -p "Proceed with $( $ARCHIVE && echo 'ARCHIVE' || echo 'DELETE' )? [y/N] " ans
    [[ "${ans,,}" == "y" ]] || { echo "Aborted."; exit 1; }
  fi

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
fi

# Recreate fresh dirs for next run
mkdir -p "$MLFLOW_DIR"
mkdir -p "$TB_DIR"
mkdir -p "$RESULTS_DIR/figures"

echo "[✓] Cleanup complete."

MLFLOW_URI="file:$MLFLOW_DIR"
OPTUNA_DSN="sqlite:///$OPTUNA_DB"

echo "[i] To restart dashboards (they’ll be empty until you run training):"
echo "    optuna-dashboard \"$OPTUNA_DSN\" --host 127.0.0.1 --port $OPTUNA_PORT"
echo "    mlflow ui --backend-store-uri \"$MLFLOW_URI\" --host 127.0.0.1 --port $MLFLOW_PORT"
echo "    tensorboard --logdir \"$TB_DIR\" --host 127.0.0.1 --port $TB_PORT"

# Final sanity (optional): show listeners
if command -v lsof >/dev/null 2>&1; then
  echo "[i] Ports in LISTEN after cleanup (should be empty for 8080/$OPTUNA_PORT, $MLFLOW_PORT, $TB_PORT):"
  lsof -i tcp:"$OPTUNA_PORT","$MLFLOW_PORT","$TB_PORT" -sTCP:LISTEN || true
fi
