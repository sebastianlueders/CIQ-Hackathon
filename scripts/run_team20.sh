#!/usr/bin/env bash
# scripts/run_team20.sh
# Turnkey launcher for Crucible on Team 20 GPUs (definitionId: team20: CIQ-CHI-SPARK39 & CIQ-CHI-SPARK40).
#
# Usage:
#   ./scripts/run_team20.sh                        # auto-selects fuzzball_mcp if `fuzzball` is on PATH, else simulator
#   ./scripts/run_team20.sh --backend fuzzball_mcp # force live Fuzzball 4.2 MCP execution on team20
#   ./scripts/run_team20.sh --backend fuzzball     # force live Fuzzball CLI execution on team20
#   ./scripts/run_team20.sh --profile stretch_80b  # run 2-node 80B stretch goal across SPARK39 + SPARK40
#   ./scripts/run_team20.sh --no-open              # skip opening browser window automatically

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="demo_30b"
BACKEND="auto"
PORT="${CRUCIBLE_PORT:-8000}"
HOST="${CRUCIBLE_HOST:-127.0.0.1}"
OPEN_BROWSER=1
KEEP_SERVING=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --no-open)
      OPEN_BROWSER=0
      shift
      ;;
    --exit-after-run)
      KEEP_SERVING=0
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# 1. Resolve Python virtual environment & install editable Crucible + UI deps
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  VENV_BIN="${REPO_ROOT}/.venv/bin"
elif command -v crucible >/dev/null 2>&1; then
  VENV_BIN="$(dirname "$(command -v crucible)")"
else
  echo "[team20] Creating local Python virtual environment at ${REPO_ROOT}/.venv ..."
  python3 -m venv "${REPO_ROOT}/.venv"
  VENV_BIN="${REPO_ROOT}/.venv/bin"
  "${VENV_BIN}/pip" install --upgrade pip
  "${VENV_BIN}/pip" install -e ".[ui,test]"
fi

CRUCIBLE="${VENV_BIN}/crucible"

# 2. Auto-select execution backend if set to 'auto'
if [[ "$BACKEND" == "auto" ]]; then
  if command -v fuzzball >/dev/null 2>&1; then
    if fuzzball mcp serve --help >/dev/null 2>&1; then
      BACKEND="fuzzball_mcp"
      echo "[team20] Detected 'fuzzball mcp serve' -> selecting live CIQ Fuzzball 4.2 MCP backend (definitionId: team20)"
    else
      BACKEND="fuzzball"
      echo "[team20] Detected 'fuzzball' CLI (without 'mcp serve') -> selecting live Fuzzball CLI backend (definitionId: team20)"
    fi
  else
    BACKEND="simulator"
    echo "[team20] 'fuzzball' CLI not found on PATH -> selecting calibrated Team 20 GB10 simulator"
  fi
fi

# 3. Discover active Team 20 cluster & validate Fuzzfiles against SPARK39/SPARK40 hardware envelope
echo "[team20] Running cluster discovery & Fuzzfile validation ..."
"$CRUCIBLE" discover
"$CRUCIBLE" validate-yaml workflows/00-fetch.yaml
"$CRUCIBLE" validate-yaml workflows/crucible-studio.yaml
"$CRUCIBLE" validate-yaml workflows/examples/final-lammps-qwen3-30b.yaml

if command -v fuzzball >/dev/null 2>&1; then
  fuzzball workflow validate workflows/00-fetch.yaml || true
fi

# 4. Start Crucible Live Studio on http://127.0.0.1:8000
RUN_DIR="${REPO_ROOT}/runs/team20-live"
mkdir -p "$RUN_DIR"

echo "[team20] Starting Crucible Live Studio at http://${HOST}:${PORT} ..."
"$CRUCIBLE" serve --run-dir "$RUN_DIR" --host "$HOST" --port "$PORT" >"${RUN_DIR}/studio.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 40); do
  if curl -fsS "http://${HOST}:${PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

if [[ "$OPEN_BROWSER" -eq 1 ]]; then
  if command -v open >/dev/null 2>&1; then
    open "http://${HOST}:${PORT}" || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://${HOST}:${PORT}" >/dev/null 2>&1 || true
  fi
fi

# 5. Trigger live Team 20 autoresearch run through the Live Studio API so UI streams every trial!
echo "[team20] Triggering 4-stage autoresearch run (profile=${PROFILE}, backend=${BACKEND}, definitionId=team20) ..."
curl -fsS -X POST "http://${HOST}:${PORT}/api/run" \
  -H "Content-Type: application/json" \
  -d "{\"profile\":\"${PROFILE}\",\"backend\":\"${BACKEND}\",\"fuzzfile_version\":\"v4\",\"definition_id\":\"team20\",\"async_mode\":true}" \
  >/dev/null

while true; do
  STATUS_JSON="$(curl -fsS "http://${HOST}:${PORT}/api/live-status")"
  STATE="$("${VENV_BIN}/python" -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d["state"])' <<<"$STATUS_JSON")"
  LABEL="$("${VENV_BIN}/python" -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d["stage_label"])' <<<"$STATUS_JSON")"
  echo "  [${STATE}] ${LABEL}"
  if [[ "$STATE" == "completed" ]]; then
    break
  fi
  if [[ "$STATE" == "failed" ]]; then
    ERR="$("${VENV_BIN}/python" -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d.get("error") or "")' <<<"$STATUS_JSON")"
    echo "[team20] Live run stopped with error: ${ERR}" >&2
    echo "[team20] Tip: verify 'fuzzball context list' / 'fuzzball workflow list' or re-run with '--backend simulator' from http://${HOST}:${PORT}" >&2
    break
  fi
  sleep 1
done

echo ""
echo "[team20] Live UI running at: http://${HOST}:${PORT}"
if [[ "$KEEP_SERVING" -eq 1 ]]; then
  echo "[team20] Press Ctrl+C to stop Crucible Live Studio."
  wait "$SERVER_PID"
fi
