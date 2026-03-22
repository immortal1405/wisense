#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

log() {
  echo "[start.sh] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' is not installed or not in PATH." >&2
    exit 1
  fi
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    log "Stopping backend (PID: $BACKEND_PID)"
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi

  if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    log "Stopping frontend (PID: $FRONTEND_PID)"
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi

  wait >/dev/null 2>&1 || true
  exit "$exit_code"
}

require_cmd python3
require_cmd npm

log "Project root: $ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating Python virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

log "Installing Python dependencies..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV_DIR/bin/pip" install -r "$ROOT_DIR/requirements.txt"

log "Installing backend dependencies..."
(
  cd "$ROOT_DIR/server"
  npm install
)

log "Installing frontend dependencies..."
(
  cd "$ROOT_DIR/web"
  npm install
)

trap cleanup EXIT INT TERM

log "Starting backend on http://localhost:4000 ..."
(
  cd "$ROOT_DIR/server"
  npm run start:dev
) &
BACKEND_PID=$!

log "Starting frontend on http://localhost:5173 ..."
(
  cd "$ROOT_DIR/web"
  npm run dev
) &
FRONTEND_PID=$!

log "Setup complete. Services are running:"
log "- Backend:  http://localhost:4000/api/health"
log "- Frontend: http://localhost:5173"
log "Press Ctrl+C to stop both services."

wait
