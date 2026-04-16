#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${1:-har_activity_phase2_expact.yaml}"
EPOCHS="${2:-30}"
MAX_BATCHES="${3:-0}"

source .venv/bin/activate
modal run src/modal_runner/train_modal.py \
  --config-name "${CONFIG_NAME}" \
  --module-name "src.training.train_har_activity" \
  --epochs "${EPOCHS}" \
  --max-batches "${MAX_BATCHES}"
