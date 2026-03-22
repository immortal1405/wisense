#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${1:-base.yaml}"
EPOCHS="${2:-20}"
MAX_BATCHES="${3:-0}"

source .venv/bin/activate
modal run src/modal_runner/train_modal.py --config-name "${CONFIG_NAME}" --epochs "${EPOCHS}" --max-batches "${MAX_BATCHES}"
