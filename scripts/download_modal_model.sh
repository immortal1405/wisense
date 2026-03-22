#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-}"
FILE_NAME="${2:-best_model.pt}"
LOCAL_DIR="${3:-outputs_modal}"

if [[ -z "${RUN_NAME}" ]]; then
  echo "Usage: $0 <run_name> [file_name] [local_dir]"
  exit 1
fi

source .venv/bin/activate
modal run src/modal_runner/train_modal.py --action download --run-name "${RUN_NAME}" --file-name "${FILE_NAME}" --local-dir "${LOCAL_DIR}"
