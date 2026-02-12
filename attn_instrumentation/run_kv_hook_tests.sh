#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY_BIN="python"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi
if ! "$PY_BIN" -c "import torch" >/dev/null 2>&1; then
  echo "torch is not installed in the selected environment: $PY_BIN" >&2
  exit 1
fi
if ! "$PY_BIN" -c "import pytest" >/dev/null 2>&1; then
  echo "pytest is not installed in the selected environment: $PY_BIN" >&2
  exit 1
fi
"$PY_BIN" -m pytest -q \
  --noconftest \
  tests/model_executor/layers/attention/test_kv_hook_utils.py \
  tests/v1/engine/test_output_processor_kv_hook.py \
  tests/entrypoints/openai/test_chat_kv_hook_protocol.py \
  tests/v1/worker/test_gpu_model_runner_kv_hook.py \
  "$@"
