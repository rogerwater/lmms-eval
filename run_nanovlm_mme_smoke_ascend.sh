#!/usr/bin/env bash
set -euo pipefail

# Single-NPU NanoVLM smoke test for Ascend/snt9b.
# Run from the lmms-eval repository root after sync_ascend_lmms_eval.sh.
# Do not launch this with torchrun or accelerate launch.

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE_ID="${DEVICE_ID:-0}"
MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/output/nanovlm_stage2_ascend/checkpoint-11540-merged}"
TASKS="${TASKS:-mme}"
LIMIT="${LIMIT:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
USE_CACHE="${USE_CACHE:-false}"
VERBOSITY="${VERBOSITY:-DEBUG}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/ma-user/work/eval_outputs/nanovlm_mme_smoke}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${DEVICE_ID}}"
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_PATH}"

echo "============================================================"
echo "NanoVLM Ascend single-NPU smoke eval"
echo "============================================================"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "TASKS: ${TASKS}"
echo "LIMIT: ${LIMIT}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "HF_HOME: ${HF_HOME}"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  echo "HF_ENDPOINT: ${HF_ENDPOINT}"
fi
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "============================================================"

"${PYTHON_BIN}" - <<PY
import os
from pathlib import Path

import torch
import torch_npu
import lmms_engine.models.nanovlm  # noqa: F401
from lmms_eval.models import get_model

model_path = Path("${MODEL_PATH}")
if not model_path.exists():
    raise SystemExit(f"MODEL_PATH does not exist: {model_path}")

print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
print("visible devices:", os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
print("nanovlm registered:", get_model("nanovlm").__name__)
PY

"${PYTHON_BIN}" -m lmms_eval \
  --model nanovlm \
  --model_args "pretrained=${MODEL_PATH},device=npu:0,attn_implementation=${ATTN_IMPLEMENTATION},use_cache=${USE_CACHE}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --limit "${LIMIT}" \
  --log_samples \
  --output_path "${OUTPUT_PATH}" \
  --verbosity "${VERBOSITY}"

echo "============================================================"
echo "NanoVLM smoke eval completed."
echo "Outputs saved to: ${OUTPUT_PATH}"
echo "============================================================"
