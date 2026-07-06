#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
WORKER_COUNT="${WORKER_COUNT:-8}"
MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/output/nanovlm_stage2_ascend/checkpoint-11540-merged}"
TASKS="${TASKS:-mme}"
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
USE_CACHE="${USE_CACHE:-false}"
VERBOSITY="${VERBOSITY:-INFO}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/ma-user/work/output/nanovlm_mme_eval_ascend}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${VISIBLE_DEVICES}}"
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_PATH}"

EVAL_ARGS=(
  --model nanovlm
  --model_args "pretrained=${MODEL_PATH},device=npu,worker_count=${WORKER_COUNT},attn_implementation=${ATTN_IMPLEMENTATION},use_cache=${USE_CACHE}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --log_samples
  --output_path "${OUTPUT_PATH}"
  --verbosity "${VERBOSITY}"
)

if [[ -n "${LIMIT}" ]]; then
  EVAL_ARGS+=(--limit "${LIMIT}")
fi

echo "============================================================"
echo "NanoVLM Ascend MME eval"
echo "============================================================"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "TASKS: ${TASKS}"
echo "LIMIT: ${LIMIT:-<full>}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "WORKER_COUNT: ${WORKER_COUNT}"
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
from huggingface_hub import get_token
from lmms_eval.models import get_model

model_path = Path("${MODEL_PATH}")
if not model_path.exists():
    raise SystemExit(f"MODEL_PATH does not exist: {model_path}")

visible_devices = [item for item in "${ASCEND_RT_VISIBLE_DEVICES}".split(",") if item]
worker_count = int("${WORKER_COUNT}")
if worker_count != 8:
    print(f"warning: WORKER_COUNT is {worker_count}, expected 8 for full 8-NPU eval.")
if len(visible_devices) < worker_count:
    print(f"warning: visible device count {len(visible_devices)} is smaller than WORKER_COUNT {worker_count}.")

print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
print("visible devices:", os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
print("hf token available:", bool(get_token()))
print("nanovlm registered:", get_model("nanovlm").__name__)
PY

"${PYTHON_BIN}" -m lmms_eval "${EVAL_ARGS[@]}"

echo "============================================================"
echo "NanoVLM Ascend MME eval completed."
echo "Outputs saved to: ${OUTPUT_PATH}"
echo "============================================================"
