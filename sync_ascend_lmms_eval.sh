#!/usr/bin/env bash
set -euo pipefail

ASCEND_PIP_INDEX_URL="${ASCEND_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
ASCEND_PIP_TRUSTED_HOST="${ASCEND_PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_VIDEO_DEPS="${INSTALL_VIDEO_DEPS:-0}"

PIP_GLOBAL_ARGS=(
  --isolated
)

PIP_INSTALL_ARGS=(
  --index-url "${ASCEND_PIP_INDEX_URL}"
  --trusted-host "${ASCEND_PIP_TRUSTED_HOST}"
  --prefer-binary
)

echo "Using pip index: ${ASCEND_PIP_INDEX_URL}"
echo "Using pip trusted host: ${ASCEND_PIP_TRUSTED_HOST}"
echo "pip will run with --isolated to ignore environment/global pip config."

echo "============================================================"
echo "[1/5] Checking bundled Ascend PyTorch and lmms-engine"
echo "============================================================"
"${PYTHON_BIN}" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"Failed to import torch from the base image: {exc}")

try:
    import torch_npu
except Exception as exc:
    raise SystemExit(f"Failed to import torch_npu from the base image: {exc}")

try:
    import lmms_engine
    import lmms_engine.models.nanovlm  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "lmms-engine NanoVLM import failed. Run lmms-engine/sync_ascend_nanovlm.sh "
        f"or install lmms-engine with `pip install --no-deps -e .` first: {exc}"
    )

print("python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("torch path:", torch.__file__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("torch_npu path:", torch_npu.__file__)
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
print("lmms_engine:", getattr(lmms_engine, "__version__", "editable"))
print("lmms-engine NanoVLM imports ok")
PY

echo "============================================================"
echo "[2/5] Upgrading pip tooling with Aliyun mirror"
echo "============================================================"
"${PYTHON_BIN}" -m pip "${PIP_GLOBAL_ARGS[@]}" install \
  "${PIP_INSTALL_ARGS[@]}" \
  --upgrade pip setuptools wheel

echo "============================================================"
echo "[3/5] Installing lmms-eval runtime dependencies"
echo "============================================================"
"${PYTHON_BIN}" -m pip "${PIP_GLOBAL_ARGS[@]}" install \
  "${PIP_INSTALL_ARGS[@]}" \
  "evaluate>=0.4.0" \
  "httpx>=0.23.3" \
  "aiohttp" \
  "numexpr" \
  "peft>=0.2.0" \
  "pybind11>=2.6.2" \
  "pytablewriter" \
  "sacrebleu>=1.5.0" \
  "scikit-learn>=0.24.1" \
  "timm" \
  "ftfy" \
  "openai" \
  "av<16.0.0" \
  "nltk" \
  "sentencepiece" \
  "yt-dlp" \
  "pycocoevalcap" \
  "tqdm-multiprocess" \
  "transformers-stream-generator" \
  "zstandard" \
  "sympy" \
  "latex2sympy2" \
  "mpmath" \
  "Jinja2" \
  "openpyxl" \
  "tenacity>=8.3.0" \
  "tiktoken" \
  "packaging" \
  "zss" \
  "protobuf" \
  "sentence-transformers" \
  "python-dotenv" \
  "math-verify"

echo "============================================================"
echo "[3.5/5] Optional video dependencies"
echo "============================================================"
if [[ "${INSTALL_VIDEO_DEPS}" == "1" ]]; then
  echo "Installing decord for video tasks. Skip this for image-only MME evals."
  "${PYTHON_BIN}" -m pip "${PIP_GLOBAL_ARGS[@]}" install \
    "${PIP_INSTALL_ARGS[@]}" \
    "decord" \
    "qwen-vl-utils>=0.0.14"
else
  echo "Skipping decord. Set INSTALL_VIDEO_DEPS=1 if you need video benchmarks."
fi

echo "============================================================"
echo "[4/5] Installing lmms-eval in editable mode without deps"
echo "============================================================"
"${PYTHON_BIN}" -m pip "${PIP_GLOBAL_ARGS[@]}" install \
  "${PIP_INSTALL_ARGS[@]}" \
  --no-deps \
  -e .

echo "============================================================"
echo "[5/5] Final import and NanoVLM registration check"
echo "============================================================"
"${PYTHON_BIN}" - <<'PY'
import torch
import torch_npu
import transformers
import datasets
import evaluate
import lmms_engine
import lmms_eval
from lmms_eval.models import get_model
from lmms_eval.models.chat.nanovlm import NanoVLM

model_cls = get_model("nanovlm")

print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("evaluate:", evaluate.__version__)
print("lmms_engine:", getattr(lmms_engine, "__version__", "editable"))
print("lmms_eval:", getattr(lmms_eval, "__version__", "editable"))
print("NanoVLM class:", NanoVLM.__name__)
print("registered nanovlm:", model_cls.__name__)
print("lmms-eval NanoVLM imports ok")
PY

echo "============================================================"
echo "Ascend lmms-eval supplemental sync completed."
echo "For a first smoke test, use:"
echo "  ASCEND_RT_VISIBLE_DEVICES=0 python -m lmms_eval \\"
echo "    --model nanovlm \\"
echo "    --model_args pretrained=/path/to/checkpoint-merged,device=npu:0,attn_implementation=sdpa,use_cache=false \\"
echo "    --tasks mme --batch_size 1 --limit 10"
echo "============================================================"
