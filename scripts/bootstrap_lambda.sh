#!/bin/bash
# bootstrap_lambda.sh -- One-time setup for GRPO training on Lambda Cloud VMs.
#
# Usage:
#   export REPT_ROOT=/home/ubuntu/ReasoningEconomicsPT
#   export REPT_VENV=/home/ubuntu/.venvs/rept-lambda
#   export REPT_FS_NAME=<lambda-filesystem-name>   # optional if REPT_DATA_ROOT set
#   export REPT_DATA_ROOT=/lambda/nfs/<fs-name>/rept  # optional override
#   bash scripts/bootstrap_lambda.sh

set -euo pipefail

: "${REPT_ROOT:?Set REPT_ROOT to the absolute path of ReasoningEconomicsPT}"
: "${REPT_VENV:?Set REPT_VENV to the absolute path for the Python venv}"

if [[ -n "${REPT_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$REPT_DATA_ROOT"
elif [[ -n "${REPT_FS_NAME:-}" ]]; then
    DATA_ROOT="/lambda/nfs/${REPT_FS_NAME}/rept"
else
    DATA_ROOT="/lambda/nfs/rept"
fi

REPT_REQUIREMENTS_FILE="${REPT_REQUIREMENTS_FILE:-$REPT_ROOT/requirements.txt}"
PYTORCH_WHEEL_INDEX="${PYTORCH_WHEEL_INDEX:-}"

CACHE_ROOT="${DATA_ROOT}/cache"
RUNS_ROOT="${DATA_ROOT}/runs"
REPT_OUTPUT_DIR="${REPT_OUTPUT_DIR:-${RUNS_ROOT}/grpo_train_lambda}"

echo "=== Lambda Bootstrap ==="
echo "  REPT_ROOT             = $REPT_ROOT"
echo "  REPT_VENV             = $REPT_VENV"
echo "  DATA_ROOT             = $DATA_ROOT"
echo "  Requirements          = $REPT_REQUIREMENTS_FILE"
if [[ -n "$PYTORCH_WHEEL_INDEX" ]]; then
    echo "  PyTorch extra index   = $PYTORCH_WHEEL_INDEX"
else
    echo "  PyTorch extra index   = <default pip indexes>"
fi
echo ""

if [[ ! -d "$REPT_ROOT" ]]; then
    echo "[ERROR] REPT_ROOT does not exist: $REPT_ROOT"
    exit 1
fi

if [[ ! -f "$REPT_REQUIREMENTS_FILE" ]]; then
    echo "[ERROR] requirements file not found: $REPT_REQUIREMENTS_FILE"
    exit 1
fi

echo ">>> Preparing filesystem directories..."
mkdir -p "$CACHE_ROOT/pip" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/tmp" "$REPT_OUTPUT_DIR"

if [[ -d "$REPT_VENV" ]]; then
    echo ">>> Reusing existing venv at $REPT_VENV"
else
    echo ">>> Creating venv at $REPT_VENV"
    python3 -m venv "$REPT_VENV"
fi

# shellcheck source=/dev/null
source "$REPT_VENV/bin/activate"
echo "    Python: $(python --version) ($(which python))"

export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TMPDIR="$CACHE_ROOT/tmp"

echo ">>> Cache config"
echo "    PIP_CACHE_DIR       = $PIP_CACHE_DIR"
echo "    HF_HOME             = $HF_HOME"
echo "    TRANSFORMERS_CACHE  = $TRANSFORMERS_CACHE"
echo "    TMPDIR              = $TMPDIR"

echo ">>> Upgrading pip..."
pip install --quiet --upgrade pip

echo ">>> Installing dependencies..."
if [[ -n "$PYTORCH_WHEEL_INDEX" ]]; then
    pip install -r "$REPT_REQUIREMENTS_FILE" --extra-index-url "$PYTORCH_WHEEL_INDEX"
else
    pip install -r "$REPT_REQUIREMENTS_FILE"
fi

echo ""
echo ">>> GPU visibility check"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
    echo "[WARN] nvidia-smi not found in PATH"
fi

echo ""
echo ">>> Smoke-testing critical imports..."
python - <<'PY'
import torch
import vllm
import trl
import transformers
import openenv

print(f"  torch        {torch.__version__}  CUDA={torch.cuda.is_available()}")
print(f"  vllm         {vllm.__version__}")
print(f"  trl          {trl.__version__}")
print(f"  transformers {transformers.__version__}")
print("  openenv-core OK")
PY

echo ""
echo "=== Bootstrap complete ==="
echo "Activate with: source \"$REPT_VENV/bin/activate\""
echo "Suggested run dir: $REPT_OUTPUT_DIR"
