#!/bin/bash
# bootstrap_lambda.sh -- One-time setup for GRPO training on Lambda Cloud VMs.
#
# Usage:
#   export REPT_ROOT=/home/ubuntu/ReasoningEconomicsPT
#   export REPT_VENV=/home/ubuntu/.venvs/rept-lambda
#   export REPT_FS_NAME=<lambda-filesystem-name>   # optional if REPT_DATA_ROOT set
#   export REPT_DATA_ROOT=/lambda/nfs/<fs-name>/rept  # optional override
#   export PYTORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu121  # recommended for torch wheels
#   export REPT_REQUIREMENTS_FILE="$REPT_ROOT/requirements.txt"  # override default pin file if needed
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

REPT_REQUIREMENTS_FILE="${REPT_REQUIREMENTS_FILE:-$REPT_ROOT/requirements.lambda.txt}"
PYTORCH_WHEEL_INDEX="${PYTORCH_WHEEL_INDEX:-}"
REPT_PYTHON_BIN="${REPT_PYTHON_BIN:-auto}"
REPT_VENV_SYSTEM_SITE_PACKAGES="${REPT_VENV_SYSTEM_SITE_PACKAGES:-auto}"
REPT_SKIP_TORCH_INSTALL="${REPT_SKIP_TORCH_INSTALL:-auto}"
REPT_RECREATE_VENV="${REPT_RECREATE_VENV:-0}"

ARCH="$(uname -m)"

resolve_python_bin() {
    local requested="$1"
    shift
    local candidate
    local -a candidates=("$@")

    if [[ "$requested" != "auto" ]]; then
        echo "$requested"
        return 0
    fi

    for candidate in "${candidates[@]}"; do
        [[ -z "$candidate" ]] && continue
        if [[ "$candidate" == */* ]]; then
            [[ -x "$candidate" ]] || continue
        else
            command -v "$candidate" >/dev/null 2>&1 || continue
        fi
        if "$candidate" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
        then
            echo "$candidate"
            return 0
        fi
    done

    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
    else
        echo "python"
    fi
}

if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    REPT_PYTHON_BIN="$(resolve_python_bin "$REPT_PYTHON_BIN" /opt/conda/bin/python python python3 python3.11)"
else
    REPT_PYTHON_BIN="$(resolve_python_bin "$REPT_PYTHON_BIN" python3 python)"
fi

if [[ "$REPT_VENV_SYSTEM_SITE_PACKAGES" == "auto" ]]; then
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        REPT_VENV_SYSTEM_SITE_PACKAGES=1
    else
        REPT_VENV_SYSTEM_SITE_PACKAGES=0
    fi
fi

if [[ "$REPT_SKIP_TORCH_INSTALL" == "auto" ]]; then
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        REPT_SKIP_TORCH_INSTALL=1
    else
        REPT_SKIP_TORCH_INSTALL=0
    fi
fi

CACHE_ROOT="${DATA_ROOT}/cache"
RUNS_ROOT="${DATA_ROOT}/runs"
REPT_OUTPUT_DIR="${REPT_OUTPUT_DIR:-${RUNS_ROOT}/grpo_train_lambda}"

echo "=== Lambda Bootstrap ==="
echo "  REPT_ROOT             = $REPT_ROOT"
echo "  REPT_VENV             = $REPT_VENV"
echo "  DATA_ROOT             = $DATA_ROOT"
echo "  Architecture          = $ARCH"
echo "  Requirements          = $REPT_REQUIREMENTS_FILE"
echo "  Python bin            = $REPT_PYTHON_BIN"
echo "  System site-packages  = $REPT_VENV_SYSTEM_SITE_PACKAGES"
echo "  Skip torch install    = $REPT_SKIP_TORCH_INSTALL"
echo "  Recreate venv         = $REPT_RECREATE_VENV"
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

if [[ "$REPT_RECREATE_VENV" == "1" && -d "$REPT_VENV" ]]; then
    echo ">>> Removing existing venv at $REPT_VENV"
    rm -rf "$REPT_VENV"
fi

if [[ -d "$REPT_VENV" ]]; then
    echo ">>> Reusing existing venv at $REPT_VENV"
else
    echo ">>> Creating venv at $REPT_VENV"
    if [[ "$REPT_VENV_SYSTEM_SITE_PACKAGES" == "1" ]]; then
        "$REPT_PYTHON_BIN" -m venv --system-site-packages "$REPT_VENV"
    else
        "$REPT_PYTHON_BIN" -m venv "$REPT_VENV"
    fi
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

# Pre-install CUDA torch before the main requirements install.
# --index-url (not --extra-index-url) replaces PyPI so pip cannot fall back to
# the CPU manylinux wheel cached from a prior run. When the main install then
# resolves torch==2.10.*, it is already satisfied by the CUDA build and vllm's
# dep resolver does not reinstall the CPU variant.
if [[ -n "$PYTORCH_WHEEL_INDEX" && "$REPT_SKIP_TORCH_INSTALL" != "1" ]]; then
    echo ">>> Pre-installing CUDA torch from $PYTORCH_WHEEL_INDEX ..."
    pip install --no-cache-dir "torch==2.10.*" \
        --index-url "$PYTORCH_WHEEL_INDEX"
fi

echo ">>> Installing dependencies..."
REQ_TO_INSTALL="$REPT_REQUIREMENTS_FILE"
if [[ "$REPT_SKIP_TORCH_INSTALL" == "1" ]]; then
    FILTERED_REQ="$TMPDIR/$(basename "$REPT_REQUIREMENTS_FILE").no-torch.txt"
    grep -v '^[[:space:]]*torch[[:space:]=<>!~]' "$REPT_REQUIREMENTS_FILE" > "$FILTERED_REQ"
    REQ_TO_INSTALL="$FILTERED_REQ"
    echo "    Skipping torch from requirements file; relying on system torch."
fi
if [[ -n "$PYTORCH_WHEEL_INDEX" ]]; then
    pip install -r "$REQ_TO_INSTALL" --extra-index-url "$PYTORCH_WHEEL_INDEX"
else
    pip install -r "$REQ_TO_INSTALL"
fi

# echo ">>> TEMORARY TRANSFORMERS PIN..."
# pip install transformers==5.3.0 --force-reinstall --no-deps

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
import jmespath

print(f"  torch        {torch.__version__}  CUDA={torch.cuda.is_available()}")
print(f"  vllm         {vllm.__version__}")
print(f"  trl          {trl.__version__}")
print(f"  transformers {transformers.__version__}")
print("  openenv-core OK")
print(f"  jmespath     {jmespath.__version__}")
PY

echo ""
echo "=== Bootstrap complete ==="
echo "Activate with: source \"$REPT_VENV/bin/activate\""
echo "Suggested run dir: $REPT_OUTPUT_DIR"
