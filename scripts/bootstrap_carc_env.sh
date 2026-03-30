#!/bin/bash
# bootstrap_carc_env.sh -- One-time setup for GRPO training on USC CARC Discovery.
#
# Usage (on a CARC login node):
#   export REPT_ROOT=/scratch1/yashaswi/rept/ReasoningEconomicsPT
#   export REPT_VENV=/scratch1/yashaswi/rept/rept-venv
#   bash scripts/bootstrap_carc_env.sh

set -euo pipefail

: "${REPT_ROOT:?Set REPT_ROOT to the absolute path of ReasoningEconomicsPT on CARC}"
: "${REPT_VENV:?Set REPT_VENV to the absolute path for the Python venv}"

SCRATCH_CACHE="/scratch1/$(whoami)/cache"

echo "=== CARC Bootstrap ==="
echo "  REPT_ROOT  = $REPT_ROOT"
echo "  REPT_VENV  = $REPT_VENV"
echo ""

# ------------------------------------------------------------------ modules
# gcc/12.3.0: loadable on Discovery2 without legacy/CentOS7 (gcc/11.3.0 requires that stack).
echo ">>> Loading modules..."
module purge
module load gcc/13.3.0
module load python/3.11.9
module load cuda/12.6.3
echo "    Loaded: gcc/13.3.0, python/3.11.9, cuda/12.6.3"

# ------------------------------------------------------------------ venv
if [[ -d "$REPT_VENV" ]]; then
    echo ">>> Venv already exists at $REPT_VENV, reusing."
else
    echo ">>> Creating venv at $REPT_VENV ..."
    python3 -m venv "$REPT_VENV"
fi
source "$REPT_VENV/bin/activate"
echo "    Python: $(python --version)  ($(which python))"

# ------------------------------------------------------------------ caches on scratch
echo ">>> Setting cache dirs to scratch..."
mkdir -p "$SCRATCH_CACHE/pip" "$SCRATCH_CACHE/huggingface" "$SCRATCH_CACHE/tmp"
export PIP_CACHE_DIR="$SCRATCH_CACHE/pip"
export HF_HOME="$SCRATCH_CACHE/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH_CACHE/huggingface/transformers"
export TMPDIR="$SCRATCH_CACHE/tmp"
echo "    PIP_CACHE_DIR      = $PIP_CACHE_DIR"
echo "    HF_HOME            = $HF_HOME"
echo "    TMPDIR             = $TMPDIR"

# ------------------------------------------------------------------ pip upgrade
echo ">>> Upgrading pip..."
pip install --quiet --upgrade pip

# ------------------------------------------------------------------ install pinned deps
echo ">>> Installing pinned requirements (CUDA 12.1)..."
pip install -r "$REPT_ROOT/requirements.carc-cu121.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------------ smoke test
echo ""
echo ">>> Smoke-testing critical imports..."
python -c "
import torch
print(f'  torch        {torch.__version__}  CUDA={torch.cuda.is_available()}')
import vllm
print(f'  vllm         {vllm.__version__}')
import trl
print(f'  trl          {trl.__version__}')
import transformers
print(f'  transformers {transformers.__version__}')
import openenv
print(f'  openenv-core OK')
"

echo ""
echo "=== Bootstrap complete ==="
echo "Activate with:  source $REPT_VENV/bin/activate"
