#!/bin/bash
#SBATCH --job-name=rept-grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rept-grpo-%j.out

set -euo pipefail

# Required exports (set in shell before sbatch, or in submit_grpo_carc.sh):
#   REPT_ROOT                absolute path to ReasoningEconomicsPT on CARC
#   REPT_VENV                absolute path to Python venv with dependencies installed
#   ENV_BASE_URL             OpenEnv space URL (https://owner-space.hf.space)
#
# Optional exports:
#   REPT_MODEL               default: Qwen/Qwen2.5-0.5B-Instruct
#   REPT_OUTPUT_DIR          default: /scratch1/$USER/rept/runs/grpo_train_carc
#   REPT_NUM_EPOCHS          default: 1
#   REPT_NUM_GENERATIONS     default: 8
#   REPT_BATCH_SIZE          default: 2
#   REPT_GRAD_ACCUM          default: 8
#   REPT_VLLM_MODE           default: colocate
#   REPT_DEFAULT_BUDGET_MODE default: hard
#   REPT_ALPHA               default: 1.0
#   REPT_BETA                default: 0.0
#   REPT_LOG_EVERY           default: 1

: "${REPT_ROOT:?REPT_ROOT is required}"
: "${REPT_VENV:?REPT_VENV is required}"
: "${ENV_BASE_URL:?ENV_BASE_URL is required}"

REPT_MODEL="${REPT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
REPT_OUTPUT_DIR="${REPT_OUTPUT_DIR:-/scratch1/$USER/rept/runs/grpo_train_carc}"
REPT_NUM_EPOCHS="${REPT_NUM_EPOCHS:-1}"
REPT_NUM_GENERATIONS="${REPT_NUM_GENERATIONS:-8}"
REPT_BATCH_SIZE="${REPT_BATCH_SIZE:-2}"
REPT_GRAD_ACCUM="${REPT_GRAD_ACCUM:-8}"
REPT_VLLM_MODE="${REPT_VLLM_MODE:-colocate}"
REPT_DEFAULT_BUDGET_MODE="${REPT_DEFAULT_BUDGET_MODE:-hard}"
REPT_ALPHA="${REPT_ALPHA:-1.0}"
REPT_BETA="${REPT_BETA:-0.0}"
REPT_LOG_EVERY="${REPT_LOG_EVERY:-1}"

SCRATCH_CACHE="/scratch1/$USER/cache"

# ------------------------------------------------------------------ modules
# gcc/12.3.0: loadable on Discovery2 without legacy/CentOS7 (gcc/11.3.0 requires that stack).
echo ">>> Loading modules..."
module purge
module load gcc/13.3.0
module load python/3.11.9
module load cuda/12.6.3
echo "    Loaded: gcc/13.3.0, python/3.11.9, cuda/12.6.3"
# ------------------------------------------------------------------ venv
cd "$REPT_ROOT"
mkdir -p logs
source "$REPT_VENV/bin/activate"

# ------------------------------------------------------------------ runtime caches on scratch
mkdir -p "$SCRATCH_CACHE/huggingface" "$SCRATCH_CACHE/tmp" "$REPT_OUTPUT_DIR"
export HF_HOME="$SCRATCH_CACHE/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH_CACHE/huggingface/transformers"
export TMPDIR="$SCRATCH_CACHE/tmp"

# ------------------------------------------------------------------ diagnostics
echo "=== REPT GRPO Training ==="
echo "  Job ID:      ${SLURM_JOB_ID:-local}"
echo "  Node:        $(hostname)"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "  Python:      $(python --version)"
echo "  Torch:       $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA avail:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Model:       $REPT_MODEL"
echo "  Output dir:  $REPT_OUTPUT_DIR"
echo "  Env URL:     $ENV_BASE_URL"
echo "  Budget mode: $REPT_DEFAULT_BUDGET_MODE"
echo "=========================="

# ------------------------------------------------------------------ training
python -m training.grpo_train \
  --model "$REPT_MODEL" \
  --env_base_url "$ENV_BASE_URL" \
  --default_budget_mode "$REPT_DEFAULT_BUDGET_MODE" \
  --alpha "$REPT_ALPHA" \
  --beta "$REPT_BETA" \
  --log_every_n_steps "$REPT_LOG_EVERY" \
  --num_train_epochs "$REPT_NUM_EPOCHS" \
  --num_generations "$REPT_NUM_GENERATIONS" \
  --per_device_train_batch_size "$REPT_BATCH_SIZE" \
  --gradient_accumulation_steps "$REPT_GRAD_ACCUM" \
  --vllm_mode "$REPT_VLLM_MODE" \
  --output_dir "$REPT_OUTPUT_DIR"

echo "=== Training complete. Artifacts at: $REPT_OUTPUT_DIR ==="
