#!/bin/bash
#SBATCH --job-name=rept-grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rept-grpo-%j.out

set -euo pipefail

# Required exports (can also be set in shell before sbatch):
#   REPT_ROOT                absolute path to ReasoningEconomicsPT on CARC
#   REPT_VENV                absolute path to Python venv with dependencies installed
#   ENV_BASE_URL             OpenEnv space URL (https://owner-space.hf.space)
#
# Optional exports:
#   REPT_MODEL               default: Qwen/Qwen2.5-0.5B-Instruct
#   REPT_OUTPUT_DIR          default: runs/grpo_train_carc
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
REPT_OUTPUT_DIR="${REPT_OUTPUT_DIR:-runs/grpo_train_carc}"
REPT_NUM_EPOCHS="${REPT_NUM_EPOCHS:-1}"
REPT_NUM_GENERATIONS="${REPT_NUM_GENERATIONS:-8}"
REPT_BATCH_SIZE="${REPT_BATCH_SIZE:-2}"
REPT_GRAD_ACCUM="${REPT_GRAD_ACCUM:-8}"
REPT_VLLM_MODE="${REPT_VLLM_MODE:-colocate}"
REPT_DEFAULT_BUDGET_MODE="${REPT_DEFAULT_BUDGET_MODE:-hard}"
REPT_ALPHA="${REPT_ALPHA:-1.0}"
REPT_BETA="${REPT_BETA:-0.0}"
REPT_LOG_EVERY="${REPT_LOG_EVERY:-1}"

cd "$REPT_ROOT"
mkdir -p logs

# If your CARC setup requires modules, load them before activating venv.
# module purge
# module load python/3.11
# module load cuda/12.1

source "$REPT_VENV/bin/activate"

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
