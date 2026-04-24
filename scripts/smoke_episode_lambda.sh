#!/bin/bash
# smoke_episode_lambda.sh -- Minimal episode-mode GRPO smoke for a Lambda GPU VM.

set -euo pipefail

: "${REPT_ROOT:?REPT_ROOT is required}"
: "${REPT_VENV:?REPT_VENV is required}"
: "${ENV_BASE_URL:?ENV_BASE_URL is required}"

if [[ -n "${REPT_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$REPT_DATA_ROOT"
elif [[ -n "${REPT_FS_NAME:-}" ]]; then
    DATA_ROOT="/lambda/nfs/${REPT_FS_NAME}/rept"
else
    DATA_ROOT="/lambda/nfs/rept"
fi

REPT_MODEL="${REPT_MODEL:-Qwen/Qwen3-1.7B}"
REPT_OUTPUT_DIR="${REPT_OUTPUT_DIR:-${DATA_ROOT}/runs/grpo_episode_smoke_lambda}"
REPT_REWARD_LOG_PATH="${REPT_REWARD_LOG_PATH:-$REPT_OUTPUT_DIR/reward_log.jsonl}"
REPT_NUM_EPOCHS="${REPT_NUM_EPOCHS:-1}"
REPT_NUM_GENERATIONS="${REPT_NUM_GENERATIONS:-4}"
REPT_BATCH_SIZE="${REPT_BATCH_SIZE:-4}"
REPT_GRAD_ACCUM="${REPT_GRAD_ACCUM:-4}"
REPT_VLLM_MODE="${REPT_VLLM_MODE:-colocate}"
REPT_VLLM_GPU_UTIL="${REPT_VLLM_GPU_UTIL:-${REPT_VLLM_GPU_MEMORY_UTILIZATION:-0.25}}"
REPT_MAX_COMPLETION_LENGTH="${REPT_MAX_COMPLETION_LENGTH:-2048}"
REPT_MAX_TOKENS_PER_STEP="${REPT_MAX_TOKENS_PER_STEP:-$REPT_MAX_COMPLETION_LENGTH}"
REPT_DEFAULT_BUDGET_MODE="${REPT_DEFAULT_BUDGET_MODE:-soft}"
REPT_MAX_EPISODE_TURNS="${REPT_MAX_EPISODE_TURNS:-20}"

if (( REPT_BATCH_SIZE % REPT_NUM_GENERATIONS != 0 )); then
    echo "[ERROR] REPT_BATCH_SIZE ($REPT_BATCH_SIZE) must be divisible by REPT_NUM_GENERATIONS ($REPT_NUM_GENERATIONS)"
    exit 1
fi

mkdir -p "$REPT_OUTPUT_DIR"

cd "$REPT_ROOT"
# shellcheck source=/dev/null
source "$REPT_VENV/bin/activate"

python -m training.grpo_train \
  --model "$REPT_MODEL" \
  --env_base_url "$ENV_BASE_URL" \
  --reward_log_path "$REPT_REWARD_LOG_PATH" \
  --num_train_epochs "$REPT_NUM_EPOCHS" \
  --num_generations "$REPT_NUM_GENERATIONS" \
  --per_device_train_batch_size "$REPT_BATCH_SIZE" \
  --gradient_accumulation_steps "$REPT_GRAD_ACCUM" \
  --vllm_mode "$REPT_VLLM_MODE" \
  --vllm_gpu_memory_utilization "$REPT_VLLM_GPU_UTIL" \
  --max_completion_length "$REPT_MAX_COMPLETION_LENGTH" \
  --max_tokens_per_step "$REPT_MAX_TOKENS_PER_STEP" \
  --default_budget_mode "$REPT_DEFAULT_BUDGET_MODE" \
  --max_episode_turns "$REPT_MAX_EPISODE_TURNS" \
  --output_dir "$REPT_OUTPUT_DIR"

python scripts/summarize_episode_run.py "$REPT_REWARD_LOG_PATH"
