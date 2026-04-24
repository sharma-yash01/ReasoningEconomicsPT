#!/bin/bash
# Episode mode correctness smoke on CPU (MacBook 24GB).
# Forces everything to CPU via ACCELERATE_USE_CPU=1 + use_cpu=True in GRPOConfig.
# Uses Qwen3-0.6B float32, 1 prompt, 1 step, 2 generations, 4 questions.
#
# Env must be started with:
#   REASON_BUDGET_NUM_QUESTIONS=4 REASON_BUDGET_MAX_TOKENS_PER_STEP=40 \
#   uvicorn server.app:app --host 127.0.0.1 --port 8010
#
# Budget math: suffix grows with history size (0→1→2 history entries per step):
#   4 answers × 40 tokens  =  160 tokens  (model generation)
#   suffixes: S1≈149, S2≈147, S3≈310 (history grows) ≈ 606 tokens total
#   Grand total ≈ 766 tokens → 1024 has 258 tokens of headroom.
#   768 was marginal: episode_idx=1 hit 302 remaining < S3≈310, terminated at step=2.
#
# Success: both episodes reach step=3, final done=True questions_remaining=0.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONUNBUFFERED=1 REPT_DEBUG_EPISODE=1 ACCELERATE_USE_CPU=1 python -m training.grpo_train \
    --model Qwen/Qwen3-0.6B \
    --torch_dtype float32 \
    --openenv_mode episode \
    --env_base_url http://127.0.0.1:8010 \
    --n_prompts 1 \
    --max_steps 1 \
    --num_generations 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 1024 \
    --reward_log_path runs/grpo_openenv_episode_cpu_smoke4q/reward_log.jsonl \
    --output_dir runs/grpo_openenv_episode_cpu_smoke4q
