# ReasoningEconomicsPT

Post-training code for LLM/LRMs against the deployed `ReasoningEconomicsEnv` OpenEnv environment.

## Install

Install core training dependencies:

```bash
pip install -r requirements.txt
```

Note: GRPO `environment_factory` tool-calling requires `jmespath` (already included in requirements).

### Production (recommended on hyperscalers)

Install the environment client artifact from your Hugging Face Space:

```bash
pip install "git+https://huggingface.co/spaces/<owner>/<space_repo>"
```

Set base URL to your deployed Space endpoint:

```bash
export ENV_BASE_URL="https://<owner>-<space_repo>.hf.space"
```

### Development fallback (local sibling repo)

```bash
pip install -e "../ReasoningEconomicsEnv"
```

## Run GRPO training (direct)

```bash
python -m training.grpo_train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --env_base_url "$ENV_BASE_URL" \
  --alpha 1.0 \
  --num_train_epochs 1 \
  --num_generations 8 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --vllm_mode colocate \
  --output_dir runs/grpo_train
```

You can also provide a Space URL and let the script derive the base URL:

```bash
python -m training.grpo_train \
  --space_url "https://huggingface.co/spaces/<owner>/<space_repo>"
```

## Lambda Labs On-Demand -- Recommended Workflow

Use this path for single-node GRPO training on a Lambda GPU VM. It replaces CARC-specific Slurm/module flow with direct shell scripts.

### 1. One-time bootstrap

```bash
export REPT_ROOT=/home/ubuntu/ReasoningEconomicsPT
export REPT_VENV=/home/ubuntu/.venvs/rept-lambda
export REPT_FS_NAME=<lambda-filesystem-name>
export REPT_DATA_ROOT=/lambda/nfs/$REPT_FS_NAME/rept
export ENV_BASE_URL="https://<owner>-<space_repo>.hf.space"

cd "$REPT_ROOT"
bash scripts/bootstrap_lambda.sh
```

Notes:

- `bootstrap_lambda.sh` and `run_grpo_lambda.sh` (when `REPT_INSTALL_DEPS_ON_RUN=1`) default to `requirements.lambda.txt`. Override with `REPT_REQUIREMENTS_FILE` if needed (e.g. `requirements.txt` for experiments).
- Recommended for reliable CUDA wheels:

```bash
export PYTORCH_WHEEL_INDEX="https://download.pytorch.org/whl/cu121"
```

### 2. Preflight checks

```bash
bash scripts/preflight_lambda.sh
```

This validates env vars, venv health, imports, GPU visibility, `torch.cuda.is_available()`, endpoint reachability, and output path placement.

### 3. Launch training

```bash
export REPT_OUTPUT_DIR=/lambda/nfs/$REPT_FS_NAME/rept/runs/grpo_train_lambda

# Optional dry run:
bash scripts/run_grpo_lambda.sh --dry-run

# Actual run:
bash scripts/run_grpo_lambda.sh
```

Override defaults via exports before launch:

```bash
export REPT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export REPT_NUM_EPOCHS=1
export REPT_NUM_GENERATIONS=8
export REPT_BATCH_SIZE=2
export REPT_GRAD_ACCUM=8
export REPT_VLLM_MODE=colocate
export REPT_ALPHA=1.0
export REPT_LOG_EVERY=1
```

### 4. Optional local env server on the same Lambda VM

If you want to avoid remote endpoint dependency, run the env server locally and point training at localhost:

```bash
cd /home/ubuntu/ReasoningEconomicsEnv
python -m server.app

# In another shell:
export ENV_BASE_URL="http://127.0.0.1:8000"
```

### 5. Smoke validation checklist

For a quick end-to-end check before long runs:

```bash
export REPT_NUM_EPOCHS=1
export REPT_NUM_GENERATIONS=2
export REPT_BATCH_SIZE=1
export REPT_GRAD_ACCUM=1
bash scripts/run_grpo_lambda.sh
```

Confirm:

- Checkpoints and trainer outputs appear under `$REPT_OUTPUT_DIR`.
- Reward logs appear in `$REPT_OUTPUT_DIR/reward_logs.jsonl`.
- Caches are under `$REPT_DATA_ROOT/cache` (`HF_HOME`, `TRANSFORMERS_CACHE`, `PIP_CACHE_DIR`).

## USC CARC (Discovery) -- Existing Workflow

All outputs (model checkpoints, reward logs, caches) go to `/scratch1/` to avoid home-dir quota exhaustion.

### 1. One-time bootstrap

SSH into a CARC login node and run:

```bash
export REPT_ROOT=/scratch1/$USER/rept/ReasoningEconomicsPT
export REPT_VENV=/scratch1/$USER/rept/rept-venv

cd "$REPT_ROOT"
bash scripts/bootstrap_carc_env.sh
```

This loads modules (`gcc/13.3.0`, `python/3.11.9`, `cuda/12.6.3`), creates a venv, installs pinned CARC dependencies, and runs a smoke import test.

### 2. Pre-submission check

```bash
export REPT_ROOT=/scratch1/$USER/rept/ReasoningEconomicsPT
export REPT_VENV=/scratch1/$USER/rept/rept-venv
export ENV_BASE_URL="https://<owner>-<space_repo>.hf.space"
export REPT_OUTPUT_DIR=/scratch1/$USER/rept/runs/grpo_train_carc

bash scripts/preflight_carc.sh
```

### 3. Submit training job

```bash
# Dry run (print config, no submission):
bash scripts/submit_grpo_carc.sh --dry-run

# Actual submission:
bash scripts/submit_grpo_carc.sh
```

## Dependency files

- `requirements.txt` -- default training dependencies (generic; used if `REPT_REQUIREMENTS_FILE` unset).
- `requirements.lambda.txt` -- Lambda-recommended pins (matches v2.1 / cross-chat stack: torch 2.8.*, vLLM 0.10.2, trl 1.0.0, transformers>=5.2.0).
- `requirements.carc-cu121.txt` -- CARC-specific lock file for Discovery CUDA stack.

Use `requirements.lambda.txt` on Lambda instead of loosening `vllm` in `requirements.txt` (avoids pip resolver conflicts with `torch` / `pydantic`).

## Notes

- Training uses OpenEnv `reset/step` calls over HTTP instead of importing `ReasonBudgetEnvironment` in-process.
- Hosted Spaces may have limited concurrency; for sustained RL training, use your own duplicated Space or a local deployment.
