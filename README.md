# ReasoningEconomicsPT

Post-training code for LLM/LRMs against the deployed `ReasoningEconomicsEnv` OpenEnv environment.

## Install

Install core training dependencies:

```bash
pip install -r requirements.txt
```

### Production (recommended on CARC/hyperscalers)

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

## Run GRPO training

```bash
python -m training.grpo_train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --env_base_url "$ENV_BASE_URL" \
  --alpha 1.0 \
  --beta 0.0 \
  --default_budget_mode hard \
  --num_train_epochs 1 \
  --output_dir runs/grpo_train
```

You can also provide a Space URL and let the script derive the base URL:

```bash
python -m training.grpo_train \
  --space_url "https://huggingface.co/spaces/<owner>/<space_repo>"
```

## Metadata contract for token-cap mode

`grpo_train.py` reads `observation.metadata["budget_mode"]` each step:

- `hard`: cap generation by remaining budget (plus minimum floor).
- `soft`: use `max_tokens_per_step` without remaining-budget cap.

If metadata is missing, training falls back to `--default_budget_mode` unless `--strict_budget_mode_metadata` is set.

## Reward topology used in PT (current)

- PT trains on the **incoming total reward signal** from the environment (`StepResult.reward`).
- In prod env behavior, terminal-step reward already includes episode bonus.
- `alpha` scales total signal; `beta` is reserved for future decomposed reward wiring.

## Reward logging

By default, reward logs are emitted:

- Console logs every `--log_every_n_steps`.
- JSONL file at `runs/<output_dir>/reward_logs.jsonl` (or `--reward_log_path`).

Disable logs with:

```bash
python -m training.grpo_train --no_log_rewards ...
```

## USC CARC (Discovery) -- Full Workflow

All outputs (model checkpoints, reward logs, caches) go to `/scratch1/` to avoid home-dir quota exhaustion.

### 1. One-time bootstrap

SSH into a CARC login node and run:

```bash
# Clone or copy ReasoningEconomicsPT to scratch
export REPT_ROOT=/scratch1/$USER/rept/ReasoningEconomicsPT
export REPT_VENV=/scratch1/$USER/rept/rept-venv

# (Optional) install env client from HF Space artifact
export ENV_CLIENT_INSTALL="git+https://huggingface.co/spaces/<owner>/<space_repo>"
# Or for local dev:
# export ENV_CLIENT_INSTALL="-e /scratch1/$USER/rept/ReasoningEconomicsEnv"

cd "$REPT_ROOT"
bash scripts/bootstrap_carc_env.sh
```

This loads modules (`gcc/12.3.0`, `python/3.11.6`, `cuda/12.1.1`), creates a venv, installs all pinned dependencies from `requirements.carc-cu121.txt`, and runs a smoke import test. (`gcc/11.3.0` on Discovery2 requires `legacy/CentOS7` first; `gcc/12.3.0` loads directly.)

### 2. Pre-submission check

```bash
export REPT_ROOT=/scratch1/$USER/rept/ReasoningEconomicsPT
export REPT_VENV=/scratch1/$USER/rept/rept-venv
export ENV_BASE_URL="https://<owner>-<space_repo>.hf.space"
export REPT_OUTPUT_DIR=/scratch1/$USER/rept/runs/grpo_train_carc

bash scripts/preflight_carc.sh
```

Fails fast on missing vars, broken venv, unreachable endpoint, or non-scratch output path.

### 3. Submit training job

```bash
# Dry run (print config, no submission):
bash scripts/submit_grpo_carc.sh --dry-run

# Actual submission:
bash scripts/submit_grpo_carc.sh
```

Override defaults via exports before submitting:

```bash
export REPT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export REPT_NUM_EPOCHS=1
export REPT_NUM_GENERATIONS=8
export REPT_BATCH_SIZE=2
export REPT_GRAD_ACCUM=8
export REPT_VLLM_MODE=colocate
export REPT_DEFAULT_BUDGET_MODE=hard
export REPT_ALPHA=1.0
export REPT_BETA=0.0
```

### 4. Monitor

```bash
# Check job status
squeue -u "$USER"
scontrol show job <JOBID>
sacct -u "$USER" -S today --format=JobID,JobName%25,Partition,State,Elapsed,ExitCode

# Tail training log
tail -f logs/rept-grpo-<JOBID>.out
```

### 5. Artifact locations

After training completes, outputs are at:

| Artifact | Path |
|---|---|
| Model + checkpoints | `$REPT_OUTPUT_DIR/` (e.g. `/scratch1/$USER/rept/runs/grpo_train_carc/`) |
| Reward JSONL log | `$REPT_OUTPUT_DIR/reward_logs.jsonl` |
| Slurm stdout log | `$REPT_ROOT/logs/rept-grpo-<JOBID>.out` |
| HF cache | `/scratch1/$USER/cache/huggingface/` |

### CARC module versions

The scripts load these modules explicitly:

- `gcc/12.3.0`
- `python/3.11.6`
- `cuda/12.1.1`

### Dependency pinning

- `requirements.txt` -- platform-agnostic pinned deps (exact `==` versions).
- `requirements.carc-cu121.txt` -- CARC-specific lock file; install with `--extra-index-url https://download.pytorch.org/whl/cu121`.

## Notes

- Training uses remote OpenEnv `reset/step` calls instead of importing `ReasonBudgetEnvironment` in-process.
- Hosted Spaces may have limited concurrency; for sustained RL training, use your own duplicated Space or a local Docker deployment as recommended in OpenEnv docs.
