# ReasoningEconomicsPT

Post-training code for LLM/LRMs against the deployed `ReasoningEconomicsEnv` OpenEnv environment.

## Submitted branch

ReasoningEconomicsPT branch: `deepseed-harshawn`
https://github.com/sharma-yash01/ReasoningEconomicsPT/tree/deepseed-harshawn

This branch adds the training code and Lambda scripts used for the Qwen3-14B
GRPO run. The trainer now runs full OpenEnv episodes with explicit `reset` /
`step` calls, handles Qwen3 answer parsing, writes reward/debug JSONL logs,
supports server-mode vLLM, and includes DeepSpeed configs for A100/H100 runs.

Key scripts:

- `scripts/bootstrap_lambda.sh`
- `scripts/preflight_lambda.sh`
- `scripts/run_grpo_lambda.sh`
- `scripts/smoke_episode_cpu.sh`
- `scripts/smoke_episode_lambda.sh`

The main 14B run used Lambda Labs 8x A100. Six GPUs were used for GRPO training
and two GPUs were used for `trl vllm-serve`. Small smoke tests can run on
smaller hardware, but positive 14B training results need multiple 80 GB GPUs,
at least A100 class.

Main run:

```bash
cd ReasoningEconomicsPT

export REPT_ROOT="$PWD"
export REPT_VENV="/home/ubuntu/rept-venv"
export REPT_DATA_ROOT="/lambda/nfs/csci-544/rept"
export ENV_BASE_URL="http://127.0.0.1:8000"

export REPT_MODEL=Qwen/Qwen3-14B
export REPT_NUM_GPUS=8
export REPT_VLLM_MODE=server
export REPT_VLLM_TP=2
export REPT_SHARDING_BACKEND=deepspeed
export REPT_DEEPSPEED_CONFIG="$REPT_ROOT/configs/deepspeed/zero3_8x_a100_40gb_offload_optimizer.json"

bash scripts/bootstrap_lambda.sh
bash scripts/preflight_lambda.sh
bash scripts/run_grpo_lambda.sh --dry-run
bash scripts/run_grpo_lambda.sh
```

Results are generated from:

- `reward_log.jsonl`
- `rollout_debug.jsonl`
- `vllm_serve.log`
- trainer metrics/checkpoint files in `REPT_OUTPUT_DIR`

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

## Run GRPO training (`training.grpo_train`)

`python -m training.grpo_train` is the entry point. **`--vllm_mode`** may be **`colocate`** (trainer + vLLM on the same GPU process layout) or **`server`** (vLLM in a separate **`trl vllm-serve`** process; you must start that server yourself unless you use **`run_grpo_lambda.sh`**, which starts it in server mode).

Minimal example (single GPU, colocate):

```bash
python -m training.grpo_train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --env_base_url "$ENV_BASE_URL" \
  --alpha 1.0 \
  --num_train_epochs 1 \
  --num_generations 8 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --vllm_mode colocate \
  --output_dir runs/grpo_train
```

You can also provide a Space URL and let the script derive the base URL:

```bash
python -m training.grpo_train \
  --space_url "https://huggingface.co/spaces/<owner>/<space_repo>"
```

### Full CLI reference (`grpo_train.py`)

| Argument | Default | Notes |
|----------|---------|--------|
| `--model` | `Qwen/Qwen3-0.6B` | Hub id or local checkpoint path |
| `--num_train_epochs` | `1` | Training epochs |
| `--num_generations` | `8` | GRPO samples per prompt; must divide `per_device_train_batch_size` |
| `--max_completion_length` | `4096` | Max new tokens for generation |
| `--max_tokens_per_step` | `2048` | Step-level cap (rollout) |
| `--min_tokens_per_step` | `10` | |
| `--default_budget_mode` | `hard` | |
| `--strict_budget_mode_metadata` | off (flag) | Require budget metadata from obs |
| `--alpha` | `1.0` | Scales per-step env reward |
| `--no_log_rewards` | off | Disable reward JSONL |
| `--log_every_n_steps` | `1` | |
| `--reward_log_path` | `""` | Default: `{output_dir}/reward_logs.jsonl` |
| `--per_device_train_batch_size` | `8` | Must be divisible by `num_generations` |
| `--gradient_accumulation_steps` | `4` | |
| `--vllm_mode` | `colocate` | **`colocate`** \| **`server`** |
| `--vllm_tensor_parallel_size` | `1` | vLLM TP (server mode) |
| `--vllm_gpu_memory_utilization` | `0.9` | vLLM memory fraction (server) |
| `--vllm_server_host` | `127.0.0.1` | TRL vLLM HTTP host (server mode) |
| `--vllm_server_port` | `8001` | Avoid **8000** if OpenEnv uses it |
| `--vllm_group_port` | `51216` | TRL weight-sync TCP port |
| `--gradient_checkpointing` | off | Flag; reduces training VRAM |
| `--no_bf16` | off | bf16 on by default |
| `--output_dir` | `runs/grpo_train` | |
| `--learning_rate` | `5e-7` | |
| `--env_base_url` | `None` | OpenEnv base URL |
| `--space_url` | `None` | HF Space → derived base URL |
| `--max_episode_turns` | `256` | |
| `--env_tokenizer_name` | `None` | HF tokenizer id for remote env (use when `--model` is a local path) |
| `--env_total_budget` | `None` | Override env token budget on reset |
| `--model_profiles_path` | package default | JSON profiles (`enable_thinking`, parsers) |
| `--reasoning_mode` | `auto` | **`auto`** \| **`on`** \| **`off`** — merges `enable_thinking` into chat template kwargs |

Startup guard: **`per_device_train_batch_size % num_generations == 0`** (GRPO requirement).

---

## Lambda Labs on-demand — recommended workflow

Use **`scripts/bootstrap_lambda.sh`**, **`preflight_lambda.sh`**, and **`run_grpo_lambda.sh`** for single-node GRPO on a Lambda GPU VM.

**Design notes and debugging history** (WebSocket sessions, vLLM 404, multi-GPU, NFS cache, tokenizer alignment): see repo-root **`cross-chat-completion.md`** and **`impl-context/`** (e.g. **`dist-train-issue-vllm-call-issue.md`**, **`v2.2-impl.md`** erratum for TRL 1.0.0 + `trl vllm-serve`).

### vLLM modes (`REPT_VLLM_MODE` / `--vllm_mode`)

| Mode | Behavior |
|------|----------|
| **`auto`** (default) | **≥2** visible GPUs → **`server`**; **1** GPU → **`colocate`**. |
| **`server`** | **`run_grpo_lambda.sh`** starts **`trl vllm-serve`** on the **last** **`REPT_VLLM_TP`** GPU(s), waits for **`GET /get_world_size/`** on **`REPT_VLLM_PORT`**, then runs **`accelerate launch`** on the remaining GPUs with **`CUDA_VISIBLE_DEVICES`** set to training GPUs only. Requires **TRL-compatible** server (not plain `vllm serve` OpenAI-only HTTP). |
| **`colocate`** | Single-process **`python -m training.grpo_train`**; vLLM colocated with training (highest VRAM pressure). |

**Port split (do not collide OpenEnv with TRL):**

| Service | Default port |
|---------|----------------|
| OpenEnv (`ENV_BASE_URL`) | often **8000** |
| **`trl vllm-serve`** | **`REPT_VLLM_PORT`** — default **8001** |
| TRL weight-sync | **`REPT_VLLM_GROUP_PORT`** — default **51216** |
| Accelerate rendezvous | **`REPT_ACCELERATE_MAIN_PORT`** — default **29500** |

If TRL hits **404** on **`/get_world_size/`**, the client is usually pointed at the **wrong** HTTP server (e.g. OpenEnv on 8000) or **`trl vllm-serve`** is not running. See **`cross-chat-completion.md`** § *vLLM HTTP 404*.

### V100 / 16 GB GPUs and `trl vllm-serve`

- **Compute capability:** V100 is **sm_70**. vLLM may log **FlashAttention 2 is only supported on devices with compute capability ≥ 8** — expect fallback attention paths on V100 (not necessarily fatal).
- **KV cache / `max_model_len`:** Models such as Qwen3 advertise a large **max sequence length** (e.g. 40960). On **16 GB**, vLLM can fail during engine init with **not enough KV cache memory** (needs slightly more than available after weights). Set **`export REPT_VLLM_MAX_MODEL_LEN=8192`** (or another cap **≥ prompt + `REPT_MAX_COMPLETION_LENGTH`**) so **`run_grpo_lambda.sh`** passes **`--max_model_len`** to **`trl vllm-serve`**. Alternatives: raise **`REPT_VLLM_GPU_UTIL`** slightly (e.g. **0.92–0.95**) or use a **smaller model**. **`REPT_MAX_COMPLETION_LENGTH`** does not replace the engine’s configured max context by itself.
- **Multi-GPU training:** Script defaults **`NCCL_P2P_DISABLE=1`** (PCIe / V100-safe). On **NVLink A100**, you may set **`REPT_NCCL_P2P_DISABLE=0`**.
- **Stale processes:** If training OOMs or hangs after a crash, **`nvidia-smi`** and kill stray **`grpo_train`** / **`trl vllm-serve`** PIDs before retrying.

### Environment variables — `run_grpo_lambda.sh`

**Required**

| Variable | Meaning |
|----------|---------|
| `REPT_ROOT` | Absolute path to this repo (`ReasoningEconomicsPT`) |
| `REPT_VENV` | Absolute path to Python venv |
| `ENV_BASE_URL` | OpenEnv base URL (e.g. Space `https://…hf.space` or `http://127.0.0.1:8000`) |

**Optional — paths & install**

| Variable | Default |
|----------|---------|
| `REPT_FS_NAME` | (used if `REPT_DATA_ROOT` unset) → `/lambda/nfs/<name>/rept` |
| `REPT_DATA_ROOT` | `/lambda/nfs/<fs>/rept` or `/lambda/nfs/rept` |
| `REPT_REQUIREMENTS_FILE` | `$REPT_ROOT/requirements.lambda.txt` |
| `PYTORCH_WHEEL_INDEX` | unset (e.g. `https://download.pytorch.org/whl/cu121` for CUDA wheels) |
| `REPT_INSTALL_DEPS_ON_RUN` | `0` — set `1` to `pip install` before each run |

**Optional — training & model**

| Variable | Default | Notes |
|----------|---------|--------|
| `REPT_MODEL` | `Qwen/Qwen3-8B` | Hub id prefetched to `$DATA_ROOT/models/...`; local path skips download |
| `REPT_OUTPUT_DIR` | `$DATA_ROOT/runs/grpo_train_lambda` | |
| `REPT_NUM_EPOCHS` | `1` | |
| `REPT_NUM_GENERATIONS` | `8` | Must divide `REPT_BATCH_SIZE` |
| `REPT_BATCH_SIZE` | `8` | Maps to `--per_device_train_batch_size` |
| `REPT_GRAD_ACCUM` | `8` | Overridden by auto-tune unless override set |
| `REPT_GRAD_ACCUM_OVERRIDE` | unset | Set to **any non-empty value** to **disable** grad-accum auto-tune and use `REPT_GRAD_ACCUM` as-is |
| `REPT_ALPHA` | `1.0` | |
| `REPT_LOG_EVERY` | `1` | `--log_every_n_steps` |

**Optional — GPUs & vLLM**

| Variable | Default | Notes |
|----------|---------|--------|
| `REPT_NUM_GPUS` | `auto` | `auto` = count from `nvidia-smi` |
| `REPT_VLLM_MODE` | `auto` | **`auto`** \| **`server`** \| **`colocate`** |
| `REPT_VLLM_TP` | `1` | Tensor-parallel GPUs for vLLM in server mode |
| `REPT_VLLM_PORT` | `8001` | TRL HTTP |
| `REPT_VLLM_GROUP_PORT` | `51216` | TRL TCP |
| `REPT_VLLM_SERVER_HOST` | `127.0.0.1` | |
| `REPT_VLLM_GPU_UTIL` | `0.9` | `--vllm_gpu_memory_utilization` |
| `REPT_VLLM_MAX_MODEL_LEN` | unset | **Server only:** passed as **`trl vllm-serve --max_model_len`**; caps KV/context when the model config default (e.g. 40k) is too large for the GPU |
| `REPT_MAX_COMPLETION_LENGTH` | `4096` | |
| `REPT_GRADIENT_CHECKPOINTING` | `1` | Set `0` to omit `--gradient_checkpointing` |
| `REPT_NO_BF16` | `0` | Set `1` for `--no_bf16` |
| `REPT_ACCELERATE_MAIN_PORT` | `29500` | If port busy, change or use `0` |
| `REPT_NCCL_P2P_DISABLE` | inherits / `1` | Multi-GPU only; see V100 note above |
| `NCCL_TIMEOUT` | `1800` | Set when `TRAIN_PROCS > 1` |

**Optional — Hub**

| Variable | Notes |
|----------|--------|
| `HF_TOKEN` | Higher Hub rate limits for downloads |

The launcher passes a **fixed subset** of `grpo_train` flags (see **`scripts/run_grpo_lambda.sh`** `COMMON_ARGS`). Arguments **not** wired there (e.g. `--reasoning_mode`, `--env_total_budget`, `--model_profiles_path`) require **`python -m training.grpo_train`** directly or extending the script.

### Canonical Lambda execution process

From **`impl-context/v2-impl-prod.md`** (adjust paths and `REPT_FS_NAME` to your VM):

```bash
export REPT_ROOT="$PWD"
# export REPT_VENV="/home/ubuntu/openenv-train/rept-venv"
export REPT_VENV="/home/ubuntu/OpenEnv-Training/rept-venv"
export REPT_REQUIREMENTS_FILE="$REPT_ROOT/requirements.lambda.txt"
export PYTORCH_WHEEL_INDEX="https://download.pytorch.org/whl/cu121"
# export REPT_FS_NAME="openenv-train"
export REPT_FS_NAME="OpenEnv-Training"
export REPT_DATA_ROOT="/lambda/nfs/$REPT_FS_NAME/rept"
export REPT_MODEL="Qwen/Qwen3-4B"
# if mem limited
export REPT_GRAD_ACCUM=4
export REPT_MODEL="Qwen/Qwen3-1.7B"
export REPT_NUM_GENERATIONS=4
export ENV_BASE_URL="https://yashu2000-reasoning-economic-env.hf.space"

# local ENV
export ENV_BASE_URL="http://127.0.0.1:8000"

# distributed training
unset CUDA_VISIBLE_DEVICES

export REPT_NUM_GPUS=auto
export REPT_VLLM_MODE=auto
export REPT_VLLM_TP=1
# Server mode (auto with ≥2 GPUs): run_grpo_lambda.sh starts trl vllm-serve on REPT_VLLM_PORT (default 8001),
# not OpenEnv’s 8000. See cross-chat-completion.md § vLLM HTTP 404 and impl-context/v2.2-impl.md erratum.

bash scripts/bootstrap_lambda.sh
bash scripts/preflight_lambda.sh
bash scripts/run_grpo_lambda.sh --dry-run
bash scripts/run_grpo_lambda.sh
nohup bash scripts/run_grpo_lambda.sh > train.log 2>&1 &
watch -n 1 nvidia-smi
```

Pick **one** `ENV_BASE_URL`, **one** `REPT_MODEL` line (remove duplicates you do not want), and align mem-limited exports with your GPU.

### Out of memory (OOM) on Lambda

GRPO with **vLLM colocated** loads the policy for training and inference on the same visible GPU(s); memory scales with **parallel rollouts**. If you see CUDA OOM, reduce pressure in roughly this order:

1. **`REPT_NUM_GENERATIONS`** — fewer completions per step (try `4`, then `2` for smoke; GRPO still needs **≥2** generations and batch divisibility).
2. **`REPT_MODEL`** — smaller Hub id (e.g. **1.7B** / **0.5B** class).
3. **`REPT_BATCH_SIZE`** — smaller micro-batch; tune **`REPT_GRAD_ACCUM`** / **`REPT_GRAD_ACCUM_OVERRIDE`** for effective batch.
4. **`REPT_MAX_COMPLETION_LENGTH`** — shorter rollouts.
5. **`REPT_VLLM_MODE=server`** with **≥2 GPUs** — dedicated vLLM GPU(s) via `run_grpo_lambda.sh`.
6. **Hardware** — more VRAM per GPU.

Re-export variables, then rerun preflight and training.

### 1. One-time bootstrap

```bash
cd "$REPT_ROOT"
bash scripts/bootstrap_lambda.sh
```

Notes:

- `bootstrap_lambda.sh` installs from `REPT_REQUIREMENTS_FILE` (default `requirements.lambda.txt`).
- `run_grpo_lambda.sh` can reinstall deps each run with `REPT_INSTALL_DEPS_ON_RUN=1`.

### 2. Preflight checks

```bash
cd "$REPT_ROOT"
bash scripts/preflight_lambda.sh
```

This validates env vars, venv health, imports, GPU visibility, `torch.cuda.is_available()`, endpoint reachability, vLLM mode vs GPU count, batch/generations divisibility, and output path placement.

### 3. Launch training

```bash
cd "$REPT_ROOT"

bash scripts/run_grpo_lambda.sh --dry-run
bash scripts/run_grpo_lambda.sh
```

Long-running session in the background:

```bash
nohup bash scripts/run_grpo_lambda.sh > train.log 2>&1 &
tail -f train.log
watch -n 1 nvidia-smi
```

Server mode: if vLLM never becomes ready, inspect **`$REPT_OUTPUT_DIR/vllm_serve.log`** (KV OOM, wrong GPU, etc.).

### 4. Optional local env server on the same Lambda VM

```bash
cd /path/to/ReasoningEconomicsEnv
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another shell:
export ENV_BASE_URL="http://127.0.0.1:8000"
```

### 5. Smoke validation checklist

```bash
export REPT_NUM_EPOCHS=1
export REPT_NUM_GENERATIONS=2
export REPT_BATCH_SIZE=2
export REPT_GRAD_ACCUM_OVERRIDE=1
export REPT_GRAD_ACCUM=1
cd "$REPT_ROOT"
bash scripts/run_grpo_lambda.sh
```

Confirm:

- Checkpoints and trainer outputs appear under `$REPT_OUTPUT_DIR`.
- Reward logs: `$REPT_OUTPUT_DIR/reward_logs.jsonl` (one JSON line per completed episode).
- Analysis: `pip install -r requirements.analysis.txt` then `python scripts/analyze_reward_logs.py "$REPT_OUTPUT_DIR/reward_logs.jsonl" --out-dir ./reward_analysis`.
- Caches: `$REPT_DATA_ROOT/cache` (`HF_HOME`, `TRANSFORMERS_CACHE`, `PIP_CACHE_DIR`).

---

## USC CARC (Discovery) — existing workflow

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

---

## Dependency files

- `requirements.txt` — default training dependencies (generic).
- `requirements.lambda.txt` — Lambda pins (e.g. torch 2.8.*, vLLM 0.10.2, trl 1.0.0, transformers range compatible with vLLM).
- `requirements.carc-cu121.txt` — CARC lock file for Discovery CUDA stack.

Use `requirements.lambda.txt` on Lambda instead of loosening `vllm` in `requirements.txt` (avoids pip resolver conflicts with `torch` / `pydantic`). **`transformers`** versions far above the pin may break **`vllm==0.10.2`** at import — see **`cross-chat-completion.md`**.

---

## Notes

- Training uses OpenEnv **`reset` / `step`** over the network (WebSocket session per episode in current `rollout_func` path).
- **Zero rewards / `num_steps: 1`:** ensure one WebSocket per episode (`EpisodeSession`); align **env tokenizer** with policy (`--env_tokenizer_name` / server `EnvConfig`). See **`cross-chat-completion.md`** § *Zero rewards*.
- Hosted Spaces may have limited concurrency; for sustained RL training, duplicate the Space or run the env locally.
