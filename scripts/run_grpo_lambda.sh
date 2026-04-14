#!/bin/bash
# run_grpo_lambda.sh -- Direct GRPO launcher for Lambda Cloud VMs (no Slurm).
#
# Usage:
#   export REPT_ROOT=/home/ubuntu/ReasoningEconomicsPT
#   export REPT_VENV=/home/ubuntu/.venvs/rept-lambda
#   export ENV_BASE_URL=http://127.0.0.1:8000
#   export REPT_FS_NAME=<lambda-filesystem-name>   # optional
#   bash scripts/run_grpo_lambda.sh [--dry-run]
#
# Multi-GPU: REPT_VLLM_MODE=auto picks server if >=2 visible GPUs else colocate.
# Server mode: starts trl vllm-serve on REPT_VLLM_PORT (default 8001), then accelerate launch
# on the remaining GPUs (OpenEnv typically uses 8000 — do not collide).
#
# Unsloth (optional): REPT_USE_UNSLOTH=1 → grpo_train --use_unsloth (+ REPT_UNSLOTH_* below).
# Install Unsloth after deps: pip install "unsloth[...]==..." "unsloth_zoo>=..." --no-deps (see requirements.lambda.txt).

set -euo pipefail

DRY_RUN=0

usage() {
    echo "Usage: $0 [--dry-run]"
    echo ""
    echo "Runs GRPO training directly on a Lambda VM (no sbatch)."
    echo ""
    echo "Required exports:"
    echo "  REPT_ROOT       absolute path to ReasoningEconomicsPT"
    echo "  REPT_VENV       absolute path to Python venv"
    echo "  ENV_BASE_URL    OpenEnv endpoint base URL"
    echo ""
    echo "Optional exports:"
    echo "  REPT_FS_NAME          Lambda filesystem name (used when REPT_DATA_ROOT unset)"
    echo "  REPT_DATA_ROOT        Base data path (default: /lambda/nfs/<fs>/rept or /lambda/nfs/rept)"
    echo "  REPT_MODEL            default: Qwen/Qwen3-8B (Hub id → prefetched to \$DATA_ROOT/models/...; same id → OpenEnv tokenizer)"
    echo "  REPT_OUTPUT_DIR       default: <DATA_ROOT>/runs/grpo_train_lambda"
    echo "  REPT_NUM_EPOCHS       default: 1"
    echo "  REPT_NUM_GENERATIONS  default: 8"
    echo "  REPT_BATCH_SIZE       default: 8 (must divide evenly by REPT_NUM_GENERATIONS)"
    echo "  REPT_GRAD_ACCUM       default: 8 (auto-tuned unless REPT_GRAD_ACCUM_OVERRIDE set)"
    echo "  REPT_GRAD_ACCUM_OVERRIDE  set to any value to skip grad-accum auto-tune"
    echo "  REPT_NUM_GPUS         default: auto (nvidia-smi count); ignored when REPT_GPU_LIST is set
  REPT_GPU_LIST         optional: explicit comma-separated physical GPU indices to use, e.g. "4,5,6,7"
                        Last REPT_VLLM_TP entries go to vLLM; the rest to training. Overrides REPT_NUM_GPUS auto-detection."
    echo "  REPT_VLLM_MODE        default: auto (server if >=2 GPUs, else colocate)"
    echo "  REPT_VLLM_TP          default: 1 (vLLM tensor parallel GPUs in server mode)"
    echo "  REPT_VLLM_PORT        default: 8001 (trl vllm-serve HTTP; must differ from OpenEnv, often 8000)"
    echo "  REPT_VLLM_GROUP_PORT  default: 51216 (TRL weight-sync TCP; match training --vllm_group_port)"
    echo "  REPT_VLLM_SERVER_HOST default: 127.0.0.1 (passed to grpo_train --vllm_server_host)"
    echo "  REPT_NCCL_P2P_DISABLE optional: if set, overrides NCCL_P2P_DISABLE for multi-GPU (else 1; use 0 on NVLink A100)"
    echo "  REPT_VLLM_GPU_UTIL    default: 0.9"
    echo "  REPT_VLLM_MAX_MODEL_LEN optional: positive int → --vllm_max_model_length (colocate KV cap) and trl vllm-serve --max_model_len (server)"
    echo "  REPT_GRPO_CONFIG_JSON optional: path to JSON of trl.GRPOConfig / TrainingArguments fields (merged after CLI defaults)"
    echo "  REPT_GRADIENT_CHECKPOINTING  default: 1"
    echo "  REPT_MAX_COMPLETION_LENGTH   default: 4096"
    echo "  REPT_NO_BF16          default: 0 (set to 1 for --no_bf16)"
    echo "  REPT_ACCELERATE_MAIN_PORT  optional (default: 29500) for accelerate launch"
    echo "  REPT_MODEL_SHARDING   default: 0 (set to 1 for FSDP via config/accelerate/model-sharding.yaml; requires server mode, TRAIN_PROCS>=2)"
    echo "  REPT_FSDP2_SHARDING   default: 0 (set to 1 when REPT_MODEL_SHARDING=1 to use model-sharding-fsdp2.yaml if REPT_ACCELERATE_CONFIG unset)"
    echo "  REPT_ACCELERATE_CONFIG optional path to Accelerate YAML (used when REPT_MODEL_SHARDING=1; default: <REPT_ROOT>/config/accelerate/model-sharding.yaml)"
    echo "  REPT_ALPHA            default: 1.0"
    echo "  REPT_LOG_EVERY        default: 1   (log every N weight updates)"
    echo "  REPT_USE_UNSLOTH      default: 0 (set to 1 for grpo_train --use_unsloth; install unsloth per requirements.lambda.txt)"
    echo "  REPT_UNSLOTH_LOAD_IN_4BIT default: 0 (set to 1 for --load_in_4bit; requires REPT_USE_UNSLOTH=1)"
    echo "  REPT_UNSLOTH_LORA_R   default: 16  (passed as --lora_r when REPT_USE_UNSLOTH=1)"
    echo "  REPT_UNSLOTH_LORA_ALPHA default: 16 (passed as --lora_alpha when REPT_USE_UNSLOTH=1)"
    echo "  REPT_INSTALL_DEPS_ON_RUN  default: 0 (set to 1 to pip install before run)"
    echo "  REPT_REQUIREMENTS_FILE    default: <REPT_ROOT>/requirements.lambda.txt"
    echo "  PYTORCH_WHEEL_INDEX       optional pip extra index URL"
    exit 1
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $arg"; usage ;;
    esac
done

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

REPT_MODEL="${REPT_MODEL:-Qwen/Qwen3-8B}"
# Hub/repo id before prefetch may rewrite REPT_MODEL to a local directory (for OpenEnv --env_tokenizer_name).
REPT_MODEL_HUB_ID=""
if [[ "$REPT_MODEL" != /* ]] && [[ "$REPT_MODEL" != ./* ]]; then
    REPT_MODEL_HUB_ID="$REPT_MODEL"
fi
REPT_OUTPUT_DIR="${REPT_OUTPUT_DIR:-${DATA_ROOT}/runs/grpo_train_lambda}"
REPT_NUM_EPOCHS="${REPT_NUM_EPOCHS:-1}"
REPT_NUM_GENERATIONS="${REPT_NUM_GENERATIONS:-8}"
REPT_BATCH_SIZE="${REPT_BATCH_SIZE:-8}"
REPT_GRAD_ACCUM="${REPT_GRAD_ACCUM:-8}"
REPT_NUM_GPUS="${REPT_NUM_GPUS:-auto}"
REPT_VLLM_TP="${REPT_VLLM_TP:-1}"
if ! [[ "$REPT_VLLM_TP" =~ ^[0-9]+$ ]] || [[ "$REPT_VLLM_TP" -lt 1 ]]; then
    echo "[ERROR] REPT_VLLM_TP must be a positive integer (got: $REPT_VLLM_TP)"
    exit 1
fi
REPT_VLLM_GPU_UTIL="${REPT_VLLM_GPU_UTIL:-0.9}"
REPT_VLLM_PORT="${REPT_VLLM_PORT:-8001}"
REPT_VLLM_GROUP_PORT="${REPT_VLLM_GROUP_PORT:-51216}"
REPT_VLLM_SERVER_HOST="${REPT_VLLM_SERVER_HOST:-127.0.0.1}"
REPT_VLLM_MODE="${REPT_VLLM_MODE:-auto}"
REPT_GRADIENT_CHECKPOINTING="${REPT_GRADIENT_CHECKPOINTING:-1}"
REPT_MAX_COMPLETION_LENGTH="${REPT_MAX_COMPLETION_LENGTH:-4096}"
REPT_NO_BF16="${REPT_NO_BF16:-0}"
REPT_ALPHA="${REPT_ALPHA:-1.0}"
REPT_LOG_EVERY="${REPT_LOG_EVERY:-1}"
REPT_USE_UNSLOTH="${REPT_USE_UNSLOTH:-0}"
REPT_UNSLOTH_LOAD_IN_4BIT="${REPT_UNSLOTH_LOAD_IN_4BIT:-0}"
REPT_UNSLOTH_LORA_R="${REPT_UNSLOTH_LORA_R:-16}"
REPT_UNSLOTH_LORA_ALPHA="${REPT_UNSLOTH_LORA_ALPHA:-16}"
REPT_INSTALL_DEPS_ON_RUN="${REPT_INSTALL_DEPS_ON_RUN:-0}"
REPT_REQUIREMENTS_FILE="${REPT_REQUIREMENTS_FILE:-$REPT_ROOT/requirements.lambda.txt}"
PYTORCH_WHEEL_INDEX="${PYTORCH_WHEEL_INDEX:-}"

# ---- Model Sharding Options ----
# REPT_MODEL_SHARDING:
#   0 = No sharding (default; single-process or DDP as per Accelerate/torch)
#   1 = Enable FSDP model sharding (leverages Accelerate config for model sharding; reduces memory per GPU, enables training larger models)
#      Set REPT_MODEL_SHARDING=1; template = REPT_ACCELERATE_CONFIG if set, else model-sharding-fsdp2.yaml if REPT_FSDP2_SHARDING=1, else model-sharding.yaml (v1).
REPT_MODEL_SHARDING="${REPT_MODEL_SHARDING:-0}"
export REPT_MODEL_SHARDING
REPT_DEFAULT_SHARDING_CONFIG="${REPT_ROOT}/config/accelerate/model-sharding.yaml"
REPT_DEFAULT_SHARDING_CONFIG_FSDP2="${REPT_ROOT}/config/accelerate/model-sharding-fsdp2.yaml"
REPT_FSDP2_SHARDING="${REPT_FSDP2_SHARDING:-0}"
# Explicit GPU list (e.g. "4,5,6,7"). When set, overrides REPT_NUM_GPUS auto-detection
# and uses exactly these physical device IDs for training + vLLM assignment.
REPT_GPU_LIST="${REPT_GPU_LIST:-}"

# torch configs
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"   


if [[ "$REPT_MODEL_SHARDING" != "0" && "$REPT_MODEL_SHARDING" != "1" ]]; then
    echo "[ERROR] REPT_MODEL_SHARDING must be 0 or 1 (got: $REPT_MODEL_SHARDING)"
    exit 1
fi
if [[ "$REPT_FSDP2_SHARDING" != "0" && "$REPT_FSDP2_SHARDING" != "1" ]]; then
    echo "[ERROR] REPT_FSDP2_SHARDING must be 0 or 1 (got: $REPT_FSDP2_SHARDING)"
    exit 1
fi

if [[ "$REPT_USE_UNSLOTH" != "0" && "$REPT_USE_UNSLOTH" != "1" ]]; then
    echo "[ERROR] REPT_USE_UNSLOTH must be 0 or 1 (got: $REPT_USE_UNSLOTH)"
    exit 1
fi
if [[ "$REPT_UNSLOTH_LOAD_IN_4BIT" != "0" && "$REPT_UNSLOTH_LOAD_IN_4BIT" != "1" ]]; then
    echo "[ERROR] REPT_UNSLOTH_LOAD_IN_4BIT must be 0 or 1 (got: $REPT_UNSLOTH_LOAD_IN_4BIT)"
    exit 1
fi
if [[ "$REPT_UNSLOTH_LOAD_IN_4BIT" == "1" && "$REPT_USE_UNSLOTH" != "1" ]]; then
    echo "[ERROR] REPT_UNSLOTH_LOAD_IN_4BIT=1 requires REPT_USE_UNSLOTH=1"
    exit 1
fi
if ! [[ "$REPT_UNSLOTH_LORA_R" =~ ^[0-9]+$ ]] || [[ "$REPT_UNSLOTH_LORA_R" -lt 1 ]]; then
    echo "[ERROR] REPT_UNSLOTH_LORA_R must be a positive integer (got: $REPT_UNSLOTH_LORA_R)"
    exit 1
fi
if ! [[ "$REPT_UNSLOTH_LORA_ALPHA" =~ ^[0-9]+$ ]] || [[ "$REPT_UNSLOTH_LORA_ALPHA" -lt 1 ]]; then
    echo "[ERROR] REPT_UNSLOTH_LORA_ALPHA must be a positive integer (got: $REPT_UNSLOTH_LORA_ALPHA)"
    exit 1
fi

if [[ -n "${REPT_VLLM_MAX_MODEL_LEN:-}" ]]; then
    if ! [[ "$REPT_VLLM_MAX_MODEL_LEN" =~ ^[0-9]+$ ]] || [[ "$REPT_VLLM_MAX_MODEL_LEN" -lt 1 ]]; then
        echo "[ERROR] REPT_VLLM_MAX_MODEL_LEN must be a positive integer (got: $REPT_VLLM_MAX_MODEL_LEN)"
        exit 1
    fi
fi
if [[ -n "${REPT_GRPO_CONFIG_JSON:-}" ]] && [[ ! -f "$REPT_GRPO_CONFIG_JSON" ]]; then
    echo "[ERROR] REPT_GRPO_CONFIG_JSON must be a readable file (got: $REPT_GRPO_CONFIG_JSON)"
    exit 1
fi

# ---- GPU fleet & vLLM mode ----
if [[ "$REPT_VLLM_MODE" != "auto" && "$REPT_VLLM_MODE" != "server" && "$REPT_VLLM_MODE" != "colocate" ]]; then
    echo "[ERROR] REPT_VLLM_MODE must be auto, server, or colocate (got: $REPT_VLLM_MODE)"
    exit 1
fi

# If REPT_GPU_LIST is set (e.g. "4,5,6,7"), derive REPT_NUM_GPUS from it and use
# the explicit list for device assignment. Otherwise fall back to nvidia-smi count.
if [[ -n "$REPT_GPU_LIST" ]]; then
    if ! [[ "$REPT_GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "[ERROR] REPT_GPU_LIST must be a comma-separated list of GPU indices (got: $REPT_GPU_LIST)"
        exit 1
    fi
    # Count entries
    IFS=',' read -ra _GPU_ARRAY <<< "$REPT_GPU_LIST"
    REPT_NUM_GPUS="${#_GPU_ARRAY[@]}"
    echo "  Using explicit GPU list: $REPT_GPU_LIST ($REPT_NUM_GPUS GPUs)"
elif [[ "$REPT_NUM_GPUS" == "auto" ]]; then
    REPT_NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    if ! [[ "$REPT_NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$REPT_NUM_GPUS" -lt 1 ]]; then
        REPT_NUM_GPUS=1
    fi
    echo "  Auto-detected GPUs: $REPT_NUM_GPUS"
elif ! [[ "$REPT_NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$REPT_NUM_GPUS" -lt 1 ]]; then
    echo "[ERROR] REPT_NUM_GPUS must be auto or a positive integer (got: $REPT_NUM_GPUS)"
    exit 1
fi

if [[ "$REPT_VLLM_MODE" == "auto" ]]; then
    if [[ "$REPT_NUM_GPUS" -ge 2 ]]; then
        REPT_VLLM_MODE="server"
    else
        REPT_VLLM_MODE="colocate"
    fi
    echo "  Auto-selected vLLM mode: $REPT_VLLM_MODE"
fi

if [[ "$REPT_VLLM_MODE" == "server" ]]; then
    TRAIN_PROCS=$((REPT_NUM_GPUS - REPT_VLLM_TP))
    if [[ "$TRAIN_PROCS" -lt 1 ]]; then
        echo "[ERROR] Not enough GPUs for server mode: $REPT_NUM_GPUS total, TP=$REPT_VLLM_TP"
        echo "Need at least (TP + 1) GPUs. Use colocate mode or reduce REPT_VLLM_TP."
        exit 1
    fi
else
    TRAIN_PROCS=1
fi

TARGET_EFFECTIVE_PROMPTS=16
EFFECTIVE_PER_STEP=$((REPT_BATCH_SIZE * TRAIN_PROCS))
if [[ -z "${REPT_GRAD_ACCUM_OVERRIDE:-}" ]] && [[ "$EFFECTIVE_PER_STEP" -gt 0 ]]; then
    AUTO_GRAD_ACCUM=$(( (TARGET_EFFECTIVE_PROMPTS + EFFECTIVE_PER_STEP - 1) / EFFECTIVE_PER_STEP ))
    if [[ "$AUTO_GRAD_ACCUM" -lt 1 ]]; then
        AUTO_GRAD_ACCUM=1
    fi
    REPT_GRAD_ACCUM="$AUTO_GRAD_ACCUM"
    echo "  Auto-adjusted grad_accum to $REPT_GRAD_ACCUM (target $TARGET_EFFECTIVE_PROMPTS effective prompts/step)"
fi

if [[ "$TRAIN_PROCS" -gt 1 ]]; then
    export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
    # Prefer REPT_NCCL_P2P_DISABLE when set; else inherit NCCL_P2P_DISABLE; default 1 (PCIe/V100-safe).
    # On NVLink A100 SXM4, set REPT_NCCL_P2P_DISABLE=0 or export NCCL_P2P_DISABLE=0 before the script.
    export NCCL_P2P_DISABLE="${REPT_NCCL_P2P_DISABLE:-${NCCL_P2P_DISABLE:-1}}"
    echo "  NCCL: NCCL_TIMEOUT=${NCCL_TIMEOUT} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} (multi-process training)"
fi

if [[ "$REPT_MODEL_SHARDING" == "1" ]]; then
    if [[ "$REPT_VLLM_MODE" != "server" ]]; then
        echo "[ERROR] REPT_MODEL_SHARDING=1 requires REPT_VLLM_MODE=server (got: $REPT_VLLM_MODE)."
        exit 1
    fi
    if [[ "$TRAIN_PROCS" -lt 2 ]]; then
        echo "[ERROR] REPT_MODEL_SHARDING=1 requires at least 2 training GPUs/processes (TRAIN_PROCS=$TRAIN_PROCS)."
        exit 1
    fi
fi

CACHE_ROOT="${DATA_ROOT}/cache"
mkdir -p "$CACHE_ROOT/pip" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/tmp" "$REPT_OUTPUT_DIR"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TMPDIR="$CACHE_ROOT/tmp"

if [[ ! -d "$REPT_ROOT" ]]; then
    echo "[ERROR] REPT_ROOT does not exist: $REPT_ROOT"
    exit 1
fi

if [[ ! -f "$REPT_VENV/bin/activate" ]]; then
    echo "[ERROR] REPT_VENV has no bin/activate: $REPT_VENV"
    echo "Run scripts/bootstrap_lambda.sh first."
    exit 1
fi

if [[ "$REPT_OUTPUT_DIR" != /lambda/nfs/* ]]; then
    echo "[WARN] REPT_OUTPUT_DIR is not under /lambda/nfs: $REPT_OUTPUT_DIR"
fi

cd "$REPT_ROOT"
mkdir -p logs
# shellcheck source=/dev/null
source "$REPT_VENV/bin/activate"

if [[ "$REPT_INSTALL_DEPS_ON_RUN" == "1" ]]; then
    if [[ ! -f "$REPT_REQUIREMENTS_FILE" ]]; then
        echo "[ERROR] requirements file not found: $REPT_REQUIREMENTS_FILE"
        exit 1
    fi
    echo ">>> Installing dependencies before run..."
    pip install --quiet --upgrade pip
    if [[ -n "$PYTORCH_WHEEL_INDEX" ]]; then
        pip install -r "$REPT_REQUIREMENTS_FILE" --extra-index-url "$PYTORCH_WHEEL_INDEX"
    else
        pip install -r "$REPT_REQUIREMENTS_FILE"
    fi
fi

GPU_LINE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "n/a")
echo "=== REPT GRPO Training (Lambda) ==="
echo "  Host:            $(hostname)"
echo "  GPUs:            ${REPT_NUM_GPUS} x ${GPU_LINE}"
echo "  vLLM mode:       $REPT_VLLM_MODE"
echo "  vLLM TP:         ${REPT_VLLM_TP} GPU(s)"
echo "  Training procs:  ${TRAIN_PROCS}"
echo "  Python:          $(python --version)"
echo "  Torch:           $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA avail:      $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Model:           $REPT_MODEL"
echo "  Batch/device:    $REPT_BATCH_SIZE"
echo "  Num generations: $REPT_NUM_GENERATIONS"
echo "  Grad accum:      $REPT_GRAD_ACCUM"
echo "  Output dir:      $REPT_OUTPUT_DIR"
echo "  Env URL:         $ENV_BASE_URL"
if [[ -n "${REPT_VLLM_MAX_MODEL_LEN:-}" ]]; then
    echo "  vLLM max context:  $REPT_VLLM_MAX_MODEL_LEN tokens (--vllm_max_model_length; server: also trl vllm-serve --max_model_len)"
fi
if [[ -n "${REPT_GRPO_CONFIG_JSON:-}" ]]; then
    echo "  GRPO JSON merge:   $REPT_GRPO_CONFIG_JSON (--grpo_config_json)"
fi
if [[ "$REPT_MODEL_SHARDING" == "1" ]]; then
    if [[ -n "${REPT_ACCELERATE_CONFIG:-}" ]]; then
        echo "  Model sharding:  FSDP (Accelerate; REPT_ACCELERATE_CONFIG=$REPT_ACCELERATE_CONFIG)"
    elif [[ "$REPT_FSDP2_SHARDING" == "1" ]]; then
        echo "  Model sharding:  FSDP (Accelerate; template model-sharding-fsdp2.yaml)"
    else
        echo "  Model sharding:  FSDP (Accelerate; see config/accelerate/model-sharding.yaml)"
    fi
fi
if [[ "$REPT_USE_UNSLOTH" == "1" ]]; then
    echo "  Unsloth:         enabled (--use_unsloth; LoRA r=$REPT_UNSLOTH_LORA_R alpha=$REPT_UNSLOTH_LORA_ALPHA)"
    if [[ "$REPT_UNSLOTH_LOAD_IN_4BIT" == "1" ]]; then
        echo "  Unsloth 4-bit:   --load_in_4bit"
    fi
fi
echo "==============================="

# ------------------------------------------------------------------ dependency precheck
echo ">>> Verifying critical Python imports..."
for mod in torch vllm trl transformers datasets huggingface_hub openenv jmespath; do
    if python -c "import $mod" >/dev/null 2>&1; then
        echo "  [PASS] import $mod"
    else
        echo "  [FAIL] import $mod"
        echo "Install/update dependencies first (bootstrap_lambda.sh or REPT_INSTALL_DEPS_ON_RUN=1)."
        exit 1
    fi
done

if [[ "$REPT_USE_UNSLOTH" == "1" ]]; then
    if python -c "import unsloth" >/dev/null 2>&1; then
        echo "  [PASS] import unsloth"
    else
        echo "  [FAIL] import unsloth (REPT_USE_UNSLOTH=1)"
        echo "Install Unsloth after main deps, e.g.: pip install \"unsloth[cu128onlytorch280]==2026.4.4\" \"unsloth_zoo>=2026.4.3,<2026.5\" --no-deps"
        echo "See comments in requirements.lambda.txt for CUDA/torch extras."
        exit 1
    fi
fi

if [[ "$REPT_VLLM_MODE" == "server" ]]; then
    if ! command -v accelerate >/dev/null 2>&1; then
        echo "  [FAIL] accelerate CLI not on PATH (required for vllm_mode=server)"
        exit 1
    fi
    echo "  [PASS] accelerate CLI available"
fi

# ---- Prefetch Hub models to a plain directory (avoids NFS + concurrent Hub cache races) ----
# If REPT_MODEL is already a path (absolute or ./...), use it as-is.
if [[ $DRY_RUN -eq 0 ]] && [[ "$REPT_MODEL" != /* ]] && [[ "$REPT_MODEL" != ./* ]]; then
    MODEL_LOCAL_DIR="${DATA_ROOT}/models/${REPT_MODEL//\//_}"
    mkdir -p "${DATA_ROOT}/models"
    if [[ ! -f "$MODEL_LOCAL_DIR/model.safetensors.index.json" ]] && \
       [[ ! -f "$MODEL_LOCAL_DIR/model.safetensors" ]]; then
        echo ">>> Prefetching model to plain directory: $MODEL_LOCAL_DIR"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${REPT_MODEL}',
    local_dir='${MODEL_LOCAL_DIR}',
    local_dir_use_symlinks=False,
)
print('Prefetch complete.')
"
    else
        echo ">>> Model already present at: $MODEL_LOCAL_DIR"
    fi
    REPT_MODEL="$MODEL_LOCAL_DIR"
    echo ">>> Training will load model from: $REPT_MODEL"
fi

VLLM_PID=""
if [[ "$REPT_VLLM_MODE" == "server" ]]; then
    # Build ordered GPU list: either from REPT_GPU_LIST or sequential 0..N-1
    if [[ -n "$REPT_GPU_LIST" ]]; then
        IFS=',' read -ra _GPU_ARRAY <<< "$REPT_GPU_LIST"
    else
        _GPU_ARRAY=()
        for i in $(seq 0 $((REPT_NUM_GPUS - 1))); do _GPU_ARRAY+=("$i"); done
    fi
    # Last REPT_VLLM_TP GPUs → vLLM; remainder → training
    _TRAIN_GPUS=("${_GPU_ARRAY[@]:0:$((${#_GPU_ARRAY[@]} - REPT_VLLM_TP))}")
    _VLLM_GPUS=("${_GPU_ARRAY[@]:$((${#_GPU_ARRAY[@]} - REPT_VLLM_TP))}")
    TRAIN_CUDA_DEVS=$(IFS=,; echo "${_TRAIN_GPUS[*]}")
    VLLM_CUDA_DEVS=$(IFS=,; echo "${_VLLM_GPUS[*]}")

    VLLM_CURL_HOST="$REPT_VLLM_SERVER_HOST"
    if [[ "$VLLM_CURL_HOST" == "0.0.0.0" ]]; then
        VLLM_CURL_HOST="127.0.0.1"
    fi
    VLLM_URL="http://${VLLM_CURL_HOST}:${REPT_VLLM_PORT}"

    VLLM_SERVE_EXTRA=()
    if [[ -n "${REPT_VLLM_MAX_MODEL_LEN:-}" ]]; then
        VLLM_SERVE_EXTRA+=(--max_model_len "$REPT_VLLM_MAX_MODEL_LEN")
    fi

    if [[ $DRY_RUN -eq 0 ]]; then
        if ! command -v trl >/dev/null 2>&1; then
            echo "[ERROR] trl CLI not on PATH (required to start trl vllm-serve in server mode)"
            exit 1
        fi
        echo ">>> Starting trl vllm-serve on GPU(s) [$VLLM_CUDA_DEVS] port $REPT_VLLM_PORT ..."
        CUDA_VISIBLE_DEVICES="$VLLM_CUDA_DEVS" \
            trl vllm-serve \
                --model "$REPT_MODEL" \
                --tensor-parallel-size "$REPT_VLLM_TP" \
                --gpu-memory-utilization "$REPT_VLLM_GPU_UTIL" \
                --port "$REPT_VLLM_PORT" \
                "${VLLM_SERVE_EXTRA[@]}" \
            >> "$REPT_OUTPUT_DIR/vllm_serve.log" 2>&1 &
        VLLM_PID=$!
        echo "  vllm-serve PID: $VLLM_PID  (log: $REPT_OUTPUT_DIR/vllm_serve.log)"

        echo ">>> Waiting for trl vllm-serve at $VLLM_URL ..."
        VLLM_READY=0
        for i in $(seq 1 120); do
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                echo "[ERROR] vllm-serve process exited early (PID $VLLM_PID). Check $REPT_OUTPUT_DIR/vllm_serve.log"
                exit 1
            fi
            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
                "${VLLM_URL}/get_world_size/" 2>/dev/null || echo "000")
            if [[ "$HTTP_CODE" == "200" ]]; then
                VLLM_READY=1
                echo "  [PASS] trl vllm-serve ready at ${VLLM_URL} (attempt $i)"
                break
            fi
            sleep 5
        done

        if [[ "$VLLM_READY" -ne 1 ]]; then
            echo "[ERROR] trl vllm-serve did not become ready within 600s. Check $REPT_OUTPUT_DIR/vllm_serve.log"
            kill "$VLLM_PID" 2>/dev/null || true
            exit 1
        fi

        # shellcheck disable=SC2064
        trap "echo '>>> Stopping vllm-serve (PID $VLLM_PID)...'; kill '$VLLM_PID' 2>/dev/null || true" EXIT
    else
        echo "[DRY RUN] Would start: env CUDA_VISIBLE_DEVICES=$VLLM_CUDA_DEVS trl vllm-serve --model <path after prefetch> --tensor-parallel-size $REPT_VLLM_TP --gpu-memory-utilization $REPT_VLLM_GPU_UTIL --port $REPT_VLLM_PORT ${VLLM_SERVE_EXTRA[*]}"
    fi
fi

# OpenEnv tokenizer id = REPT_MODEL when it was a Hub id; after prefetch, --model is local so pass saved id only then.
ENV_TOKENIZER_ARG=""
if [[ -n "$REPT_MODEL_HUB_ID" ]] && { [[ "$REPT_MODEL" == /* ]] || [[ "$REPT_MODEL" == ./* ]]; }; then
    ENV_TOKENIZER_ARG="$REPT_MODEL_HUB_ID"
fi

COMMON_ARGS=(
    -m training.grpo_train
    --model "$REPT_MODEL"
    --env_base_url "$ENV_BASE_URL"
    --alpha "$REPT_ALPHA"
    --log_every_n_steps "$REPT_LOG_EVERY"
    --num_train_epochs "$REPT_NUM_EPOCHS"
    --num_generations "$REPT_NUM_GENERATIONS"
    --per_device_train_batch_size "$REPT_BATCH_SIZE"
    --gradient_accumulation_steps "$REPT_GRAD_ACCUM"
    --vllm_mode "$REPT_VLLM_MODE"
    --vllm_tensor_parallel_size "$REPT_VLLM_TP"
    --vllm_gpu_memory_utilization "$REPT_VLLM_GPU_UTIL"
    --vllm_server_host "$REPT_VLLM_SERVER_HOST"
    --vllm_server_port "$REPT_VLLM_PORT"
    --vllm_group_port "$REPT_VLLM_GROUP_PORT"
    --max_completion_length "$REPT_MAX_COMPLETION_LENGTH"
    --output_dir "$REPT_OUTPUT_DIR"
)
if [[ -n "${REPT_VLLM_MAX_MODEL_LEN:-}" ]]; then
    COMMON_ARGS+=(--vllm_max_model_length "$REPT_VLLM_MAX_MODEL_LEN")
fi
if [[ -n "${REPT_GRPO_CONFIG_JSON:-}" ]]; then
    COMMON_ARGS+=(--grpo_config_json "$REPT_GRPO_CONFIG_JSON")
fi
if [[ -n "$ENV_TOKENIZER_ARG" ]]; then
    COMMON_ARGS+=(--env_tokenizer_name "$ENV_TOKENIZER_ARG")
fi

if [[ "${REPT_GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
    COMMON_ARGS+=(--gradient_checkpointing)
fi

if [[ "${REPT_NO_BF16:-0}" == "1" ]]; then
    COMMON_ARGS+=(--no_bf16)
fi

if [[ "$REPT_USE_UNSLOTH" == "1" ]]; then
    COMMON_ARGS+=(--use_unsloth --lora_r "$REPT_UNSLOTH_LORA_R" --lora_alpha "$REPT_UNSLOTH_LORA_ALPHA")
    if [[ "$REPT_UNSLOTH_LOAD_IN_4BIT" == "1" ]]; then
        COMMON_ARGS+=(--load_in_4bit)
    fi
fi

# ---- FSDP / model sharding (Accelerate --config_file) ----
REPT_ACCEL_CONFIG_FILE=""
if [[ "$REPT_MODEL_SHARDING" == "1" ]]; then
    if [[ -n "${REPT_ACCELERATE_CONFIG:-}" ]]; then
        SHARD_SRC="$REPT_ACCELERATE_CONFIG"
    elif [[ "$REPT_FSDP2_SHARDING" == "1" ]]; then
        SHARD_SRC="$REPT_DEFAULT_SHARDING_CONFIG_FSDP2"
    else
        SHARD_SRC="$REPT_DEFAULT_SHARDING_CONFIG"
    fi
    if [[ ! -f "$SHARD_SRC" ]]; then
        echo "[ERROR] Accelerate config for model sharding not found: $SHARD_SRC"
        exit 1
    fi
    REPT_ACCEL_CONFIG_FILE="${REPT_OUTPUT_DIR}/accelerate_model_sharding_runtime.yaml"
    if [[ $DRY_RUN -eq 0 ]]; then
        mkdir -p "$REPT_OUTPUT_DIR"
        sed -E "s/^num_processes:[[:space:]]*[0-9]+/num_processes: ${TRAIN_PROCS}/" "$SHARD_SRC" > "$REPT_ACCEL_CONFIG_FILE"
        echo ">>> Model sharding: patched Accelerate config → $REPT_ACCEL_CONFIG_FILE (num_processes=$TRAIN_PROCS)"
    fi
fi

if [[ "$REPT_VLLM_MODE" == "server" ]]; then
    LAUNCH_ARGS=(
        accelerate launch
        --num_processes "$TRAIN_PROCS"
        --main_process_port "${REPT_ACCELERATE_MAIN_PORT:-29500}"
    )
    if [[ -n "$REPT_ACCEL_CONFIG_FILE" ]]; then
        LAUNCH_ARGS+=(--config_file "$REPT_ACCEL_CONFIG_FILE")
    fi
    TRAIN_CMD=(
        env CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_DEVS"
        "${LAUNCH_ARGS[@]}"
        "${COMMON_ARGS[@]}"
    )
else
    TRAIN_CMD=(python "${COMMON_ARGS[@]}")
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "[DRY RUN] Command:"
    printf '  %q' "${TRAIN_CMD[@]}"
    echo ""
    if [[ -n "$REPT_ACCEL_CONFIG_FILE" ]]; then
        echo "[DRY RUN] Would write patched Accelerate config to: $REPT_ACCEL_CONFIG_FILE"
    fi
    exit 0
fi

"${TRAIN_CMD[@]}"

echo "=== Training complete. Artifacts at: $REPT_OUTPUT_DIR ==="
