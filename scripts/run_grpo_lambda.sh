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
# Server mode uses accelerate launch; vLLM occupies REPT_VLLM_TP GPUs (tensor parallel).

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
    echo "  REPT_MODEL            default: Qwen/Qwen3-8B (Hub id → prefetched to \$DATA_ROOT/models/...)"
    echo "  REPT_OUTPUT_DIR       default: <DATA_ROOT>/runs/grpo_train_lambda"
    echo "  REPT_NUM_EPOCHS       default: 1"
    echo "  REPT_NUM_GENERATIONS  default: 8"
    echo "  REPT_BATCH_SIZE       default: 8 (must divide evenly by REPT_NUM_GENERATIONS)"
    echo "  REPT_GRAD_ACCUM       default: 8 (auto-tuned unless REPT_GRAD_ACCUM_OVERRIDE set)"
    echo "  REPT_GRAD_ACCUM_OVERRIDE  set to any value to skip grad-accum auto-tune"
    echo "  REPT_NUM_GPUS         default: auto (nvidia-smi count)"
    echo "  REPT_VLLM_MODE        default: auto (server if >=2 GPUs, else colocate)"
    echo "  REPT_VLLM_TP          default: 1 (vLLM tensor parallel GPUs in server mode)"
    echo "  REPT_VLLM_GPU_UTIL    default: 0.9"
    echo "  REPT_GRADIENT_CHECKPOINTING  default: 1"
    echo "  REPT_MAX_COMPLETION_LENGTH   default: 4096"
    echo "  REPT_NO_BF16          default: 0 (set to 1 for --no_bf16)"
    echo "  REPT_ACCELERATE_MAIN_PORT  optional (default: 29500) for accelerate launch"
    echo "  REPT_ALPHA            default: 1.0"
    echo "  REPT_LOG_EVERY        default: 1"
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
REPT_VLLM_MODE="${REPT_VLLM_MODE:-auto}"
REPT_GRADIENT_CHECKPOINTING="${REPT_GRADIENT_CHECKPOINTING:-1}"
REPT_MAX_COMPLETION_LENGTH="${REPT_MAX_COMPLETION_LENGTH:-4096}"
REPT_NO_BF16="${REPT_NO_BF16:-0}"
REPT_ALPHA="${REPT_ALPHA:-1.0}"
REPT_LOG_EVERY="${REPT_LOG_EVERY:-1}"
REPT_INSTALL_DEPS_ON_RUN="${REPT_INSTALL_DEPS_ON_RUN:-0}"
REPT_REQUIREMENTS_FILE="${REPT_REQUIREMENTS_FILE:-$REPT_ROOT/requirements.lambda.txt}"
PYTORCH_WHEEL_INDEX="${PYTORCH_WHEEL_INDEX:-}"

# ---- GPU fleet & vLLM mode (respects CUDA_VISIBLE_DEVICES via nvidia-smi) ----
if [[ "$REPT_VLLM_MODE" != "auto" && "$REPT_VLLM_MODE" != "server" && "$REPT_VLLM_MODE" != "colocate" ]]; then
    echo "[ERROR] REPT_VLLM_MODE must be auto, server, or colocate (got: $REPT_VLLM_MODE)"
    exit 1
fi

if [[ "$REPT_NUM_GPUS" == "auto" ]]; then
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
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    echo "  NCCL: NCCL_TIMEOUT=${NCCL_TIMEOUT} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} (multi-process training)"
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
    --max_completion_length "$REPT_MAX_COMPLETION_LENGTH"
    --output_dir "$REPT_OUTPUT_DIR"
)

if [[ "${REPT_GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
    COMMON_ARGS+=(--gradient_checkpointing)
fi

if [[ "${REPT_NO_BF16:-0}" == "1" ]]; then
    COMMON_ARGS+=(--no_bf16)
fi

if [[ "$REPT_VLLM_MODE" == "server" ]]; then
    TRAIN_CMD=(
        accelerate launch
        --num_processes "$TRAIN_PROCS"
        --main_process_port "${REPT_ACCELERATE_MAIN_PORT:-29500}"
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
    exit 0
fi

"${TRAIN_CMD[@]}"

echo "=== Training complete. Artifacts at: $REPT_OUTPUT_DIR ==="
