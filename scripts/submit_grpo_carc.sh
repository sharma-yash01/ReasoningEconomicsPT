#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/run_grpo_carc.sbatch"
DRY_RUN=0

usage() {
    echo "Usage: $0 [--dry-run]"
    echo ""
    echo "Validates config, prints resolved run summary, and submits run_grpo_carc.sbatch."
    echo ""
    echo "Required exports:"
    echo "  REPT_ROOT       absolute path to ReasoningEconomicsPT on CARC"
    echo "  REPT_VENV       absolute path to Python venv"
    echo "  ENV_BASE_URL    OpenEnv space URL"
    echo ""
    echo "Options:"
    echo "  --dry-run       Print config and exit without submitting"
    exit 1
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $arg"; usage ;;
    esac
done

# ------------------------------------------------------------------ validate
ERRORS=0
for var in REPT_ROOT REPT_VENV ENV_BASE_URL; do
    if [[ -z "${!var:-}" ]]; then
        echo "[ERROR] $var is not set."
        ERRORS=$((ERRORS + 1))
    fi
done

if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "[ERROR] Missing sbatch file: ${SBATCH_FILE}"
    ERRORS=$((ERRORS + 1))
fi

if [[ -n "${REPT_ROOT:-}" && ! -d "${REPT_ROOT}" ]]; then
    echo "[ERROR] REPT_ROOT does not exist: ${REPT_ROOT}"
    ERRORS=$((ERRORS + 1))
fi

if [[ -n "${REPT_VENV:-}" && ! -f "${REPT_VENV}/bin/activate" ]]; then
    echo "[ERROR] REPT_VENV has no bin/activate: ${REPT_VENV} (run bootstrap first)"
    ERRORS=$((ERRORS + 1))
fi

if [[ $ERRORS -gt 0 ]]; then
    echo ""
    echo "$ERRORS error(s). Fix before submitting."
    exit 1
fi

# ------------------------------------------------------------------ resolved config
REPT_MODEL="${REPT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
REPT_OUTPUT_DIR="${REPT_OUTPUT_DIR:-/scratch1/$USER/rept/runs/grpo_train_carc}"
REPT_NUM_EPOCHS="${REPT_NUM_EPOCHS:-1}"
REPT_NUM_GENERATIONS="${REPT_NUM_GENERATIONS:-8}"
REPT_BATCH_SIZE="${REPT_BATCH_SIZE:-2}"
REPT_GRAD_ACCUM="${REPT_GRAD_ACCUM:-8}"
REPT_VLLM_MODE="${REPT_VLLM_MODE:-colocate}"
REPT_ALPHA="${REPT_ALPHA:-1.0}"
REPT_LOG_EVERY="${REPT_LOG_EVERY:-1}"

echo "=== Resolved Run Config ==="
echo "  REPT_ROOT        = ${REPT_ROOT}"
echo "  REPT_VENV        = ${REPT_VENV}"
echo "  ENV_BASE_URL     = ${ENV_BASE_URL}"
echo "  Model            = ${REPT_MODEL}"
echo "  Output dir       = ${REPT_OUTPUT_DIR}"
echo "  Epochs           = ${REPT_NUM_EPOCHS}"
echo "  Generations      = ${REPT_NUM_GENERATIONS}"
echo "  Batch size       = ${REPT_BATCH_SIZE}"
echo "  Grad accum       = ${REPT_GRAD_ACCUM}"
echo "  vLLM mode        = ${REPT_VLLM_MODE}"
echo "  Alpha            = ${REPT_ALPHA}"
echo "  Log every        = ${REPT_LOG_EVERY}"
echo "  Partition/GPU    = gpu / a40:1"
echo "==========================="

if [[ "$REPT_OUTPUT_DIR" != /scratch1/* ]]; then
    echo "[WARN] REPT_OUTPUT_DIR is not on /scratch1. Large checkpoints may exhaust home quota."
fi

# ------------------------------------------------------------------ submit or dry-run
if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "[DRY RUN] Would submit: sbatch ${SBATCH_FILE}"
    echo "Re-run without --dry-run to submit."
    exit 0
fi

echo ""
sbatch "${SBATCH_FILE}"
