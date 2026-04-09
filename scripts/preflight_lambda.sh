#!/bin/bash
# preflight_lambda.sh -- Sanity checks before GRPO training on Lambda Cloud VMs.
#
# Usage:
#   export REPT_ROOT=/home/ubuntu/ReasoningEconomicsPT
#   export REPT_VENV=/home/ubuntu/.venvs/rept-lambda
#   export ENV_BASE_URL=http://127.0.0.1:8000
#   export REPT_FS_NAME=<lambda-filesystem-name>   # optional
#   export REPT_DATA_ROOT=/lambda/nfs/<fs-name>/rept  # optional override
#   bash scripts/preflight_lambda.sh

set -euo pipefail

FAIL=0
WARN=0

pass()  { echo "  [PASS] $*"; }
fail()  { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }
warn()  { echo "  [WARN] $*"; WARN=$((WARN + 1)); }

if [[ -n "${REPT_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$REPT_DATA_ROOT"
elif [[ -n "${REPT_FS_NAME:-}" ]]; then
    DATA_ROOT="/lambda/nfs/${REPT_FS_NAME}/rept"
else
    DATA_ROOT="/lambda/nfs/rept"
fi

REPT_OUTPUT_DIR="${REPT_OUTPUT_DIR:-${DATA_ROOT}/runs/grpo_train_lambda}"

echo "=== Lambda Preflight Checks ==="
echo ""

echo "--- Required environment variables ---"
for var in REPT_ROOT REPT_VENV ENV_BASE_URL; do
    if [[ -z "${!var:-}" ]]; then
        fail "$var is not set"
    else
        pass "$var = ${!var}"
    fi
done

echo ""
echo "--- Paths ---"
if [[ -n "${REPT_ROOT:-}" && -d "$REPT_ROOT" ]]; then
    pass "REPT_ROOT directory exists"
elif [[ -n "${REPT_ROOT:-}" ]]; then
    fail "REPT_ROOT directory does not exist: $REPT_ROOT"
fi

REQ_FILE="${REPT_REQUIREMENTS_FILE:-${REPT_ROOT:-}/requirements.lambda.txt}"
if [[ -n "${REPT_ROOT:-}" && -d "$REPT_ROOT" ]]; then
    echo ""
    echo "--- Requirements file ---"
    if [[ -f "$REQ_FILE" ]]; then
        pass "requirements file exists: $REQ_FILE"
    else
        warn "requirements file not found: $REQ_FILE"
        warn "  bootstrap_lambda.sh defaults to requirements.lambda.txt; set REPT_REQUIREMENTS_FILE to override."
    fi
fi

if [[ -n "${REPT_VENV:-}" && -f "$REPT_VENV/bin/activate" ]]; then
    pass "REPT_VENV has bin/activate"
elif [[ -n "${REPT_VENV:-}" ]]; then
    fail "REPT_VENV missing or no bin/activate: $REPT_VENV (run bootstrap first)"
fi

if [[ -d "$DATA_ROOT" ]]; then
    pass "Data root exists: $DATA_ROOT"
else
    warn "Data root does not exist yet: $DATA_ROOT"
    warn "  Create it or set REPT_DATA_ROOT to your mounted Lambda filesystem path."
fi

echo ""
echo "--- Output directory ---"
if [[ "$REPT_OUTPUT_DIR" == /lambda/nfs/* ]]; then
    pass "REPT_OUTPUT_DIR is on Lambda filesystem mount: $REPT_OUTPUT_DIR"
else
    warn "REPT_OUTPUT_DIR is not under /lambda/nfs: $REPT_OUTPUT_DIR"
    warn "  Large checkpoints can exhaust root volume; consider /lambda/nfs/<fs>/rept/runs/..."
fi

echo ""
echo "--- GPU checks ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >/dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tr '\n' '; ')
        pass "nvidia-smi OK ($GPU_INFO)"
    else
        fail "nvidia-smi command failed"
    fi
else
    fail "nvidia-smi not found in PATH"
fi

echo ""
echo "--- Python from venv ---"
if [[ -n "${REPT_VENV:-}" && -f "$REPT_VENV/bin/python" ]]; then
    VENV_PY="$REPT_VENV/bin/python"
    PY_VER=$("$VENV_PY" --version 2>&1 || echo "unknown")
    pass "Python: $PY_VER"

    echo ""
    echo "--- Critical imports ---"
    for mod in torch vllm trl transformers datasets openenv jmespath; do
        if "$VENV_PY" -c "import $mod" >/dev/null 2>&1; then
            pass "import $mod"
        else
            fail "import $mod failed (run bootstrap or reinstall deps)"
        fi
    done

    CUDA_OK=$("$VENV_PY" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo "0")
    if [[ "$CUDA_OK" == "1" ]]; then
        pass "torch.cuda.is_available() is True"
    else
        fail "torch.cuda.is_available() is False"
    fi

    echo ""
    echo "--- Multi-GPU / training layout ---"
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    if ! [[ "$GPU_COUNT" =~ ^[0-9]+$ ]]; then
        GPU_COUNT=0
    fi
    echo "  Visible GPUs: $GPU_COUNT"

    REPT_VLLM_MODE="${REPT_VLLM_MODE:-auto}"
    if [[ "$REPT_VLLM_MODE" != "auto" && "$REPT_VLLM_MODE" != "server" && "$REPT_VLLM_MODE" != "colocate" ]]; then
        fail "REPT_VLLM_MODE must be auto, server, or colocate (got: $REPT_VLLM_MODE)"
    fi
    REPT_VLLM_TP="${REPT_VLLM_TP:-1}"
    RESOLVED_MODE="$REPT_VLLM_MODE"
    if [[ "$RESOLVED_MODE" == "auto" ]]; then
        if [[ "$GPU_COUNT" -ge 2 ]]; then
            RESOLVED_MODE="server"
        else
            RESOLVED_MODE="colocate"
        fi
    fi
    echo "  Resolved vLLM mode: $RESOLVED_MODE (REPT_VLLM_TP=$REPT_VLLM_TP)"

    if [[ "$RESOLVED_MODE" == "server" ]]; then
        if [[ "$GPU_COUNT" -lt 2 ]]; then
            fail "vllm_mode=server requires >= 2 visible GPUs, found $GPU_COUNT"
        fi
        if [[ "$GPU_COUNT" -le "$REPT_VLLM_TP" ]]; then
            fail "GPU count ($GPU_COUNT) must be > REPT_VLLM_TP ($REPT_VLLM_TP) for server mode"
        fi
    fi

    REPT_BATCH_SIZE="${REPT_BATCH_SIZE:-8}"
    REPT_NUM_GENERATIONS="${REPT_NUM_GENERATIONS:-8}"
    rem=$((REPT_BATCH_SIZE % REPT_NUM_GENERATIONS))
    if [[ "$rem" -ne 0 ]]; then
        fail "REPT_BATCH_SIZE ($REPT_BATCH_SIZE) must be divisible by REPT_NUM_GENERATIONS ($REPT_NUM_GENERATIONS) for GRPO"
    else
        pass "REPT_BATCH_SIZE / REPT_NUM_GENERATIONS divisibility OK ($REPT_BATCH_SIZE / $REPT_NUM_GENERATIONS)"
    fi

    if [[ "$GPU_COUNT" -ge 2 ]]; then
        if "$VENV_PY" -c "import torch.distributed" >/dev/null 2>&1; then
            pass "torch.distributed available"
        else
            fail "torch.distributed not importable (needed for multi-GPU training)"
        fi
    fi

    if [[ "$RESOLVED_MODE" == "server" ]]; then
        if [[ -x "${REPT_VENV}/bin/accelerate" ]]; then
            pass "accelerate CLI in venv (server mode)"
        else
            fail "accelerate not found at ${REPT_VENV}/bin/accelerate (required for vllm_mode=server)"
        fi
    fi

    REPT_MODEL_SHARDING="${REPT_MODEL_SHARDING:-0}"
    if [[ "$REPT_MODEL_SHARDING" == "1" ]]; then
        echo ""
        echo "--- Model sharding (FSDP) ---"
        if [[ "$RESOLVED_MODE" != "server" ]]; then
            fail "REPT_MODEL_SHARDING=1 requires REPT_VLLM_MODE=server (resolved mode: $RESOLVED_MODE)"
        else
            pass "REPT_VLLM_MODE compatible with sharding (server)"
        fi
        SHARD_CFG="${REPT_ACCELERATE_CONFIG:-${REPT_ROOT}/config/accelerate/model-sharding.yaml}"
        if [[ -f "$SHARD_CFG" ]]; then
            pass "Accelerate sharding config exists: $SHARD_CFG"
        else
            fail "Accelerate sharding config missing: $SHARD_CFG (set REPT_ACCELERATE_CONFIG or add config/accelerate/model-sharding.yaml)"
        fi
        TRAIN_PROCS_CHECK=$((GPU_COUNT - REPT_VLLM_TP))
        if [[ "$TRAIN_PROCS_CHECK" -lt 2 ]]; then
            fail "REPT_MODEL_SHARDING=1 needs TRAIN_PROCS>=2 (GPUs=$GPU_COUNT, REPT_VLLM_TP=$REPT_VLLM_TP → TRAIN_PROCS=$TRAIN_PROCS_CHECK)"
        else
            pass "GPU layout allows TRAIN_PROCS=$TRAIN_PROCS_CHECK for FSDP"
        fi
    fi
else
    fail "Cannot run Python checks because REPT_VENV python is unavailable"
fi

echo ""
echo "--- Env endpoint ---"
if [[ -n "${ENV_BASE_URL:-}" ]]; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$ENV_BASE_URL/health" 2>/dev/null || echo "000")
    if [[ "$HTTP_CODE" == "200" ]]; then
        pass "ENV_BASE_URL /health returned 200"
    elif [[ "$HTTP_CODE" == "000" ]]; then
        fail "ENV_BASE_URL unreachable (timeout or DNS failure): $ENV_BASE_URL"
    else
        fail "ENV_BASE_URL /health returned HTTP $HTTP_CODE (expected 200)"
    fi
fi

echo ""
echo "--- Disk usage ---"
echo "  Home:"
df -h "$HOME" | sed -n '1,2p' || true
if [[ -d "$DATA_ROOT" ]]; then
    echo "  Data root:"
    df -h "$DATA_ROOT" | sed -n '1,2p' || true
fi

echo ""
echo "=== Summary: $FAIL failure(s), $WARN warning(s) ==="
if [[ $FAIL -gt 0 ]]; then
    echo "Fix failures before launching training."
    exit 1
fi
if [[ $WARN -gt 0 ]]; then
    echo "Warnings present -- review before launching."
fi
echo "Preflight complete."
