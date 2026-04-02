#!/bin/bash
# preflight_carc.sh -- Pre-submission sanity checks for GRPO training on CARC.
#
# Run from the login node BEFORE sbatch submission to catch config errors early.
#
# Usage:
#   export REPT_ROOT=... REPT_VENV=... ENV_BASE_URL=... REPT_OUTPUT_DIR=...
#   bash scripts/preflight_carc.sh

set -euo pipefail

FAIL=0
WARN=0

pass()  { echo "  [PASS] $*"; }
fail()  { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }
warn()  { echo "  [WARN] $*"; WARN=$((WARN + 1)); }

echo "=== CARC Preflight Checks ==="
echo ""

# ------------------------------------------------------------------ env vars
echo "--- Required environment variables ---"
for var in REPT_ROOT REPT_VENV ENV_BASE_URL; do
    if [[ -z "${!var:-}" ]]; then
        fail "$var is not set"
    else
        pass "$var = ${!var}"
    fi
done

# ------------------------------------------------------------------ paths exist
echo ""
echo "--- Paths ---"
if [[ -n "${REPT_ROOT:-}" && -d "$REPT_ROOT" ]]; then
    pass "REPT_ROOT directory exists"
elif [[ -n "${REPT_ROOT:-}" ]]; then
    fail "REPT_ROOT directory does not exist: $REPT_ROOT"
fi

if [[ -n "${REPT_VENV:-}" && -f "$REPT_VENV/bin/activate" ]]; then
    pass "REPT_VENV has bin/activate"
elif [[ -n "${REPT_VENV:-}" ]]; then
    fail "REPT_VENV missing or no bin/activate: $REPT_VENV (run bootstrap first)"
fi

# ------------------------------------------------------------------ output dir on scratch
echo ""
echo "--- Output directory ---"
REPT_OUTPUT_DIR="${REPT_OUTPUT_DIR:-runs/grpo_train_carc}"
if [[ "$REPT_OUTPUT_DIR" == /scratch1/* ]]; then
    pass "REPT_OUTPUT_DIR is on /scratch1: $REPT_OUTPUT_DIR"
else
    warn "REPT_OUTPUT_DIR is NOT on /scratch1: $REPT_OUTPUT_DIR"
    warn "  Model checkpoints can be large. Consider: /scratch1/$(whoami)/rept/runs/..."
fi

# ------------------------------------------------------------------ modules
echo ""
echo "--- Modules ---"
if command -v module &>/dev/null; then
    pass "module command available"
else
    warn "module command not found (expected on login/compute nodes)"
fi

# ------------------------------------------------------------------ python in venv
echo ""
echo "--- Python (from venv) ---"
if [[ -n "${REPT_VENV:-}" && -f "$REPT_VENV/bin/python" ]]; then
    VENV_PY="$REPT_VENV/bin/python"
    PY_VER=$("$VENV_PY" --version 2>&1 || echo "unknown")
    pass "Python: $PY_VER"

    echo ""
    echo "--- Critical imports ---"
    for mod in torch vllm trl transformers datasets openenv jmespath; do
        if "$VENV_PY" -c "import $mod" 2>/dev/null; then
            pass "import $mod"
        else
            fail "import $mod failed (run bootstrap or pip install)"
        fi
    done
else
    warn "Skipping import checks (venv not accessible)"
fi

# ------------------------------------------------------------------ endpoint reachability
echo ""
echo "--- Env endpoint ---"
if [[ -n "${ENV_BASE_URL:-}" ]]; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$ENV_BASE_URL/health" 2>/dev/null || echo "000")
    if [[ "$HTTP_CODE" == "200" ]]; then
        pass "ENV_BASE_URL /health returned 200"
    elif [[ "$HTTP_CODE" == "000" ]]; then
        warn "ENV_BASE_URL unreachable (timeout or DNS failure): $ENV_BASE_URL"
        warn "  CARC login nodes may block outbound HTTP; this may work on compute nodes."
    else
        warn "ENV_BASE_URL /health returned HTTP $HTTP_CODE (expected 200)"
    fi
fi

# ------------------------------------------------------------------ disk quota
echo ""
echo "--- Disk usage ---"
HOME_USAGE=$(du -sh "$HOME" 2>/dev/null | cut -f1 || echo "?")
echo "  Home ($HOME): $HOME_USAGE"
if [[ -d /scratch1/$(whoami) ]]; then
    SCRATCH_USAGE=$(du -sh "/scratch1/$(whoami)" 2>/dev/null | cut -f1 || echo "?")
    echo "  Scratch (/scratch1/$(whoami)): $SCRATCH_USAGE"
    pass "/scratch1/$(whoami) exists"
else
    warn "/scratch1/$(whoami) does not exist. Create it before training."
fi

# ------------------------------------------------------------------ summary
echo ""
echo "=== Summary: $FAIL failure(s), $WARN warning(s) ==="
if [[ $FAIL -gt 0 ]]; then
    echo "Fix failures before submitting."
    exit 1
fi
if [[ $WARN -gt 0 ]]; then
    echo "Warnings present -- review before submitting."
fi
echo "Preflight complete."
