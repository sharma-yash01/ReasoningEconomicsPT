#!/bin/bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  echo "Usage: $0"
  echo "Reads REPT_* and ENV_BASE_URL exports, then submits run_grpo_carc.sbatch."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/run_grpo_carc.sbatch"

if [[ ! -f "${SBATCH_FILE}" ]]; then
  echo "Missing sbatch file: ${SBATCH_FILE}"
  exit 1
fi

sbatch "${SBATCH_FILE}"
