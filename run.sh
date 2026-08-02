#!/usr/bin/env bash
# ==============================================================================
# SKROA Enterprise Execution Pipeline
# Target Environment: Zorin OS / Ubuntu Linux x86_64
# ==============================================================================

set -euo pipefail

# 1. Resolve absolute script path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 2. Verify virtual environment availability
if [[ ! -d "venv" ]]; then
    echo "[ERROR] Virtual environment not found. Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt" >&2
    exit 1
fi

# 3. Activate environment and export Python path
source venv/bin/activate
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# 4. Create output directory structures if missing
mkdir -p results/logs results/plots

echo "[INFO] Running KRNA Unit Testing Suite..."
python3 -m unittest discover -s tests -p "test_*.py" -v

echo "[INFO] Launching Head-to-Head SKROA vs. PSO Benchmarking Suite..."
python3 -m krna.benchmarks "$@"

echo "[SUCCESS] Pipeline execution complete. Check results/logs/ and results/plots/."