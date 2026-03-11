#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_PY="$PROJECT_ROOT/analysis_pipeline.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

SLICES=("244" "288" "320" "348")

for SLICE in "${SLICES[@]}"; do
  echo "\n=== Running pipeline for slice ${SLICE} ==="
  "$PYTHON_BIN" "$PIPELINE_PY" --slice "$SLICE"
done
