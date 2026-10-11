#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

python scripts/check_environment.py --require-gpu
ttc-eval generate \
  --model "${MODEL_PATH:-Qwen/Qwen3-0.6B}" \
  --limit 100 \
  --samples 16 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --max-tokens 2048 \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --batch-size "${BATCH_SIZE:-25}"

test "$(wc -l < artifacts/base/generations.jsonl)" -eq 100
echo "CHECKPOINT PASSED: 100 problems generated"

ttc-eval score
test "$(wc -l < artifacts/base/scored.jsonl)" -eq 100
echo "CHECKPOINT PASSED: 100 problems verified"

ttc-eval report --by-level
test -s artifacts/base/metrics.json
test -s artifacts/base/pass_at_k.png
python scripts/validate_artifacts.py base
