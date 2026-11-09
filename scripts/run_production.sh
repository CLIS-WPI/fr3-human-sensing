#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
O=results/rerun_v2/full
cp $O/manuscript_numbers.json $O/manuscript_numbers_reference.json
python -m fr3isac.experiments.rerun_v2 rt --out $O
python -m fr3isac.experiments.rerun_v2 analyze --out $O
python -m fr3isac.experiments.rerun_v2 numbers --out $O
python scripts/verify.py --tier 2 --reference $O/manuscript_numbers_reference.json
echo "Tier 2 OK: required EIRP reproduced."
