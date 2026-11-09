#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
O=results/rerun_v2/full
cp $O/gates.json $O/gates_reference.json
python -m fr3isac.experiments.validation_gates --out $O/gates_rt
python -m fr3isac.experiments.rerun_v2 gates --out $O
python scripts/verify.py --tier 1 --reference $O/gates_reference.json
echo "Tier 1 OK: gates G1-G11 pass and match the reference."
