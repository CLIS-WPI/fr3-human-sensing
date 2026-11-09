#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
O=results/rerun_v2/full
cp $O/manuscript_numbers.json $O/manuscript_numbers_reference.json
cp results/rerun_v2/control/control.json results/rerun_v2/control/control_reference.json
python -m fr3isac.experiments.rerun_v2 rt --out $O
python -m fr3isac.experiments.rerun_v2 analyze --out $O
python -m fr3isac.experiments.rerun_v2 numbers --out $O
python -m fr3isac.experiments.control_36_energy --run $O --out results/rerun_v2/control
python scripts/verify.py --tier 2 --reference $O/manuscript_numbers_reference.json
echo "Tier 2 OK: required EIRP reproduced."
