#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
cp results/commag/manuscript_numbers.json results/commag/manuscript_numbers_reference.json
python -m fr3isac.experiments.magazine_production --out results/commag --scenarios 50 --no-gate11
python -m fr3isac.analysis.estimands --out results/commag
python scripts/verify.py --tier 2 --reference results/commag/manuscript_numbers_reference.json
echo "Tier 2 OK: manuscript_numbers.json reproduced within tolerance."
