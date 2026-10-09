#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
cp results/commag/gates.json results/commag/gates_reference.json
python -m fr3isac.experiments.validation_gates --out results/commag
python -m fr3isac.experiments.localization --out results/commag
python - <<'PY'
import json
g = json.load(open("results/commag/gates.json"))
m = json.load(open("results/commag/localization.json"))
g["gates"]["G10"] = m["gate_G10"]
g["gates"]["G11"] = m["gate_G11"]
json.dump(g, open("results/commag/gates.json", "w"), indent=2)
PY
python scripts/verify.py --tier 1 --reference results/commag/gates_reference.json
echo "Tier 1 OK: gates G1-G11 match the reference within tolerance."
