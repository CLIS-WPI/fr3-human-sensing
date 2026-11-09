#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
python scripts/verify.py --tier 0
python scripts/make_figure_concept.py --out figures
python scripts/make_figures_tutorial.py --out figures
python scripts/make_figures_results_v7.py --numbers results/rerun_v2/full/manuscript_numbers.json --out figures
echo "Tier 0 OK: numbers rebuilt on CPU and figures regenerated."
