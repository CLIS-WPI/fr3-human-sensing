#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
# 1) recompute every estimand on CPU and compare with the shipped numbers;
python scripts/verify.py --tier 0
# 2) only then regenerate the figures;
python scripts/make_figure_concept.py --out figures
python scripts/make_figures_tutorial.py --out figures
python scripts/make_figures_results_v7.py --numbers results/commag/manuscript_numbers.json --out figures
# 3) extra checks (figures present + shipped-numbers hash)
python scripts/verify.py --tier 0 --figures
echo "Tier 0 OK: estimands reproduced on CPU and figures regenerated."
