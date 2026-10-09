# Reproducing

## Tier 0 (CPU, minutes)
`scripts/reproduce_figures.sh` regenerates Figs. 1-4 from `results/commag/` and runs
`scripts/verify.py` (exact match on the shipped numbers hash). Requires Python with
numpy, pandas, matplotlib, pyarrow.

## Tier 1 (GPU, < 30 min)
`scripts/run_gates.sh` re-runs gates G1-G11 with Sionna 2.2 on one GPU and compares
them with `results/commag/gates.json` within the stored tolerances.

## Tier 2 (GPU, ~12 min on one H100)
`scripts/run_production.sh` re-runs the production (50 placements, 4 carriers, R1-A
D0/D1, F1/F2), rebuilds the estimands, and `verify.py` compares the result with the
shipped `manuscript_numbers.json` (max abs difference 0 dB). Deterministic seeds.
Measured on one NVIDIA H100 (Sionna 2.2.0): production ~11 m 49 s, full Tier 2
~12 m 05 s. The pipeline uses the fast slow-time synthesis (8 RT anchors per motion),
so it is far below the 1-3 h estimate.

Excluded from this repository: the parametric baseline and its results, internal
docs and review notes, the LaTeX manuscript sources, caches and large raw data.
