# Reproducing

## Tier 0 (CPU)
`scripts/reproduce_figures.sh` rebuilds `manuscript_numbers.json` from the shipped
per-onset miss curves (`miss_curves.npz`), `analysis.json` and `rt_summary.npz` (bootstrap
included), requires a match with the shipped file within 1e-9, then regenerates Figs. 1-4.

## Tier 1 (GPU)
`scripts/run_gates.sh` re-runs G1-G9 with Sionna 2.2 and re-evaluates G10/G11 from the
shipped production manifest; all gates must pass and G1-G9 must match the reference
within 1e-6.

## Tier 2 (GPU)
`scripts/run_production.sh` re-runs the production (ray tracing, analysis, numbers) from
`configs/rerun_v3.yaml` and compares every required EIRP with the shipped values
(tolerance 0.05 dB; the pipeline is deterministic). Development placements and the frozen
banks come from the design run (`results/rerun_v2/full/analysis.json` and
`rt_summary.npz`).

The noise-free ray-traced records (`rt_cache.npz`) will be deposited on Zenodo (DOI to be added); see the
README for the SHA-256 check. Earlier pipelines, internal notes and the LaTeX sources are
not part of this repository.
