# Validation gates

All gates are re-run by `scripts/run_gates.sh` and compared with
`results/commag/gates.json`.

| Gate | Checks |
| --- | --- |
| G1 | phase and delay vs geometry (4 carriers) |
| G2 | received power vs the radar equation |
| G3 | R^-4 range slope |
| G4 | noise variance and bandwidth scaling |
| G5 | RCS scaling |
| G6 / G6-e2e / G6b / G6c | Doppler (solver, end-to-end, articulated, no-aliasing) |
| G7 | blockage |
| G8 | wall reflection vs the image method |
| G9 | fast method vs per-sample reference |
| G10 | data-driven localization (one-sided 95% coverage) |
| G11 | determinism across two processes |

Tolerances are stored per gate in `results/commag/gates.json`.
