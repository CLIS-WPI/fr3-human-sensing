# Validation gates

`results/rerun_v2/full/gates.json` holds the measured value and tolerance of each gate.

| Gate | Checks |
| --- | --- |
| G1 | phase and delay vs geometry (4 carriers) |
| G2 | received power vs the radar equation |
| G3 | R^-4 range slope |
| G4 | noise variance and bandwidth scaling |
| G5 | RCS scaling |
| G6 / G6-e2e / G6b / G6c | Doppler (solver, end-to-end, articulated, no aliasing) |
| G7 | blockage |
| G8 | wall reflection vs the image method |
| G9 | fast method vs per-sample reference |
| G10 | localization from the noise-free static return, every test placement x carrier x model |
| G11 | determinism: re-simulated records bit-identical; repeated numbers stage identical |

The production run also records direct-RT checks of the fast method on its own onsets
(`rt_manifest.json`, `fast_vs_direct_nmse`).
