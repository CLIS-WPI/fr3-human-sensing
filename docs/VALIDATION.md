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

## False-alarm thresholds that missed the acceptance rule

Each threshold is calibrated on one no-motion set and validated on a disjoint one; it is accepted
when the target Pfa lies inside the Wilson 95% interval of the validated rate. The cells below
were not accepted; they are used as calibrated and reported here rather than recalibrated.

4 of 32 threshold cells missed the acceptance rule (target Pfa outside the Wilson 95% interval of the validated rate):

| source | carrier (GHz) | bank / detector | validated Pfa | Wilson 95% CI |
| --- | --- | --- | --- | --- |
| main run | 12 | per_carrier | 0.000921 | [0.000863, 0.000982] |
| control | 12 | 36-set | 0.000912 | [0.000855, 0.000973] |
| review-2 | 12 | baseline_pfa1e-5 | 1.24e-05 | [1.04e-05, 1.48e-05] |
| review-2 | 28 | baseline_pfa1e-5 | 1.45e-05 | [1.23e-05, 1.71e-05] |
