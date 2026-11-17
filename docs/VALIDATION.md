# Validation gates

`results/rerun_v3/full/gates.json` holds the measured value and tolerance of each gate.

| Gate | Checks |
| --- | --- |
| G1 | phase and delay vs geometry |
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

The production run also records, in `rt_manifest.json`: direct-RT checks of the fast method
on its own onsets (`fast_vs_direct_nmse`) and agreement of the geometric onset labels with
the ray tracer (`label_rt_agreement`).

## False-alarm thresholds that missed the acceptance rule

Each threshold is calibrated on one no-motion set and validated on a disjoint one; it is accepted
when the target Pfa lies inside the Wilson 95% interval of the validated rate. The cells below
were not accepted; they are used as calibrated and reported here rather than recalibrated.

4 of 96 threshold cells (carrier x detector x K) missed the acceptance rule (target Pfa outside the Wilson 95% interval of the validated rate):

| carrier (GHz) | detector | K | validated Pfa | Wilson 95% CI |
| --- | --- | --- | --- | --- |
| 12 | per_carrier | 256 | 0.00107 | [0.00101, 0.00114] |
| 12 | per_carrier | inf | 0.000928 | [0.00087, 0.00099] |
| 12 | energy | inf | 0.000932 | [0.000874, 0.000994] |
| 15 | energy | 64 | 0.000924 | [0.000866, 0.000985] |
