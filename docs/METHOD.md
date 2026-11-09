# Rerun v2 — method notes

Code: `src/fr3isac/experiments/rerun_v2.py` (RT + orchestration),
`src/fr3isac/experiments/rerun_v2_core.py` (numpy core), config `configs/rerun_v2.yaml`,
tests `tests/unit/test_rerun_v2.py`. Owner decision of 2026-11-08: implement the study
the ComMag text describes, then rerun.

## Data split
- Eligible placements: 0.5 m grid over the human ROI, kept iff the BS→torso segment is
  unobstructed (90 placements).
- Test placements: the same 50 evenly spaced placements as the earlier run.
- Development placements: the remaining 40. Bank design and tuning use only development
  placements and development motions. `assert_disjoint` runs in every RT stage, and a
  unit test fails on any overlap.

## Motions
- Class: amplitude U[0.10, 0.35] m, rate U[0.5, 1.5] Hz, initial phase U[0, 2π).
- The forearm is at rest until the onset t0. After it,
  d(t) = A[sin(2πf(t−t0)+φ) − sin φ] along x. Position is continuous at the onset;
  φ sets where in the swing cycle the motion starts.
- Onset t0 ~ U[0.10, 0.15) s, i.e. uniform within one 50 ms hop.
- Development and test motions come from separate SeedSequence streams (distinct
  spawn keys), with one child stream per placement.
- K = 10 motions per placement, for both test and development.

## Ray-traced chain
- Target paths are **direct only**: RCSSolver with `los=True`,
  `specular_reflection=False`, `refraction=False`, `max_depth=1`.
- Room clutter (background) uses PathSolver: LoS plus single specular reflections.
- The forearm counts as hidden when the BS→forearm segment intersects a scene object at
  any sample of the record. Hidden onsets are simulated (their returns are near zero)
  and reported separately.
- Fast method: the channel is solved at anchors every 12.5 ms (8 per 100 ms window) from
  the onset to the end of the record. Segment amplitudes are interpolated by magnitude
  and unwrapped phase; the phase follows the kinematic range. Before the onset the arm
  is static, so the record equals the static reference. The pre-onset residual is
  checked.
- Range bin: noise-free static return minus the empty-room reference.
- Static removal: noise-free occupied-room reference with the arm at rest.

## Detection
- Slow time is 1280 Hz. One continuous record runs 0.05–0.35 s (384 samples).
- Windows are 100 ms (128 samples) with a 50 ms hop (64 samples). A window's decision
  time is its last sample, so decisions are causal.
- Each noise trial is one continuous complex AWGN record (σ² per range-bin sample),
  shared by all overlapping windows.
- Success means some window ending in (t0, t0 + 200 ms] crosses the threshold, i.e. the
  first post-onset crossing falls within the deadline. That is always exactly 4 windows.
  Crossings in windows that end before the onset are false alarms; they don't count
  against detection.
- Statistic: max over templates of |⟨t, x_w⟩|² / (‖t‖² σ²).
- Threshold: one per (carrier, bank), set at window-level Pfa = 1e-3 from 10⁷ H0 windows
  and validated on 10⁶ fresh windows. The acceptance rule is fixed in advance: the 95%
  Wilson interval must contain 1e-3.
- Event-level false alarms come from a continuous no-motion stream (20 h per cell). An
  event is a maximal run of consecutive crossing windows.

## Banks
- **Baseline:** the 36-template grid (0.15/0.22/0.28 m × 0.6/0.8/1.0 Hz × 4 phases),
  rebuilt at each carrier.
- **Redesigned:** 64 (A, f, φ) parameter sets, rebuilt at each carrier.
  - Candidates: 20,000 random parameter sets (A ∈ [0, 0.35] m, so the small radial
    excursions seen by the BS are included).
  - Selection: greedy facility location maximising the summed best-template capture
    over visible development onsets at **all four carriers jointly**.
- **Per-carrier variant:** the same selection, run on one carrier at a time.
- Capture: |⟨t, s_w⟩|² / (‖t‖² max_w ‖s_w‖²), taking the best in-deadline window. This is
  the energy fraction relative to a perfectly matched filter on the best window.
- Stated criterion: 90th-percentile capture loss ≤ 1 dB on development onsets at each
  carrier. It is checked and reported, not enforced; if 64 templates cannot meet it, the
  report says so.

## Required EIRP
- Signals are simulated at the nominal EIRP of the regime and scaled linearly.
- 200 noise trials per onset, shared across banks (paired).
- EIRP grid: 1 dB steps from −20 to +12 dB around the baseline-bank noise-free P90.
- Required EIRP = the 0.10 crossing of the mean miss over visible test onsets, linearly
  interpolated.
- The placement bootstrap (2000 resamples) and paired intervals are Stage 2.

## Point targets
- R0 (3GPP human, mean RCS) and R1-P (constant 1 m²) are stationary at the torso. They go
  through the same chain: RT static solve, localization, reference subtraction, then the
  windowed detector with the baseline bank, at every tested EIRP up to eirp0 + 30 dB.
- The miss rates are computed, not assigned.

## Stage 2 additions
- **Static reference.** Noise-free, occupied room, arm at rest, taken from the same
  synthesis as the motion record (OQ-14). The RT static solve is still used to locate
  the range bin.
- **Nominal EIRP.** Conducted power (30 dBm) + simulated transmit gain (F1: 10.21 dBi
  gives 40.21 dBm; F2: 30 dBm + G_F2). Required EIRP = nominal EIRP + the dB offset of
  the 0.10 crossing.
- **Coverage criterion** (fixed 2026-11-09, before the full run). At each carrier, the
  redesigned bank's P90 capture loss on development onsets must be within 0.5 dB of the
  P90 loss of the best template in the 20,000-candidate pool. The run uses 64 templates;
  the smallest greedy size that meets the criterion is reported as well.
- **Constant-term floor.** Energy share of the window mean in the best in-deadline
  window of each visible test onset. A chirp template cannot capture it, so capture
  loss is at least −10·log10(1 − share).
- **Required-EIRP grid.** 1 dB grid, then a 0.25 dB grid over ±1 dB around each bank's
  coarse crossing. The same per-onset noise is regenerated from per-onset seeds.
- **F2.** F2 = F1 + (G_rx,F1 − G_F2), with G_rx,F1 = 20 dBi; under F2 the same array
  transmits and receives. Direct spot check at 12 GHz: simulated gain 2·G_F2, nominal
  EIRP 30 dBm + G_F2, same noise.
- **Bootstrap.** 2000 placement resamples. One resample index set is shared by every
  carrier, bank and regime, so every contrast is paired. 95% percentile intervals.
- **Hidden onsets.** Visibility is decided by geometry. Status per onset:
  `no_target_path` (no forearm path at any anchor) or `partial_or_weak_path`. The
  report gives the noise-free baseline requirement and the miss at the search limit
  (nominal EIRP + 30 dB).
- **Gates.**
  - G1–G9: `validation_gates`.
  - G10: localization of every test placement × carrier × model from the noise-free
    static return (pass if the one-sided 95% lower bound of P(|bin error| ≤ 1) is
    ≥ 0.90).
  - G11: re-simulated records are bit-identical, and a repeated numbers stage gives
    identical output.
