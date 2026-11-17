# Rerun v3: method notes

**Code**
- `src/fr3isac/experiments/rerun_v3.py`: stages and the numbers.
- `src/fr3isac/experiments/rt_v3.py`: v3 scene and ray tracing.
- `src/fr3isac/experiments/rerun_v2_core.py`: the numpy detection core, shared with v2.

**Inputs and outputs**
- Config: `configs/rerun_v3.yaml`. Scene: `scenes/tier1_primary_v3/scene.json`.
- Results: `results/rerun_v3/full/`.

**Design run.** Development data and all banks come from the earlier design run, `results/rerun_v2/full` (`v3.design_run`).
- Its `analysis.json` holds the redesigned and per-carrier bank parameters. They are read, not reselected.
- Its development placements (`rt_summary.npz`, `dev_xy`) are checked to be disjoint from the v3 test placements.

## Scene and ray tracing
- The office has a 6 cm glass partition, modeled as one plane with a 0.06 m slab thickness (`material_thickness_m` in the scene). Materials follow ITU-R P.2040.
- **Target paths** use RCSSolver with `los=True`, `specular_reflection=False` and `refraction=True`, at `max_depth: 3`. This is the minimum depth that returns the refracted forearm path through the glass plane.
  - There are no target-path reflections, diffraction or diffuse scattering.
- **Target size:** every target is an explicit cuboid with a 0.01 m edge. The Sionna default of 0.3 m lets the torso box shadow the forearm.
- **Room clutter** uses PathSolver: LoS plus single specular reflections (`clutter_max_depth: 1`).
- **Onset labels** come from geometry and do not depend on the carrier:
  - **visible:** the forearm leg is unobstructed at every record sample;
  - **glass-only:** blocked only by the partition, so the refracted path exists;
  - **other-blocked.**
  - The label is checked against the ray tracer at every anchor (`rt_manifest.json`, `label_rt_agreement`).
- **Fast method:** the channel is solved at 8 anchors per 100 ms window (every 12.5 ms). Segment amplitudes are interpolated, and the phase follows the kinematic range.
  - It is checked against direct ray tracing at every sample on 2 visible onsets per carrier (`fast_vs_direct_nmse`).

## Data split and motions
- **Test placements:** 50 evenly spaced placements from the eligible points of the 0.25 m-offset grid. A point is eligible when the BS-to-torso segment is unobstructed.
  - These placements are disjoint from the 0.5 m grid used for development.
- **Motions:** amplitude U[0.10, 0.35] m, rate U[0.5, 1.5] Hz, initial phase U[0, 2π).
  - The displacement is d(t) = A[sin(2πf(t−t0)+φ) − sin φ] for t ≥ t0, with the onset t0 ~ U[0.10, 0.15) s.
  - There are 10 motions per test placement.
- **Seed streams:** v3 uses its own streams, `SeedSequence(derive_seed(master_seed, "rerun_v3", name))`, independent of the v2 streams.

## Carriers and antennas
- **Carriers:** 3.5, 7.5, 12, 15 and 28 GHz, each with a 100 MHz bandwidth.
- **F1 (fixed gain):** 10.21 dBi transmit and 20 dBi receive.
- **F2 (idealized aperture-based gain):** one 0.25 m² half-wavelength array transmits and receives.
  - F2 = F1 + (G_rx,F1 − G_F2).
  - This is checked by a direct simulation at 12 GHz (`f2_spot_check`).

## Detection
- **Timing:** slow time runs at 1280 Hz over one continuous 0.05–0.35 s record. Windows are 100 ms (128 samples) with a 50 ms hop.
- **Statistic:** max over templates of |⟨t, x_w⟩|² / (‖t‖² σ²).
- **Reference.** The occupied-room reference is averaged over K frames, with K ∈ {64, 256, 1024} or a noise-free reference (K = ∞).
  - Reference error is one constant r ~ CN(0, σ²/K) per record, with the same unit draw for every K.
  - Thresholds are recalibrated for each K.
- **Thresholds:** one per (carrier, detector, K) at a window-level Pfa of 1e-3.
  - Calibration uses 10⁷ no-motion windows and validation uses 10⁶ fresh ones.
  - A cell is accepted if the 95% Wilson interval contains 1e-3; cells that miss are reported, not recalibrated.
  - Event-level false alarms come from 20 h no-motion streams.
- **Deadlines:** 200 ms (primary) and 100 ms. An onset is detected if a window ending within the deadline crosses the threshold.

## Detectors
- **baseline:** 36 templates (0.15/0.22/0.28 m × 0.6/0.8/1.0 Hz × 4 phases), rebuilt at each carrier.
- **base_plus_dc:** the baseline plus one constant (zero-Doppler) template.
- **redesigned:** 64 parameter sets chosen in the design run by greedy energy capture on development data, rebuilt at each carrier.
- **per_carrier:** the same procedure for one carrier at a time. It exists only at the carriers of the design run (3.5, 12, 15, 28 GHz), so there is none at 7.5 GHz.
- **energy:** a slow-time energy detector at the same per-window Pfa.

## Required EIRP and intervals
- **Signals and trials:** signals are simulated at the regime's nominal EIRP and scaled linearly. There are 200 noise trials per onset, shared by all detectors.
- **Grid:** 1 dB steps, refined to 0.25 dB within ±1 dB of each crossing.
- **Required EIRP:** the 0.10 crossing of the mean miss over visible test onsets.
- **Glass-only onsets** are reported separately with transmission enabled (200 ms, K = ∞).
- **Bootstrap:** 2000 placement resamples with one shared index set, so every contrast is paired. Intervals are 95% percentile intervals.

## Gates
- **G1–G9:** `validation_gates`, written to `gates_rt/gates.json`.
- **G10:** localization of every test placement × carrier × model.
- **G11:** re-simulated records (2 onsets per carrier) are bit-identical, and a repeated numbers stage gives identical output.

## Files
- **`rt_cache.npz`:** the noise-free ray-traced records, about 13 MB. Not in the repository; deposited on Zenodo, and its SHA-256 is in `rt_cache.npz.sha256`.
- **`rt_summary.npz`:** the small arrays of `rt_cache.npz` (labels, placements, gains, nominal EIRP), written by `rerun_v3 summary`. The numbers stage needs only these.
- The release adds a small `summary` stage (`rt_summary.npz`) after tag `rerun-v3-production`, so the CPU tier runs without `rt_cache.npz`; the rebuilt numbers are identical (max difference 0).
