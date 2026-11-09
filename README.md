# FR3 ISAC human-motion sensing

Ray-traced study (Sionna 2.2) of monostatic human-motion sensing at **3.5, 12, 15 and
28 GHz** with a fixed 100 MHz bandwidth. A base station must detect the onset of a
forearm motion within 200 ms. The study compares target models (stationary point targets
vs. a torso with a moving forearm), two antenna regimes (fixed gain; idealized
aperture-based gain) and three template banks (baseline; redesigned once on development
data for all carriers; re-designed per carrier).

This is a controlled simulation in a synthetic room, not a measurement; the articulated
target is a reference model, not ground truth.

## Paper
*Human-Motion Sensing in FR3: How Target, Antenna, and Detector Choices Shape
Cross-Band Comparisons.* Submitted to IEEE Communications Magazine. Citation to be added.

## Definitions
- **F1 (fixed gain):** 10.21 dBi transmit, 20 dBi receive at every carrier.
- **F2 (idealized aperture-based gain):** one 0.25 m^2 half-wavelength array transmits
  and receives; F2 values follow from F1 by the receive-gain difference (checked by a
  direct simulation at 12 GHz).
- **Banks:** baseline = 36-template grid rebuilt per carrier; redesigned = 64 motion
  parameter sets chosen on development placements/motions to cover all four carriers;
  per-carrier = the same procedure per carrier.
- **Required EIRP:** the lowest EIRP at which the mean deadline-miss probability over
  onsets with the forearm in view is <= 0.10 (200 noise trials per onset, continuous
  slow time shared by overlapping 100 ms windows with a 50 ms hop, per-window
  false-alarm probability 1e-3). Intervals: 95% placement bootstrap, 2000 resamples,
  one resample set shared by all cells (paired).

## Results
Test set: 50 placements x 10 random motions. The forearm was hidden by the scene
in 43 of 500 onsets (6 placements); they are excluded from the
required EIRP and reported separately in `results/rerun_v2/full/manuscript_numbers.json`.

### Required EIRP (dBm) [95% interval]
| carrier | F1 baseline | F1 redesigned | F1 per_carrier | F2 baseline | F2 redesigned | F2 per_carrier |
| --- | --- | --- | --- | --- | --- | --- |
| 3.5 GHz | -53.8 [-55.6, -52.5] | -54.1 [-56.0, -52.9] | -54.2 [-56.1, -52.9] | -53.6 [-55.4, -52.3] | -53.9 [-55.8, -52.6] | -54.0 [-55.9, -52.7] |
| 12 GHz | -39.9 [-41.8, -38.4] | -43.0 [-44.9, -41.8] | -43.0 [-44.8, -41.7] | -50.4 [-52.3, -48.9] | -53.5 [-55.3, -52.3] | -53.5 [-55.3, -52.2] |
| 15 GHz | -36.7 [-38.4, -35.3] | -40.6 [-42.2, -39.4] | -40.6 [-42.1, -39.4] | -49.2 [-50.9, -47.8] | -53.0 [-54.6, -51.9] | -53.0 [-54.6, -51.8] |
| 28 GHz | -29.2 [-31.2, -27.6] | -35.1 [-36.9, -33.7] | -35.1 [-36.9, -33.8] | -43.7 [-45.8, -42.2] | -49.6 [-51.5, -48.3] | -49.7 [-51.4, -48.3] |

### Extra EIRP relative to 3.5 GHz (dB) [95% paired interval]
| carrier | F1 baseline | F1 redesigned | F2 baseline | F2 redesigned | lambda^2 term |
| --- | --- | --- | --- | --- | --- |
| 12 GHz | 13.9 [13.1, 14.5] | 11.1 [10.4, 11.7] | 3.2 [2.4, 3.8] | 0.4 [-0.3, 1.0] | 10.7 |
| 15 GHz | 17.0 [16.4, 17.9] | 13.6 [13.0, 14.3] | 4.4 [3.7, 5.2] | 0.9 [0.3, 1.7] | 12.6 |
| 28 GHz | 24.6 [23.7, 25.3] | 19.1 [18.4, 19.6] | 9.8 [8.9, 10.5] | 4.3 [3.6, 4.8] | 18.1 |

Point targets, event-level false alarms, capture loss, the constant-term floor, the
bank coverage criterion and the hidden-onset status are in `manuscript_numbers.json`.
The controls (36-set bank chosen by the redesigned-bank procedure; slow-time energy
detector at the same per-window Pfa; F1, paired against the baseline bank) are in
`results/rerun_v2/control/control.json` (`src/fr3isac/experiments/control_36_energy.py`).

## Installation and reproduction
```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
bash scripts/reproduce_figures.sh          # Tier 0, CPU
docker build -t fr3isac docker/
docker run --gpus all -v "$PWD":/work -w /work fr3isac bash scripts/run_gates.sh       # Tier 1
docker run --gpus all -v "$PWD":/work -w /work fr3isac bash scripts/run_production.sh  # Tier 2
```
See `docs/REPRODUCE.md`, `docs/METHOD.md` and `docs/VALIDATION.md`.

## License
To be decided (see `LICENSE-TBD`).
