# FR3 ISAC human-motion sensing

Ray-traced study (Sionna RT 2.2) of monostatic human-motion sensing at **3.5, 7.5, 12, 15
and 28 GHz**, each with 100 MHz bandwidth. A base station must detect the onset of a
forearm motion within 200 ms (100 ms as a sensitivity case). The study compares target
models (stationary point targets vs. a torso with a moving forearm), two antenna regimes
(fixed gain; idealized aperture-based gain), five detectors, and the quality of the
static reference (averaged over K = 64, 256 or 1024 frames, or noise-free).

This is a controlled simulation in a synthetic room, not a measurement; the articulated
target is a reference model, not ground truth.

## Paper
*Human-Motion Sensing in FR3: How Target, Antenna, and Detector Choices Shape
Cross-Band Comparisons.* Submitted to IEEE Communications Magazine. Citation to be added.

## Definitions
- **F1 (fixed gain):** 10.21 dBi transmit, 20 dBi receive at every carrier.
- **F2 (idealized aperture-based gain):** one 0.25 m^2 half-wavelength array transmits
  and receives; F2 values follow from F1 by the receive-gain difference (checked by a
  direct simulation at 12 GHz, `f2_spot_check`).
- **Detectors:** `baseline` (36-template grid rebuilt per carrier), `base_plus_dc`
  (baseline + one constant, zero-Doppler template), `redesigned` (64 templates chosen on
  development data), `per_carrier` (the same procedure per carrier; designed at 3.5, 12,
  15 and 28 GHz only), `energy` (slow-time energy detector).
- **K:** frames averaged for the static reference; `inf` = noise-free reference.
  Thresholds are recalibrated per carrier, detector and K at a per-window false-alarm
  probability of 1e-3.
- **Required EIRP:** the lowest EIRP at which the mean deadline-miss probability over
  onsets with the forearm in view is <= 0.10 (200 noise trials per onset). Intervals: 95%
  placement bootstrap, 2000 resamples, one resample set shared by all cells (paired).

## Results (`results/rerun_v3/full/manuscript_numbers.json`)
Test set: 50 placements x 10 motions = 500 onsets: 464 with the forearm in
view, 36 seen only through the glass partition, 0 blocked otherwise.
Tables below: 200 ms deadline, noise-free reference (K = inf) unless stated.

Field names in `manuscript_numbers.json` (`numbers`): `required_eirp_dbm`
(regime, detector, K, deadline_s, carrier_ghz, required_eirp_dbm, ci95_dbm),
`savings_vs_baseline_db` and `extra_eirp_vs_3p5_db` (value_db, ci95_db, paired),
`glass_only_with_transmission`, `false_alarm_calibration`, `event_false_alarms`,
`pfa_with_reference_error_at_perfect_reference_thresholds`, `test_capture_loss_db`,
`constant_term`, `point_targets`, `onsets`, `placements`, `localization`,
`fast_vs_direct_nmse`, `antennas`, `bootstrap`, `rt_cache_sha256`.

### Required EIRP (dBm) [95% interval]
| carrier | F1 baseline | F1 redesigned | F2 baseline | F2 redesigned |
| --- | --- | --- | --- | --- |
| 3.5 GHz | -54.2 [-56.0, -51.8] | -54.5 [-56.3, -52.1] | -54.0 [-55.8, -51.6] | -54.3 [-56.1, -51.9] |
| 7.5 GHz | -45.9 [-47.2, -44.8] | -47.3 [-48.8, -46.3] | -52.3 [-53.6, -51.3] | -53.7 [-55.2, -52.7] |
| 12 GHz | -40.0 [-41.6, -38.8] | -42.9 [-44.4, -41.7] | -50.5 [-52.1, -49.3] | -53.4 [-54.9, -52.2] |
| 15 GHz | -38.1 [-39.4, -37.1] | -41.6 [-42.9, -40.5] | -50.5 [-51.8, -49.5] | -54.0 [-55.3, -52.9] |
| 28 GHz | -29.9 [-31.2, -28.9] | -35.6 [-36.9, -34.6] | -44.5 [-45.8, -43.5] | -50.2 [-51.4, -49.2] |

### Extra EIRP relative to 3.5 GHz (dB) [95% paired interval]
| carrier | F1 baseline | F1 redesigned | F2 baseline | F2 redesigned | wavelength-squared term |
| --- | --- | --- | --- | --- | --- |
| 7.5 GHz | 8.3 [6.0, 9.6] | 7.2 [4.8, 8.5] | 1.7 [-0.6, 3.0] | 0.6 [-1.8, 1.9] | 6.6 |
| 12 GHz | 14.2 [11.8, 15.5] | 11.6 [9.3, 12.9] | 3.5 [1.1, 4.8] | 0.9 [-1.4, 2.2] | 10.7 |
| 15 GHz | 16.1 [13.8, 17.1] | 12.9 [10.7, 13.8] | 3.5 [1.2, 4.5] | 0.3 [-2.0, 1.2] | 12.6 |
| 28 GHz | 24.4 [22.0, 25.6] | 18.9 [16.6, 20.0] | 9.6 [7.2, 10.8] | 4.1 [1.8, 5.2] | 18.1 |

### EIRP saved relative to the baseline bank (dB) [95% paired interval], F1
| carrier | redesigned, K=inf | redesigned, K=256 | base_plus_dc, K=inf | energy, K=inf |
| --- | --- | --- | --- | --- |
| 3.5 GHz | 0.3 [0.2, 0.4] | -0.1 [-0.2, 0.0] | 0.1 [0.1, 0.2] | -4.9 [-5.0, -4.6] |
| 7.5 GHz | 1.4 [1.4, 1.6] | 0.8 [0.7, 0.9] | 1.1 [1.0, 1.2] | -3.0 [-3.2, -2.8] |
| 12 GHz | 2.8 [2.6, 3.1] | 2.2 [1.9, 2.4] | 2.3 [2.1, 2.6] | -1.2 [-1.5, -0.9] |
| 15 GHz | 3.5 [3.3, 3.7] | 2.8 [2.6, 3.0] | 2.9 [2.7, 3.3] | -0.4 [-0.7, -0.0] |
| 28 GHz | 5.8 [5.4, 6.1] | 5.1 [4.8, 5.4] | 5.4 [5.0, 5.7] | 2.3 [1.9, 2.6] |

### Forearm seen only through the glass partition
| carrier | glass-only onsets (placements) | extra EIRP vs visible onsets, redesigned bank (dB) |
| --- | --- | --- |
| 3.5 GHz | 36 (5) | 8.4 [1.1, 11.2] |
| 7.5 GHz | 36 (5) | 10.4 [1.4, 12.7] |
| 12 GHz | 36 (5) | 11.0 [0.8, 13.4] |
| 15 GHz | 36 (5) | 15.1 [2.4, 17.4] |
| 28 GHz | 36 (5) | 29.8 [26.5, 31.9] |

## Reproducing the figures
| Figure | Script | Inputs |
| --- | --- | --- |
| Fig. 1 (concept, schematic, no data) | `scripts/make_figure_concept.py --out figures` | none |
| Fig. 2 (Doppler scaling, template capture) | `scripts/make_figures_v3.py` (function `fig2`) | `configs/rerun_v3.yaml`, v3 scene, v3 test motions |
| Fig. 3 (required EIRP) | `scripts/make_figures_v3.py` (function `fig3`) | `manuscript_numbers.json` |
| Fig. 4 (reference quality, glass) | `scripts/make_figures_v3.py` (function `fig4`) | `manuscript_numbers.json` |

`bash scripts/reproduce_figures.sh` (CPU) rebuilds `manuscript_numbers.json` from the
shipped per-onset miss curves, checks it against the shipped file, and regenerates all
four figures into `figures/`.

## Installation and reproduction
```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
bash scripts/reproduce_figures.sh          # Tier 0, CPU
docker build -t fr3isac docker/
IMG=fr3isac; DIGEST=$(docker image inspect --format '{{.Id}}' $IMG)
docker run --gpus all -e FR3ISAC_CONTAINER_IMAGE=$IMG -e FR3ISAC_CONTAINER_DIGEST=$DIGEST   -v "$PWD":/work -w /work $IMG bash scripts/run_gates.sh       # Tier 1
docker run --gpus all -e FR3ISAC_CONTAINER_IMAGE=$IMG -e FR3ISAC_CONTAINER_DIGEST=$DIGEST   -v "$PWD":/work -w /work $IMG bash scripts/run_production.sh  # Tier 2
```
The run manifests record the image name and digest passed this way, and the release
version from `RELEASE_VERSION` (release tag and source commit) when the tree is not a
git checkout (`provenance_note: "release copy"`).
See `docs/REPRODUCE.md`, `docs/METHOD.md` and `docs/VALIDATION.md`.

## Ray-traced records (`rt_cache.npz`)
The noise-free ray-traced records of the production run (`rt_cache.npz`, about 13 MB)
are not in this repository. They will be deposited on Zenodo.

> **Zenodo DOI: DOI to be added** (placeholder; the deposit is not yet published).

Once available, place the file in `results/rerun_v3/full/` and check it:
```
cd results/rerun_v3/full && sha256sum -c rt_cache.npz.sha256
```
SHA-256: `ba605b65d4a78437b971e9add33dafa47cd5b41aa621d73ba5ca24e8625634ee`. Tier 0 does not need this file: the small arrays it uses are shipped in
`rt_summary.npz`. With it, the `analyze` and `numbers` stages of
`python -m fr3isac.experiments.rerun_v3` run without ray tracing; Tier 2 regenerates it.

## License
MIT (see `LICENSE`).
