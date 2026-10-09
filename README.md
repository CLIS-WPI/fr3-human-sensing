# FR3 ISAC human-motion sensing

Human-target modelling choices can decide whether upper-mid-band (FR3) sensing is
affordable. This repository accompanies a ray-traced study (Sionna 2.2) of cross-band
integrated sensing and communication at **3.5, 12, 15 and 28 GHz** under a fixed
100 MHz bandwidth, two antenna regimes and two limb-motion detectors. The headline
finding is that the **human model, the antenna regime and the detector - not the
carrier - set the FR3 cost**: with a fixed physical aperture and a carrier-adapted
detector, **12 GHz and 15 GHz need the same transmit power as 3.5 GHz** to detect
transient arm motion (within 0.3 dB), while a fixed-gain regime or a single-point
human model makes FR3 look far more expensive.

## Paper
Title: *Human-Motion Sensing in FR3: How Target, Antenna, and Detector Choices Shape
Cross-Band Comparisons.* Submitted to IEEE Communications Magazine.
Citation to be added.

## Definitions
- **F1, fixed antenna gain:** the same antenna gain at every carrier, so the
  aperture grows at lower carriers.
- **F2, fixed aperture:** the same physical aperture at every carrier, so the array
  gain grows with carrier.
- **R0 / R1-P / R1-A:** 3GPP-style single-point human / aggregate-RCS-matched point
  target / aggregate-RCS-matched articulated target (torso plus one moving forearm).
- **D0 / Dfix / D1:** baseline 36-template bank / shared redesigned 64-template bank
  (designed once at 12 GHz, applied unchanged at every carrier) / per-carrier
  redesigned 64-template bank.
- **Required EIRP:** the transmit power at which the mean visible-onset deadline-miss
  rate reaches 0.10 (200 noise realisations per onset; 95% bootstrap interval over
  placements, 2000 resamples). Values are quoted in dBm; the F2 rows use the
  receive-gain shift.

## Results

### Required EIRP (dBm), operational mean-miss <= 0.10 over onsets with the forearm in view
| carrier | F1 D0 | F1 Dfix | F1 D1 | F2 D0 | F2 Dfix | F2 D1 |
| --- | --- | --- | --- | --- | --- | --- |
| 3.5 GHz | -53.3 | -53.1 | -53.0 | -53.1 | -52.9 | -52.8 |
| 12 GHz | -39.2 | -42.5 | -42.5 | -49.7 | -53.0 | -53.0 |
| 15 GHz | -35.9 | -40.2 | -40.3 | -48.3 | -52.7 | -52.7 |
| 28 GHz | -27.5 | -35.2 | -34.5 | -42.1 | -49.8 | -49.1 |

Extra EIRP relative to 3.5 GHz (dB), for the four design choices (paired over placements):
| carrier | F1 D0 | F1 Dfix | F2 D0 | F2 Dfix |
| --- | --- | --- | --- | --- |
| 12 GHz | 14.1 | 10.6 | 3.4 | -0.1 |
| 15 GHz | 17.4 | 12.9 | 4.7 | 0.3 |
| 28 GHz | 25.7 | 17.9 | 11.0 | 3.1 |

8% of onsets have the forearm occluded by the scene (identical at every carrier);
they are excluded from the visible estimate above and reported separately in
`results/commag/manuscript_numbers.json`. R0 and R1-P (single-point models) never
detect the transient limb motion at any tested power.

## Installation
CPU-only (Tier 0):
```
python -m venv .venv && . .venv/bin/activate
pip install numpy pandas matplotlib pyarrow pydantic pyyaml
bash scripts/reproduce_figures.sh
```
Docker (GPU, for Tiers 1-2; image builds from `docker/`):
```
docker build -t fr3isac docker/
# Tier 1: re-run the validation gates G1-G11
docker run --gpus all -v "$PWD":/work -w /work fr3isac bash scripts/run_gates.sh
# Tier 2: re-run the production and regenerate the numbers
docker run --gpus all -v "$PWD":/work -w /work fr3isac bash scripts/run_production.sh
# Tier 0 (figures and numbers only) also runs in Docker:
docker run -v "$PWD":/work -w /work fr3isac bash scripts/reproduce_figures.sh
```

## Reproduction tiers
| Tier | Hardware | Command | Time |
| --- | --- | --- | --- |
| 0 | CPU | `scripts/reproduce_figures.sh` | < 1 min |
| 1 | GPU | `scripts/run_gates.sh` | ~2.5 min |
| 2 | GPU (H100) | `scripts/run_production.sh` | ~12 min |

See `docs/REPRODUCE.md` for hardware, runtimes and tolerances, `docs/METHOD.md`
for the sensing chain, and `docs/VALIDATION.md` for the validation gates.

## Layout
`src/fr3isac/` (pipeline), `results/commag/` (numbers and per-onset data),
`figures/` (Figs. 1-4), `scripts/`, `tests/`.

## License
To be decided (see `LICENSE-TBD`).
