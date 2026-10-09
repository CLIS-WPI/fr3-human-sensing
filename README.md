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
Title: *When Human Motion Models Change FR3 Sensing Decisions: Lessons from a
Ray-Traced Cross-Band ISAC Study.* Submitted to IEEE Communications Magazine.
Citation to be added.

## Definitions
- **F1, fixed antenna gain:** the same antenna gain at every carrier, so the
  aperture grows at lower carriers.
- **F2, fixed aperture:** the same physical aperture at every carrier, so the array
  gain grows with carrier.
- **R0 / R1-P / R1-A:** 3GPP-style single-point human / aggregate-RCS-matched point
  target / aggregate-RCS-matched articulated target (torso plus one moving forearm).
- **D0 / D1:** fixed 36-template bank / carrier-adapted 64-template bank.
- **Required EIRP:** the transmit power at which the miss rate of transient arm
  motion reaches 0.10. The quoted value is the 90th percentile of per-onset required
  EIRP over the placements whose forearm is in view (95% cluster-bootstrap interval
  over placements).

## Results

### Required EIRP (dBm), 90th percentile over onsets with the forearm in view
| carrier | F1 D0 | F1 D1 | F2 D0 | F2 D1 |
| --- | --- | --- | --- | --- |
| 3.5 GHz | -52.5 | -53.3 | -52.2 | -53.1 |
| 12 GHz | -38.5 | -42.8 | -49.0 | -53.3 |
| 15 GHz | -35.5 | -40.3 | -47.9 | -52.8 |
| 28 GHz | -27.0 | -34.4 | -41.5 | -49.0 |

Extra EIRP relative to 3.5 GHz (dB), for the four design choices:
| carrier | F1 D0 | F1 D1 | F2 D0 | F2 D1 |
| --- | --- | --- | --- | --- |
| 12 GHz | 13.9 | 10.5 | 3.2 | -0.2 |
| 15 GHz | 16.9 | 13.0 | 4.3 | 0.3 |
| 28 GHz | 25.5 | 18.9 | 10.7 | 4.1 |

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
