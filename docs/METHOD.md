# Method

This is a controlled synthetic study of monostatic (base-station) sensing in a
10 x 8 x 3.2 m indoor room. It does not use measurements; the room is a
representative simulation scene, not a validated digital twin.

## Sensing chain
For each placement and each slow-time frame the target return is obtained with the
Sionna 2.2 `RCSSolver`; background clutter comes from a `PathSolver` pass over the
scene. The detector input is the passband frequency-domain channel
`H[k, n] = sum_p a_p(n) exp(-j 2 pi (f_c + df_k) tau_p(n))`, built from the raw path
coefficients so that the carrier phase is applied exactly once and no library
delay or power normalization is introduced. Thermal noise is added per subcarrier
with variance `k T B_sc NF`.

The slow-time dimension (1280 Hz, 128 samples per 100 ms CPI) is synthesized by a
fast method: the per-segment amplitudes come from a few RT anchor solves and the
phase evolution is the exact geometric phase from the target kinematics. Complex
coefficients are never interpolated.

## Targets, detectors and regimes
Targets: R0 (3GPP-style single-point human), R1-P (aggregate-RCS-matched point target)
and R1-A (aggregate-RCS-matched articulated target: one torso plus one moving
forearm). Detectors: D0 (fixed 36-template chirp bank) and D1 (carrier-adapted
64-template bank chosen on a development split). Detection thresholds are
calibrated on noise-only windows at a target false-alarm probability of 1e-3.

Two antenna regimes are compared. **F1 (fixed antenna gain)** holds the antenna
gain fixed across carriers, so the aperture grows at lower carriers. **F2 (fixed
aperture)** holds the physical aperture fixed, so the array gain grows with
carrier; under F2 the same array transmits and receives, so the receive gain
follows the aperture. Because the required EIRP depends only on the receive gain,
the F2 numbers are obtained from F1 by the per-carrier receive-gain difference.

## Estimand
The primary quantity is the required EIRP (dBm) at which the miss rate of transient
arm motion reaches 0.10. It is the 90th percentile of per-onset required EIRP
pooled over the placements whose forearm is in view (forearm visibility is decided
from geometry alone, never from detector output), with a 95% cluster-bootstrap
interval over placements. The unconditional 90th percentile (all onsets) and the
8% occlusion floor are also reported, together with the per-placement median.
