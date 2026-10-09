"""Production run: required-EIRP estimands with cluster-bootstrap intervals.

Runs 50 eligible placements (line-of-sight rule applied at generation), four
carriers, targets R0/R1-P/R1-A, detectors D0/D1, the F1 and F2 antenna regimes, an
additive-interference sweep with an explicit interference-off baseline, and
RCS-scaling exponents {-1,0,+1}. Writes ``results/commag/manuscript_numbers.json``
with estimands and interval types, and persists per-onset statistics.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC
from pathlib import Path

import numpy as np

from fr3isac.experiments import localization as loc
from fr3isac.sensing.rt_channel import detector_input_scale, thermal_noise_variance
from fr3isac.sensing.rt_localize import localize_body_bin
from fr3isac.sensing.rt_slowtime import reconstruction_nmse
from fr3isac.targets.builders import analytic_arm_trajectory

CARRIERS = loc.CARRIERS
REGIMES = loc.REGIMES
MODELS = loc.MODELS
NSC, SCS, NSLOW = loc.NSC, loc.SCS, loc.NSLOW
OCC_BW = loc.OCC_BW
PFA = loc.PFA
P_TX_W = loc.P_TX_W
FC_REF = 3.5e9  # RCS-scaling anchor carrier
RCS_EXPONENTS = (-1, 0, 1)
INR_LEVELS_DB = (0.0, 5.0, 10.0, 20.0, 30.0)
N_BOOT = 2000
VALIDATION_FRAC = 0.05
TARGET_MISS = 0.10


def select_scenarios(n=50, step_m=0.5):
    elig, excluded = loc.eligible_scenarios(step_m=step_m)
    idx = np.linspace(0, len(elig) - 1, min(n, len(elig))).round().astype(int)
    return [elig[i] for i in idx], excluded


def _boot_ci(per_scenario, stat_fn, n_boot=N_BOOT, seed=0, alpha=0.05):
    vals = np.asarray(per_scenario, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([stat_fn(vals[rng.integers(0, vals.size, vals.size)])
                      for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"point": float(stat_fn(vals)), "ci95_low": float(lo), "ci95_high": float(hi),
            "interval_type": "cluster_bootstrap_over_scenarios", "n_scenarios": int(vals.size)}


def _null_thresholds(bank, sigma2_body, rng, n_null):
    stats = np.empty(n_null)
    for i in range(n_null):
        x = (rng.standard_normal(NSLOW) + 1j * rng.standard_normal(NSLOW)) * np.sqrt(sigma2_body / 2)
        stats[i] = loc._statistic(x, bank, sigma2_body)
    return float(np.quantile(stats, 1.0 - PFA))


def run_m4(n_scen=50, n_null=5000, seed=0):
    offsets = (np.arange(NSC) - NSC // 2) * SCS
    sigma2 = thermal_noise_variance(SCS, noise_figure_db=loc.NOISE_FIGURE_DB)
    sigma2_body = sigma2 / NSC
    c2 = detector_input_scale(P_TX_W, NSC)
    scenarios, excluded = select_scenarios(n_scen)
    rng = np.random.default_rng(seed)

    cells = []          # per (fc, regime, model, detector, scenario)
    onsets = []         # per (scenario, carrier, regime, detector, onset)
    thresholds_rows = []
    val_nmse = []
    rng_val = np.random.default_rng(seed + 1)

    for fc in CARRIERS:
        bg = loc._bg_cfr(fc, offsets)
        d0 = loc._bank(fc, loc.D0_AMPLITUDES, loc.D0_OMEGAS, loc.D0_PHASES)
        cand = loc._bank(fc, loc.D1_AMPLITUDES, loc.D1_OMEGAS, loc.D1_PHASES)

        for regime in REGIMES:
            gain = loc._regime_gain_db(fc, regime)
            eirp0 = loc._regime_eirp_dbm(fc, regime)
            tx0 = loc.FairnessConfig(regime=regime).conducted_power_dbm

            dev_scores = np.zeros(cand.shape[0])
            for xy in scenarios:
                root = (float(xy[0]), float(xy[1]), loc.TARGET_HEIGHT_M)
                static = loc._solve_cfr(fc, "R1-A", loc._static_sample(root), offsets, gain)
                r_hat, _ = localize_body_bin(bg[:, None] + static[:, None], bg[:, None], OCC_BW, 1.0, 12.0)
                x_ref = c2 * loc._body_vector(bg, static, r_hat)
                for motion in loc.DEV_MOTIONS:
                    x = c2 * loc._body_vector(bg, loc._r1a_fast(fc, root, motion, offsets, gain), r_hat)
                    dev_scores += loc._scores(x - x_ref, cand, sigma2_body).ravel()
            d1 = cand[np.argsort(dev_scores)[-loc.D1_TOP_K:]]
            banks = {"D0": d0, "D1": d1}
            thr = {name: _null_thresholds(b, sigma2_body, rng, n_null) for name, b in banks.items()}
            for name, t in thr.items():
                thresholds_rows.append({"carrier_hz": fc, "regime": regime,
                                        "detector": name, "threshold": t})

            for sid, xy in enumerate(scenarios):
                root = (float(xy[0]), float(xy[1]), loc.TARGET_HEIGHT_M)
                for model in MODELS:
                    static = loc._solve_cfr(fc, model, loc._static_sample(root), offsets, gain)
                    r_hat, _ = localize_body_bin(bg[:, None] + static[:, None], bg[:, None], OCC_BW, 1.0, 12.0)
                    x_ref = c2 * loc._body_vector(bg, static, r_hat)
                    if model != "R1-A":
                        for name, t in thr.items():
                            cells.append({"fc_hz": fc, "regime": regime, "model": model,
                                          "detector": name, "scenario": list(root),
                                          "required_eirp_dbm": None, "miss": 1.0, "threshold": t})
                        continue
                    test_frames = [loc._r1a_fast(fc, root, mm, offsets, gain) for mm in loc.TEST_MOTIONS]
                    svs = [c2 * loc._body_vector(bg, fr, r_hat) - x_ref for fr in test_frames]
                    for name, bank in banks.items():
                        t = thr[name]
                        stats = np.asarray([loc._statistic(sv, bank, sigma2_body) for sv in svs])
                        s10 = float(np.quantile(stats, 0.10))
                        req = eirp0 + 10 * np.log10(t / s10) if s10 > 0 else float("inf")
                        inr_max = 10 * np.log10(s10 / t - 1.0) if s10 > t else None
                        cells.append({"fc_hz": fc, "regime": regime, "model": model,
                                      "detector": name, "scenario": list(root),
                                      "required_eirp_dbm": req, "required_tx_dbm": tx0 + 10 * np.log10(t / s10),
                                      "miss": float(np.mean(stats < t)), "threshold": t,
                                      "s10": s10, "max_tolerable_inr_db": inr_max})
                        for onset_id, (sv, st) in enumerate(zip(svs, stats, strict=False)):
                            cap = loc._capture(sv, bank)
                            onsets.append({
                                "scenario_id": sid, "scenario": list(root),
                                "carrier_hz": fc, "regime": regime, "detector": name,
                                "model": model, "onset_id": onset_id,
                                "statistic": float(st), "threshold": t,
                                "capture_loss_db": 10 * np.log10(cap + 1e-30),
                                "required_eirp_dbm": (eirp0 + 10 * np.log10(t / st)) if st > 0 else None,
                            })
                    # 5% direct-RT validation of the fast method (articulated)
                    if rng_val.random() < VALIDATION_FRAC:
                        times = np.arange(NSLOW) * (loc.CPI_S / NSLOW)
                        fast_cols, direct_cols = [], []
                        for mm in loc.TEST_MOTIONS:
                            samples = analytic_arm_trajectory(root, times, amplitude_m=mm[0],
                                                              omega_rad_s=2 * np.pi * mm[1])
                            direct_cols.append(np.stack(
                                [loc._solve_cfr(fc, "R1-A", s, offsets, gain) for s in samples], axis=1))
                            fast_cols.append(loc._r1a_fast(fc, root, mm, offsets, gain))
                        val_nmse.append({"fc_hz": fc, "regime": regime, "scenario": list(root),
                                         "nmse": reconstruction_nmse(np.stack(fast_cols, axis=1),
                                                                     np.stack(direct_cols, axis=1))})
    return {"cells": cells, "onsets": onsets, "thresholds": thresholds_rows,
            "validation": val_nmse, "excluded_count": len(excluded),
            "n_eligible": len(scenarios)}


def _est(cells, fc, model, detector, key="required_eirp_dbm"):
    vals = [c[key] for c in cells if c["fc_hz"] == fc and c["model"] == model
            and c["detector"] == detector and c[key] is not None and c["regime"] == "F1"]
    return vals


def build_numbers(res):
    cells = res["cells"]
    numbers = {
        "provenance": {
            "estimand_units": {
                "required_eirp_dbm": "dBm",
                "required_tx_dbm": "dBm",
                "d1_minus_d0_eirp_db": "dB (difference of two dBm values)",
                "rcs_exponent_required_eirp_dbm": "dBm",
                "max_tolerable_inr_db": "dB",
            },
            "note": "D1-D0 differences are in dB, not dBm.",
            "primary_miss_target": TARGET_MISS, "pfa": PFA,
            "interval_type_default": "cluster_bootstrap_over_scenarios"},
        "required_eirp_dbm": [],
        "d1_minus_d0_eirp_db": [],
        "rcs_exponent_required_eirp_dbm": [],
        "inr": [],
        "validation": {},
    }
    for fc in CARRIERS:
        for model in MODELS:
            for detector in ("D0", "D1"):
                vals = _est(cells, fc, model, detector)
                if vals:
                    numbers["required_eirp_dbm"].append({"fc_hz": fc, "model": model,
                        "detector": detector, "regime": "F1", **_boot_ci(vals, np.median)})
        # D1 - D0 per scenario
        d0s = {(tuple(c["scenario"])): c["required_eirp_dbm"] for c in cells
               if c["fc_hz"] == fc and c["model"] == "R1-A" and c["detector"] == "D0"
               and c["regime"] == "F1" and c["required_eirp_dbm"] is not None}
        d1s = {(tuple(c["scenario"])): c["required_eirp_dbm"] for c in cells
               if c["fc_hz"] == fc and c["model"] == "R1-A" and c["detector"] == "D1"
               and c["regime"] == "F1" and c["required_eirp_dbm"] is not None}
        diffs = [d1s[k] - d0s[k] for k in d0s if k in d1s]
        if diffs:
            numbers["d1_minus_d0_eirp_db"].append({"fc_hz": fc, "model": "R1-A",
                                                   "regime": "F1", **_boot_ci(diffs, np.median)})
        # RCS exponent anchor (post-hoc on the R1-A median)
        base = _est(cells, fc, "R1-A", "D0")
        if base:
            base_med = float(np.median(base))
            shift = 10 * np.log10(fc / FC_REF)
            for exp in RCS_EXPONENTS:
                numbers["rcs_exponent_required_eirp_dbm"].append({
                    "fc_hz": fc, "model": "R1-A", "detector": "D0", "regime": "F1",
                    "rcs_exponent": exp, "required_eirp_dbm": base_med - exp * shift,
                    "interval_type": "point_estimate_shifted_from_baseline"})
    # INR sweep (R1-A, D0/D1, F1)
    for fc in CARRIERS:
        for detector in ("D0", "D1"):
            inrmax = [c["max_tolerable_inr_db"] for c in cells
                      if c["fc_hz"] == fc and c["model"] == "R1-A" and c["detector"] == detector
                      and c["regime"] == "F1" and c["max_tolerable_inr_db"] is not None]
            row = {"fc_hz": fc, "detector": detector, "interference_off": True,
                   "max_tolerable_inr_db": (float(np.median(inrmax)) if inrmax else None),
                   "n": len(inrmax)}
            eirp_at = {}
            base = _est(cells, fc, "R1-A", detector)
            for inr in INR_LEVELS_DB:
                eirp_at[f"inr_{inr:g}db"] = (float(np.median(base)) + 10 * np.log10(1 + 10 ** (inr / 10)))
            row["required_eirp_dbm_by_inr"] = eirp_at
            numbers["inr"].append(row)
    # validation
    if res["validation"]:
        nm = [v["nmse"] for v in res["validation"]]
        numbers["validation"] = {"method": "articulated_fast_vs_direct_rt",
                                 "frac_cells": VALIDATION_FRAC, "n": len(nm),
                                 "mean_nmse": float(np.mean(nm)), "max_nmse": float(np.max(nm)),
                                 "pass": bool(np.max(nm) <= 0.01)}
    return numbers


def main(argv=None):
    import os
    import subprocess
    import time
    from datetime import datetime

    import pandas as pd

    parser = argparse.ArgumentParser(description="production run")
    parser.add_argument("--out", default="results/commag")
    parser.add_argument("--scenarios", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-gate11", action="store_true")
    args = parser.parse_args(argv)
    n = 4 if args.quick else args.scenarios
    t0 = datetime.now(UTC)
    wall0 = time.time()
    res = run_m4(n_scen=n)
    numbers = build_numbers(res)
    wall1 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Persist per-onset data, per-cell thresholds, and a run manifest.
    pd.DataFrame(res["onsets"]).to_parquet(out / "onsets.parquet", index=False)
    pd.DataFrame(res["thresholds"]).to_parquet(out / "thresholds.parquet", index=False)
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        git_sha = "unknown"
    manifest = {
        "git_sha": git_sha,
        "container_image": os.environ.get("FR3_CONTAINER_IMAGE", "unknown"),
        "container_id": os.environ.get("FR3_CONTAINER_ID", "unknown"),
        "gpu": os.environ.get("FR3_GPU_UUID", "unknown"),
        "start_time": t0.isoformat(), "end_time": datetime.now(UTC).isoformat(),
        "gpu_wall_seconds": wall1 - wall0,
        "n_scenarios": n, "n_cells": len(res["cells"]), "n_onsets": len(res["onsets"]),
        "n_eligible": res["n_eligible"], "seed": 0,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    gate11 = ({"status": "skipped"} if (args.quick or args.no_gate11)
              else loc.determinism_gate(args.out))
    payload = {"release": "v1.0-commag", "gate_G11": gate11, "numbers": numbers,
               "raw_summary": {"n_cells": len(res["cells"]),
                               "n_eligible": res["n_eligible"],
                               "excluded_logged": res["excluded_count"]}}
    (out / "manuscript_numbers.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("gates:", gate11.get("status"))
    print("validation:", numbers["validation"])
    print("n cells:", len(res["cells"]), "n onsets:", len(res["onsets"]), "eligible:", res["n_eligible"])
    print("gpu_wall_seconds:", round(wall1 - wall0, 1))


if __name__ == "__main__":
    main()
