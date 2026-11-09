"""Control on the rerun-v2 data (no new ray tracing), F1, all four carriers.

1. Same-count bank: the redesigned-bank selection (same candidates, development data and
   coverage criterion) limited to as many parameter sets as the baseline bank (36).
2. Energy detector: per-window slow-time energy ||x_w||^2 / sigma^2 in the same windows,
   thresholded at the same per-window Pfa.

Both use the same per-onset noise as the main run (regenerated from the same seeds), so
the stored baseline miss curves pair with them under the same bootstrap resamples.
New thresholds use calibration/validation seed children beyond those of the main run.

Usage: python -m fr3isac.experiments.control_36_energy --run results/rerun_v2/full \
           --out results/rerun_v2/control
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import gammainccinv

from fr3isac.experiments import rerun_v2 as rv
from fr3isac.experiments import rerun_v2_core as core

DETS = ("same_count", "energy")


def energy_null(L, n, ss, chunk=100_000):
    rng = np.random.default_rng(ss)
    out = np.empty(n)
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        # ||n||^2 / s2 with n ~ CN(0, s2 I_L): sum of L unit exponentials (Gamma(L, 1))
        out[i:i + m] = np.sum(np.abs(core._cn(rng, (L, m), 1.0)) ** 2, axis=0)
    return out


def energy_calibrate(cfg, ss_cal, ss_val):
    d, L = cfg["detector"], cfg["slow_time"]["window_samples"]
    pfa = d["pfa_per_window"]
    thr = float(np.quantile(energy_null(L, d["n_calibration_windows"], ss_cal), 1.0 - pfa))
    val = energy_null(L, d["n_validation_windows"], ss_val)
    fa = int(np.sum(val > thr))
    lo, hi = core.wilson(fa, val.size)
    return {"threshold": thr, "analytic_threshold": float(gammainccinv(L, pfa)),
            "calibration_N": d["n_calibration_windows"], "validation_N": int(val.size),
            "false_alarms": fa, "pfa": fa / val.size, "wilson_ci95": [lo, hi],
            "accept": bool(lo <= pfa <= hi)}


def energy_miss(sig, onset_s, s2, thr, amps, X, cfg):
    m = core.deadline_mask(onset_s, cfg)
    S = core.windows(sig, cfg)[m]            # (w, L)
    N = core.windows(X, cfg)[m]              # (w, L, T)
    out = np.empty(len(amps))
    for j, a in enumerate(amps):
        stat = np.sum(np.abs(a * S[:, :, None] + N) ** 2, axis=1) / s2   # (w, T)
        out[j] = 1.0 - np.mean((stat > thr).any(axis=0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rerun_v2.yaml")
    ap.add_argument("--run", default="results/rerun_v2/full")
    ap.add_argument("--out", default="results/rerun_v2/control")
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = core.load_cfg(a.config)
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(run / "rt_cache.npz")
    ana = json.loads((run / "analysis.json").read_text())
    cv = np.load(run / "miss_curves.npz")
    ss = core.seed_sequences(cfg["master_seed"])
    st, det, rb = cfg["slow_time"], cfg["detector"], cfg["banks"]["redesigned"]
    s2 = float(z["s2"])
    all_fc = cfg["carriers_hz"]
    n_main = len(all_fc) * len(rv.BANKS)
    lim_db = cfg["point_targets"]["max_eirp_above_nominal_db"]

    # ---- same-count bank: identical selection, size = baseline bank size -------
    k36 = len(core.baseline_params(cfg))
    cand = core.candidate_params(ss["design_candidates"], cfg)
    dev_on = z["m_dev"].reshape(-1, 4)[:, 3]
    caps = []
    for fc in all_fc:
        vis = z[f"dev_vis_{fc}"]
        caps.append(core.capture_matrix(core.bank_from_params(fc, cand, cfg), z[f"dev_sig_{fc}"][vis],
                                        dev_on[vis], cfg))
    cov = core.coverage_search(caps, rb["coverage_tolerance_db"], k36, rb["max_templates_search"])
    p36 = cand[cov["chosen"]]
    p64 = np.asarray(ana["redesigned_params"])
    assert np.allclose(p36, p64[:k36]), "greedy is prefix-consistent: 36-set must equal first 36 of the 64-set"
    coverage = {f"{fc / 1e9:g}": {"pool_p90_loss_db": cov["pool_p90_loss_db"][i],
                                  "same_count_p90_loss_db": cov["bank_p90_loss_db"][i],
                                  "meets": bool(cov["bank_p90_loss_db"][i]
                                                <= cov["pool_p90_loss_db"][i] + rb["coverage_tolerance_db"])}
                for i, fc in enumerate(all_fc)}

    cal_kids = ss["calibration"].spawn(n_main + 2 * len(all_fc))[n_main:]
    val_kids = ss["validation"].spawn(n_main + 2 * len(all_fc))[n_main:]
    noise_par = ss["noise_trials"].spawn(len(all_fc))
    T, n = det["n_noise_trials"], st["record_samples"]
    cells, curves = {}, {}
    for ci, fc in enumerate(all_fc):
        main_cell = next(c for c in ana["cells"] if c["carrier_ghz"] == fc / 1e9)
        eirp0 = float(z[f"eirp0_{fc}"])
        sig, vis = z[f"test_sig_{fc}"], z[f"test_vis_{fc}"]
        on = z["m_test"][z["test_idx"]].reshape(-1, 4)[:, 3]
        assert np.array_equal(vis, cv[f"vis_{fc}"])
        B36 = core.bank_from_params(fc, p36, cfg)
        cal = {"same_count": core.calibrate(B36, s2, cfg, cal_kids[2 * ci], val_kids[2 * ci]),
               "energy": energy_calibrate(cfg, cal_kids[2 * ci + 1], val_kids[2 * ci + 1])}
        thr = {d: cal[d]["threshold"] for d in DETS}
        # coarse grid exactly as the main run (baseline-bank noise-free P90)
        Bb = core.bank_from_params(fc, core.baseline_params(cfg), cfg)
        thr_b = main_cell["banks"]["baseline"]["calibration"]["threshold"]
        nf = np.array([core.noisefree_required_db(s, t, Bb, s2, thr_b, cfg)
                       for s, t in zip(sig[vis], on[vis], strict=True)])
        coarse = float(np.percentile(nf, 90)) + np.arange(det["eirp_grid_span_db"][0],
                                                          det["eirp_grid_span_db"][1] + 1e-9,
                                                          det["eirp_grid_step_db"])
        noise_ss = noise_par[ci].spawn(len(sig))

        def evaluate(grids, sig=sig, on=on, noise_ss=noise_ss, B36=B36, thr=thr):
            P = {d: np.zeros((len(sig), len(grids[d]))) for d in grids}
            for i in range(len(sig)):
                r = np.random.default_rng(noise_ss[i])
                X = (r.standard_normal((n, T)) + 1j * r.standard_normal((n, T))) * math.sqrt(s2 / 2)
                for d, g in grids.items():
                    if not len(g):
                        continue
                    amps = 10 ** (np.asarray(g) / 20)
                    P[d][i] = (core.miss_curve(sig[i], on[i], B36, s2, thr[d], amps, X, cfg) if d == "same_count"
                               else energy_miss(sig[i], on[i], s2, thr[d], amps, X, cfg))
            return P

        Pc = evaluate({d: coarse for d in DETS})
        # extend the coarse grid upward (all detectors) while any crossing lies above it
        while any(Pc[d][vis].mean(0)[-1] > 0.10 for d in DETS) and coarse[-1] < lim_db:
            ext = coarse[-1] + det["eirp_grid_step_db"] * np.arange(1, 9)
            Pe = evaluate({d: ext for d in DETS})
            Pc = {d: np.concatenate([Pc[d], Pe[d]], axis=1) for d in DETS}
            coarse = np.concatenate([coarse, ext])
        grids, P, _ = rv._refine(Pc, evaluate, vis, coarse, DETS, cfg)
        for d in DETS:
            curves[f"grid_{fc}_{d}"], curves[f"P_{fc}_{d}"] = grids[d], P[d]
        cells[fc] = {"eirp0_dbm": eirp0, "calibration": cal}
        print(f"control {fc / 1e9:g} GHz done, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(out / "control_curves.npz", **curves)

    # ---- paired placement bootstrap vs the stored baseline (and 64-template) curves
    n_place = len(z["test_idx"])
    W = core.resample_weights(ss["bootstrap"], n_place, cfg["bootstrap"]["n_resamples"])
    ci_pct = cfg["bootstrap"]["ci_percent"]
    rows = []
    for fc in all_fc:
        vis, pid, e0 = cv[f"vis_{fc}"], cv[f"pid_{fc}"], cells[fc]["eirp0_dbm"]
        req, boot = {}, {}
        for d, g, Pm in [("baseline", cv[f"grid_{fc}_baseline"], cv[f"P_{fc}_baseline"]),
                         ("redesigned_64", cv[f"grid_{fc}_redesigned"], cv[f"P_{fc}_redesigned"]),
                         *[(d, curves[f"grid_{fc}_{d}"], curves[f"P_{fc}_{d}"]) for d in DETS]]:
            S, C = core.placement_sums(Pm, vis, pid, n_place)
            pt = core.crossing(S.sum(0) / C.sum(), g)
            req[d] = None if pt is None else e0 + pt
            boot[d] = e0 + core.boot_crossings(S, C, g, W)
        for d in ("redesigned_64", *DETS):
            sv = None if req[d] is None or req["baseline"] is None else req["baseline"] - req[d]
            rows.append({"carrier_ghz": fc / 1e9, "detector": d, "regime": "F1",
                         "required_eirp_dbm": req[d], "ci95_dbm": core.pct_ci(boot[d], ci_pct),
                         "baseline_required_eirp_dbm": req["baseline"],
                         "saving_vs_baseline_db": sv,
                         "saving_ci95_db": core.pct_ci(boot["baseline"] - boot[d], ci_pct),
                         "n_resamples_without_crossing": int(np.sum(~np.isfinite(boot[d])))})
    res = {"manifest": rv.manifest(cfg, "control_36_energy", {"wall_s": time.time() - t0,
                                                              "search_limit_db_above_nominal": lim_db}),
           "same_count_size": k36, "same_count_params": p36.tolist(),
           "same_count_coverage": coverage,
           "calibration": {f"{fc / 1e9:g}": cells[fc]["calibration"] for fc in all_fc},
           "rows": rows}
    (out / "control.json").write_text(json.dumps(res, indent=2))
    for r in rows:
        print(r["carrier_ghz"], r["detector"], r["required_eirp_dbm"], r["ci95_dbm"],
              r["saving_vs_baseline_db"], r["saving_ci95_db"])


if __name__ == "__main__":
    main()
