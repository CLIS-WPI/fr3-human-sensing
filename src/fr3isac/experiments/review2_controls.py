"""Second external review: correctness controls on the rerun-v2 data (no new ray tracing), F1.

1. Residual-matched bank: templates u(n) = exp(j 4 pi d(n) / lambda) - 1 for n >= n0, 0 before,
   with d(n) = A [sin(2 pi f (n - n0) / PRF + phi) - sin(phi)] (the data's motion law and
   occupied-reference subtraction). (A, f, phi) = the 36-set redesigned selection; the onset
   offset n0 (samples from window start) is a parameter on OFFSETS. Same statistic,
   per-window Pfa and calibration sizes as the main run.
2. The intermittent-path hidden events: per-event breakdown.
3. Pfa = 1e-5 per window: baseline and 64-template redesigned banks, calibration 1e7 and
   validation 1e7 fresh windows, event-level false alarms on 20-h continuous streams.
4. Reference-noise check: the main run models a K-frame averaged reference as white
   extra noise (variance s2/K per sample); here the reference error is one constant per
   window (r * 1, r ~ CN(0, s2/K)), as an averaged static reference produces.

All miss curves reuse the main run's per-onset noise (same seeds) and pair with the
stored curves under the same bootstrap resamples. New thresholds use seed children beyond
those of the main run (12) and of control_36_energy (8).

Usage: python -m fr3isac.experiments.review2_controls
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fr3isac import geometry as geo
from fr3isac.experiments import rerun_v2 as rv
from fr3isac.experiments import rerun_v2_core as core

OFFSETS = np.arange(-128, 121, 8)     # onset offset inside the window, samples (6.25 ms step)
DETS = ("residual", "baseline_pfa1e-5", "redesigned_pfa1e-5")
PFA_LOW = 1e-5
N_LOW = 10_000_000                    # calibration and validation windows at PFA_LOW
SEED_BASE = 20                        # main run 12 + control_36_energy 8
REF_K = (16, 64)
N_REF = 1_000_000


def residual_bank(fc, params, cfg, offsets=OFFSETS):
    st = cfg["slow_time"]
    L, prf = st["window_samples"], st["prf_hz"]
    n = np.arange(L)
    rows, keep = [], []
    for a, f, p in params:
        for n0 in offsets:
            d = np.where(n >= n0, a * (np.sin(2 * np.pi * f * (n - n0) / prf + p) - np.sin(p)), 0.0)
            u = np.exp(1j * 4 * np.pi * fc / core.C * d) - 1.0
            if np.sum(np.abs(u) ** 2) > 1e-6:   # drop templates with (numerically) no motion
                rows.append(u)
                keep.append((a, f, p, int(n0)))
    return np.asarray(rows), keep


def const_fraction(v):
    """Share of energy along the constant vector: |<1, v>|^2 / (L ||v||^2), per row."""
    v = np.atleast_2d(v)
    return np.abs(v.sum(axis=1)) ** 2 / (v.shape[1] * np.sum(np.abs(v) ** 2, axis=1))


def ref_noise_pfa(bank, s2, thr, k, n, ss, chunk=100_000):
    """Per-window Pfa when the subtracted reference carries a constant error r ~ CN(0, s2/k)."""
    rng = np.random.default_rng(ss)
    L, fa = bank.shape[1], 0
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        X = core._cn(rng, (L, m), s2) + core._cn(rng, (1, m), s2 / k)
        fa += int(np.sum(core.window_stat(bank, X, s2) > thr))
    return {"pfa": fa / n, "wilson_ci95": list(core.wilson(fa, n)), "N": n}


def _carrier(args):
    ci, fc, cfg, run, ctrl_dir = args
    t0 = time.time()
    z = np.load(Path(run) / "rt_cache.npz")
    ana = json.loads((Path(run) / "analysis.json").read_text())
    p36 = np.asarray(json.loads((Path(ctrl_dir) / "control.json").read_text())["same_count_params"])
    ss = core.seed_sequences(cfg["master_seed"])
    det, st = cfg["detector"], cfg["slow_time"]
    s2 = float(z["s2"])
    lim_db = cfg["point_targets"]["max_eirp_above_nominal_db"]
    nk = SEED_BASE + 3 * len(cfg["carriers_hz"])
    cal_kids, val_kids = ss["calibration"].spawn(nk)[SEED_BASE:], ss["validation"].spawn(nk + 2 * 4)[SEED_BASE:]
    ref_kids = val_kids[3 * len(cfg["carriers_hz"]):]
    fa_kids = ss["fa_stream"].spawn(12 + 8)[12:]
    cfg5 = copy.deepcopy(cfg)
    cfg5["detector"].update(pfa_per_window=PFA_LOW, n_calibration_windows=N_LOW, n_validation_windows=N_LOW,
                            reference_averaging_frames=[])
    main_cell = next(c for c in ana["cells"] if c["carrier_ghz"] == fc / 1e9)
    sig, vis = z[f"test_sig_{fc}"], z[f"test_vis_{fc}"]
    on = z["m_test"][z["test_idx"]].reshape(-1, 4)[:, 3]

    Bres, kept = residual_bank(fc, p36, cfg)
    Bb = core.bank_from_params(fc, core.baseline_params(cfg), cfg)
    Br = core.bank_from_params(fc, np.asarray(ana["redesigned_params"]), cfg)
    B = {"residual": Bres, "baseline_pfa1e-5": Bb, "redesigned_pfa1e-5": Br}
    cal = {"residual": core.calibrate(Bres, s2, cfg, cal_kids[3 * ci], val_kids[3 * ci]),
           "baseline_pfa1e-5": core.calibrate(Bb, s2, cfg5, cal_kids[3 * ci + 1], val_kids[3 * ci + 1]),
           "redesigned_pfa1e-5": core.calibrate(Br, s2, cfg5, cal_kids[3 * ci + 2], val_kids[3 * ci + 2])}
    for j, d in enumerate(("baseline_pfa1e-5", "redesigned_pfa1e-5")):
        cal[d]["events"] = core.event_false_alarms(B[d], s2, cal[d]["threshold"], cfg, fa_kids[2 * ci + j])
    thr = {d: cal[d]["threshold"] for d in DETS}
    print(f"[{fc / 1e9:g}] calibrated, {time.time() - t0:.0f} s", flush=True)

    # constant-vector content (item 1) and coherent reference error (item 4)
    share = core.constant_term_share(sig[vis], on[vis], cfg)
    tmpl = {"baseline_36": const_fraction(Bb), "redesigned_64": const_fraction(Br),
            "redesigned_36": const_fraction(Br[:36]), "residual": const_fraction(Bres)}
    refn = {}
    for bi, (b, bank) in enumerate((("baseline", Bb), ("redesigned", Br))):
        thr_main = main_cell["banks"][b]["calibration"]["threshold"]
        refn[b] = {str(k): ref_noise_pfa(bank, s2, thr_main, k, N_REF, ref_kids[2 * ci + bi].spawn(len(REF_K))[ki])
                   for ki, k in enumerate(REF_K)}
        refn[b]["white_model_main_run"] = main_cell["banks"][b]["calibration"]["reference_averaging"]

    # miss curves: same coarse grid rule and per-onset noise as the main run
    thr_b = main_cell["banks"]["baseline"]["calibration"]["threshold"]
    nf = np.array([core.noisefree_required_db(s, t, Bb, s2, thr_b, cfg) for s, t in zip(sig[vis], on[vis], strict=True)])
    coarse = float(np.percentile(nf, 90)) + np.arange(det["eirp_grid_span_db"][0],
                                                      det["eirp_grid_span_db"][1] + 1e-9, det["eirp_grid_step_db"])
    noise_ss = ss["noise_trials"].spawn(len(cfg["carriers_hz"]))[ci].spawn(len(sig))
    T, n = det["n_noise_trials"], st["record_samples"]

    def evaluate(grids):
        P = {d: np.zeros((len(sig), len(grids[d]))) for d in grids}
        for i in range(len(sig)):
            r = np.random.default_rng(noise_ss[i])
            X = (r.standard_normal((n, T)) + 1j * r.standard_normal((n, T))) * math.sqrt(s2 / 2)
            for d, g in grids.items():
                if len(g):
                    P[d][i] = core.miss_curve(sig[i], on[i], B[d], s2, thr[d], 10 ** (np.asarray(g) / 20), X, cfg)
        return P

    Pc = evaluate({d: coarse for d in DETS})
    while any(Pc[d][vis].mean(0)[-1] > 0.10 for d in DETS) and coarse[-1] < lim_db:
        ext = coarse[-1] + det["eirp_grid_step_db"] * np.arange(1, 9)
        Pe = evaluate({d: ext for d in DETS})
        Pc = {d: np.concatenate([Pc[d], Pe[d]], axis=1) for d in DETS}
        coarse = np.concatenate([coarse, ext])
    grids, P, _ = rv._refine(Pc, evaluate, vis, coarse, DETS, cfg)
    print(f"[{fc / 1e9:g}] curves done, {time.time() - t0:.0f} s", flush=True)
    return {"fc": fc, "calibration": cal, "grids": grids, "P": P,
            "residual_bank": {"n_templates": int(Bres.shape[0]),
                              "n_dropped_no_motion": int(len(p36) * len(OFFSETS) - Bres.shape[0])},
            "constant_offset_fraction_test_visible": {
                "mean": float(np.mean(share)), "median": float(np.median(share)),
                "p10": float(np.percentile(share, 10)), "p90": float(np.percentile(share, 90))},
            "template_constant_fraction": {k: {"min": float(v.min()), "median": float(np.median(v)),
                                               "max": float(v.max()), "values": v.tolist()} for k, v in tmpl.items()},
            "reference_error_coherent": refn}


def hidden_events(cfg, run):
    """Item 2: per-event breakdown of hidden onsets with a forearm path at some anchors."""
    z = np.load(Path(run) / "rt_cache.npz")
    ana = json.loads((Path(run) / "analysis.json").read_text())
    cv = np.load(Path(run) / "miss_curves.npz")
    desc = geo.load_descriptor(cfg["scene"])
    bs, blk = geo.bs_position(desc), geo.blockers(desc)
    times = core.record_times(cfg)
    st = cfg["slow_time"]
    L = st["window_samples"]
    s2 = float(z["s2"])
    starts = core.window_starts(cfg)
    mot = z["m_test"][z["test_idx"]].reshape(-1, 4)
    lim_db = cfg["point_targets"]["max_eirp_above_nominal_db"]
    out = {}
    for fc in cfg["carriers_hz"]:
        cell = next(c for c in ana["cells"] if c["carrier_ghz"] == fc / 1e9)
        thr = cell["banks"]["baseline"]["calibration"]["threshold"]
        e0 = cell["eirp0_dbm"]
        Bb = core.bank_from_params(fc, core.baseline_params(cfg), cfg)
        sig, vis, frac = z[f"test_sig_{fc}"], z[f"test_vis_{fc}"], z[f"test_fafrac_{fc}"]
        pid = z[f"test_pid_{fc}"]
        g, Pm = cv[f"grid_{fc}_baseline"], cv[f"P_{fc}_baseline"]
        jl = int(np.argmin(np.abs(g - lim_db)))
        hid = np.flatnonzero(~vis)
        part = hid[frac[hid] > 0]
        ev = []
        for i in part:
            a, f, p, t0 = mot[i]
            root = (*z["test_xy"][pid[i]], geo.TARGET_HEIGHT_M)
            los = np.array([not any(geo.segment_aabb_intersect(bs, np.asarray(s.forearm_position, float),
                                                              o["center_m"], o["size_m"]) for o in blk)
                            for s in geo.onset_arm_trajectory(root, times, a, f, p, t0)])
            m = core.deadline_mask(t0, cfg)
            legal = np.zeros(times.size, bool)
            wins = []
            for s0 in starts[m]:
                legal[s0:s0 + L] = True
                w = sig[i][s0:s0 + L]
                e = float(np.sum(np.abs(w) ** 2) / s2)
                b = float(core.window_stat(Bb, w[:, None], s2)[0])
                wins.append({"start_s": float(times[s0]), "end_s": float(times[s0 + L - 1]),
                             "residual_energy_snr_db": 10 * math.log10(e) if e > 0 else None,
                             "best_template_snr_db": 10 * math.log10(b) if b > 0 else None,
                             "capture_db": 10 * math.log10(b / e) if e > 0 and b > 0 else None,
                             "constant_fraction": float(const_fraction(w)[0]) if e > 0 else None,
                             "los_samples": int(np.sum(los[s0:s0 + L] & (times[s0:s0 + L] > t0)))})
            vt = times[legal & los & (times > t0)]
            post = legal & (times > t0)
            nfr = core.noisefree_required_db(sig[i], t0, Bb, s2, thr, cfg)
            # signal energy where the forearm segment has LoS vs. where it does not (post-onset, legal)
            el = float(np.sum(np.abs(sig[i][post & los]) ** 2))
            en = float(np.sum(np.abs(sig[i][post & ~los]) ** 2))
            ev.append({"onset_index": int(i), "placement": int(pid[i]), "motion": [a, f, p, t0],
                       "anchor_fraction_with_forearm_path": float(frac[i]),
                       "legal_samples_post_onset": int(post.sum()),
                       "los_samples_in_legal_windows": int(vt.size),
                       "los_time_intervals_s": _intervals(vt, 1 / st["prf_hz"]),
                       "residual_energy_share_on_los_samples": el / (el + en) if el + en > 0 else None,
                       "windows": wins,
                       "noisefree_crossing_dbm": e0 + nfr if np.isfinite(nfr) else None,
                       "miss_curve": {"eirp_dbm": (e0 + g).tolist(), "miss": Pm[i].tolist()},
                       "miss_at_search_limit": float(Pm[i, jl])})
        out[f"{fc / 1e9:g}"] = {
            "search_limit_dbm": e0 + float(g[jl]), "n_hidden": int(hid.size), "n_intermittent": int(part.size),
            "mean_miss_at_search_limit": {
                "all_hidden": float(Pm[hid, jl].mean()),
                "no_path_hidden": float(Pm[hid[frac[hid] == 0], jl].mean()),
                "intermittent": float(Pm[part, jl].mean())},
            "events": ev}
    return out


def _intervals(t, dt):
    if t.size == 0:
        return []
    br = np.flatnonzero(np.diff(t) > 1.5 * dt)
    s = np.r_[0, br + 1]
    e = np.r_[br, t.size - 1]
    return [[float(t[a]), float(t[b])] for a, b in zip(s, e, strict=True)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rerun_v2.yaml")
    ap.add_argument("--run", default="results/rerun_v2/full")
    ap.add_argument("--control", default="results/rerun_v2/control")
    ap.add_argument("--out", default="results/rerun_v2/review2")
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = core.load_cfg(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    all_fc = cfg["carriers_hz"]
    with ProcessPoolExecutor(len(all_fc)) as ex:
        res = {r["fc"]: r for r in ex.map(_carrier, [(ci, fc, cfg, a.run, a.control) for ci, fc in enumerate(all_fc)])}
    curves = {}
    for fc, r in res.items():
        for d in DETS:
            curves[f"grid_{fc}_{d}"], curves[f"P_{fc}_{d}"] = r["grids"][d], r["P"][d]
    np.savez_compressed(out / "review2_curves.npz", **curves)

    # paired placement bootstrap over all detectors and carriers (one resample set)
    z = np.load(Path(a.run) / "rt_cache.npz")
    cv = np.load(Path(a.run) / "miss_curves.npz")
    ccv = np.load(Path(a.control) / "control_curves.npz")
    ss = core.seed_sequences(cfg["master_seed"])
    n_place = len(z["test_idx"])
    W = core.resample_weights(ss["bootstrap"], n_place, cfg["bootstrap"]["n_resamples"])
    pc = cfg["bootstrap"]["ci_percent"]
    req, boot = {}, {}
    for fc in all_fc:
        vis, pid, e0 = cv[f"vis_{fc}"], cv[f"pid_{fc}"], float(cv[f"eirp0_{fc}"])
        src = {"baseline": (cv[f"grid_{fc}_baseline"], cv[f"P_{fc}_baseline"]),
               "redesigned_64": (cv[f"grid_{fc}_redesigned"], cv[f"P_{fc}_redesigned"]),
               "redesigned_36": (ccv[f"grid_{fc}_same_count"], ccv[f"P_{fc}_same_count"]),
               **{d: (curves[f"grid_{fc}_{d}"], curves[f"P_{fc}_{d}"]) for d in DETS}}
        for d, (g, Pm) in src.items():
            S, C = core.placement_sums(Pm, vis, pid, n_place)
            pt = core.crossing(S.sum(0) / C.sum(), g)
            req[(d, fc)] = None if pt is None else e0 + pt
            boot[(d, fc)] = e0 + core.boot_crossings(S, C, g, W)

    def paired(k1, k2):
        v = None if req[k1] is None or req[k2] is None else req[k1] - req[k2]
        return {"value_db": v, "ci95_db": core.pct_ci(boot[k1] - boot[k2], pc)}

    dets_all = ("baseline", "redesigned_64", "redesigned_36", *DETS)
    rows = [{"carrier_ghz": fc / 1e9, "detector": d, "pfa_per_window": PFA_LOW if "1e-5" in d else 1e-3,
             "required_eirp_dbm": req[(d, fc)], "ci95_dbm": core.pct_ci(boot[(d, fc)], pc),
             "n_resamples_without_crossing": int(np.sum(~np.isfinite(boot[(d, fc)])))}
            for fc in all_fc for d in dets_all]
    savings = []
    for fc in all_fc:
        for d, ref in (("residual", "baseline"), ("residual", "redesigned_64"), ("residual", "redesigned_36"),
                       ("redesigned_pfa1e-5", "baseline_pfa1e-5")):
            savings.append({"carrier_ghz": fc / 1e9, "detector": d, "vs": ref,
                            **paired((ref, fc), (d, fc)), "note": "positive = detector needs less EIRP than vs"})
    extra = [{"carrier_ghz": fc / 1e9, "detector": d, **paired((d, fc), (d, all_fc[0]))}
             for d in dets_all for fc in all_fc[1:]]
    first = res[all_fc[0]]
    result = {
        "manifest": rv.manifest(cfg, "review2_controls", {"wall_s": time.time() - t0}),
        "residual_bank": {"params": "36-set redesigned selection (control.json same_count_params)",
                          "onset_offset_grid_samples": OFFSETS.tolist(),
                          "onset_offset_grid_ms": (OFFSETS / cfg["slow_time"]["prf_hz"] * 1e3).tolist(),
                          **first["residual_bank"]},
        "per_carrier": {f"{fc / 1e9:g}": {k: res[fc][k] for k in
                                          ("calibration", "residual_bank", "constant_offset_fraction_test_visible",
                                           "template_constant_fraction", "reference_error_coherent")}
                        for fc in all_fc},
        "rows": rows, "savings": savings, "extra_vs_3p5": extra,
        "hidden_intermittent": hidden_events(cfg, a.run)}
    (out / "review2.json").write_text(json.dumps(result, indent=2))
    print(f"done, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
