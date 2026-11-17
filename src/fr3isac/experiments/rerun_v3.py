"""v3 production run (review 3, phase 3): clean rerun with the approved scene and model changes.

- Scene v3 (glass partition = one plane, 0.06 m slab, refraction on for target paths at the
  minimum max_depth that returns the refracted path; explicit target size), cfg["rt"].
- Fresh test placements (0.25 m-offset grid, same eligibility rule and evenly spaced subset)
  and freshly drawn test motions. Development data and all banks stay as designed in the v2
  run (cfg["v3"]["design_run"]): redesigned 64-set and per-carrier parameters are read, not
  reselected; no per-carrier variant exists at a carrier it was not designed for.
- Carriers 3.5/7.5/12/15/28 GHz; detectors baseline, baseline + constant, redesigned,
  per-carrier, energy; reference frames K (thresholds recalibrated per K; one constant error
  r ~ CN(0, s2/K) on the whole record, same unit draw for every K); deadlines 100/200 ms.
- Onset label from the RCSSolver rule with the explicit target size: visible (forearm leg
  unobstructed at every record sample), glass-only (blocked only by the partition, so the
  refracted path exists), other-blocked. The primary metric uses visible onsets; glass-only
  onsets are reported separately with transmission enabled.

Stages: rt (GPU), analyze, numbers, gates, summary (writes rt_summary.npz from rt_cache.npz). Usage:
  python -m fr3isac.experiments.rerun_v3 rt --out results/rerun_v3/full
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fr3isac import geometry as geo
from fr3isac.experiments import rerun_v2 as rv
from fr3isac.experiments import rerun_v2_core as core
from fr3isac.experiments.control_36_energy import energy_miss
from fr3isac.seeding import derive_seed

STREAMS = ("test_motions", "noise", "ref_error", "calibration", "validation", "fa_stream", "pfa_ref", "bootstrap")
DETS = ("baseline", "base_plus_dc", "redesigned", "per_carrier", "energy")
LABELS = {0: "visible", 1: "glass_only", 2: "other_blocked"}


def streams(master_seed):
    """v3 streams from derive_seed(master, "rerun_v3", name); independent of the v2 streams."""
    return {n: np.random.SeedSequence(derive_seed(master_seed, "rerun_v3", n)) for n in STREAMS}


def kname(k):
    return "inf" if k is None else str(k)


SUMMARY_MAX_BYTES = 50_000   # rt_summary.npz keeps every rt_cache array up to this size


def write_rt_summary(run_dir):
    """Write rt_summary.npz: the small arrays of rt_cache.npz (labels, placements, gains, nominal EIRP)."""
    run_dir = Path(run_dir)
    z = np.load(run_dir / "rt_cache.npz")
    small = {k: z[k] for k in z.files if z[k].nbytes <= SUMMARY_MAX_BYTES}
    small["source_rt_cache_sha256"] = np.array(hashlib.sha256((run_dir / "rt_cache.npz").read_bytes()).hexdigest())
    np.savez_compressed(run_dir / "rt_summary.npz", **small)
    return sorted(small)


def rt_small(run_dir):
    """rt_cache.npz if present, else the shipped rt_summary.npz (enough for the numbers stage)."""
    run_dir = Path(run_dir)
    full = run_dir / "rt_cache.npz"
    return np.load(full if full.exists() else run_dir / "rt_summary.npz")


def test_placements(cfg, desc):
    bs, blk = geo.bs_position(desc), geo.blockers(desc)
    off, step = cfg["test_grid_offset_m"], cfg["grid_step_m"]
    pts = [(float(x), float(y)) for x in np.arange(geo.ROI_X[0] + off, geo.ROI_X[1] + 1e-9, step)
           for y in np.arange(geo.ROI_Y[0] + off, geo.ROI_Y[1] + 1e-9, step)]
    elig = [p for p in pts if not any(geo.segment_aabb_intersect(bs, np.array([*p, geo.TARGET_HEIGHT_M]),
                                                                 o["center_m"], o["size_m"]) for o in blk)]
    return elig, core.split_placements(elig, cfg["n_test_placements"])[1]


# --------------------------------------------------------------------------- #
# Stage rt
# --------------------------------------------------------------------------- #
def _rt_stage(cfg, out, dry):
    from fr3isac.config import FairnessConfig
    from fr3isac.experiments import localization as loc
    from fr3isac.experiments import rt_v3
    from fr3isac.fairness.normalize import normalize
    from fr3isac.sensing.rt_channel import detector_input_scale, thermal_noise_variance
    from fr3isac.sensing.rt_localize import range_m_to_bin
    from fr3isac.sensing.rt_slowtime import interp_complex_series, path_records, synthesize_cfr

    t_start = time.time()
    rt_calls = [0]
    rtc = cfg["rt"]
    desc = geo.load_descriptor(cfg["scene"])
    bs, blk = geo.bs_position(desc), geo.blockers(desc)
    half = rtc["target_size_m"] / 2
    elig, test = test_placements(cfg, desc)
    dz = rt_small(cfg["v3"]["design_run"])
    core.assert_disjoint([tuple(p) for p in dz["dev_xy"].tolist()], test)
    if dry:
        n = cfg["dry_run"]["n_test_placements"]
        test = test[:: max(1, len(test) // n)][:n]
    ss = streams(cfg["master_seed"])
    m_test = core.motions_per_placement(ss["test_motions"], test, cfg["k_test_motions_per_placement"], cfg)
    times = core.record_times(cfg)
    st = cfg["slow_time"]
    dt_anchor = st["window_samples"] / st["prf_hz"] / st["anchors_per_window"]
    offsets = (np.arange(loc.NSC) - loc.NSC // 2) * loc.SCS
    c2 = detector_input_scale(loc.P_TX_W, loc.NSC)
    f2c = cfg["f2_spot_check_carrier_hz"]
    carriers = cfg["dry_run"]["carriers_hz"] if dry else cfg["carriers_hz"]

    def label(root, motion, ts):
        """0 visible, 1 glass-only, 2 other-blocked; per-sample blocker names."""
        names, per = set(), []
        for s in geo.onset_arm_trajectory(root, ts, *motion):
            h = rt_v3.blockers_on_leg(bs, np.asarray(s.forearm_position, float), np.asarray(s.torso_position, float),
                                      blk, half)
            names.update(h)
            per.append(h)
        lab = 0 if not names else (1 if names == {"partition"} else 2)
        return lab, sorted(names), per

    def static_ref(fc, model, root, gain, bg):
        rt_calls[0] += 1
        stc = rt_v3.solve_cfr(fc, desc, model, geo.static_sample(root), offsets, gain, rtc)
        r_hat, _ = loc.localize_body_bin(bg[:, None] + stc[:, None], bg[:, None], loc.OCC_BW,
                                         desc["range_search_min_m"], desc["range_search_max_m"])
        return stc, r_hat, complex(c2 * loc._body_vector(bg, stc, r_hat)[0])

    def onset_record(fc, root, motion, gain, bg, r_hat):
        A, f, phi, t0 = motion
        anchors = t0 + np.arange(int(math.ceil((times[-1] - t0) / dt_anchor)) + 1) * dt_anchor
        ta_amp, fa_amp = [], []
        for ta in anchors:
            s = geo.onset_arm_trajectory(root, [ta], A, f, phi, t0)[0]
            rt_calls[0] += 1
            r_t = float(np.linalg.norm(np.asarray(s.torso_position) - bs))
            r_f = float(np.linalg.norm(np.asarray(s.forearm_position) - bs))
            a_t = a_f = 0.0 + 0.0j
            for rec in path_records(rt_v3.rcs_paths(fc, desc, "R1-A", s, rtc)):
                if abs(rec["tau"] - 2 * r_t / core.C) <= abs(rec["tau"] - 2 * r_f / core.C):
                    a_t += rec["a"]
                else:
                    a_f += rec["a"]
            ta_amp.append(a_t)
            fa_amp.append(a_f)
        g = 10.0 ** (gain / 20.0)
        samples = geo.onset_arm_trajectory(root, times, A, f, phi, t0)
        r_torso = np.array([np.linalg.norm(np.asarray(s.torso_position) - bs) for s in samples])
        r_fa = np.array([np.linalg.norm(np.asarray(s.forearm_position) - bs) for s in samples])
        amps = np.vstack([interp_complex_series(anchors, np.asarray(ta_amp) * g, times),
                          interp_complex_series(anchors, np.asarray(fa_amp) * g, times)])
        cfr = synthesize_cfr(fc, offsets, amps, np.vstack([r_torso, r_fa]))
        raw = c2 * loc._body_vector(bg, cfr, r_hat)
        return raw - raw[0], float(np.mean(np.abs(np.asarray(fa_amp)) > 0)), raw, anchors

    def direct_record(fc, root, motion, gain, bg, r_hat, xr):
        cols = []
        for s in geo.onset_arm_trajectory(root, times, *motion):
            rt_calls[0] += 1
            cols.append(rt_v3.solve_cfr(fc, desc, "R1-A", s, offsets, gain, rtc))
        return c2 * loc._body_vector(bg, np.stack(cols, axis=1), r_hat) - xr

    data = {"times": times,
            "s2": thermal_noise_variance(loc.SCS, noise_figure_db=loc.NOISE_FIGURE_DB) / loc.NSC,
            "f1_rx_gain_db": FairnessConfig(regime="F1").bs_rx_gain_db,
            "m_test": m_test, "test_xy": np.asarray(test), "eligible_xy": np.asarray(elig)}
    checks = {"pre_onset_residual": [], "fast_vs_direct_nmse": [], "localization": [], "determinism": [],
              "label_vs_rt_anchor_fraction": [], "reference_mismatch_rel": []}
    # labels are geometric (carrier-independent)
    labels, blockers, expected_frac = [], [], []
    for ti, xy in enumerate(test):
        root = (xy[0], xy[1], geo.TARGET_HEIGHT_M)
        for mot in m_test[ti]:
            lab, names, _ = label(root, mot, times)
            A, f, phi, t0 = mot
            anchors = t0 + np.arange(int(math.ceil((times[-1] - t0) / dt_anchor)) + 1) * dt_anchor
            _, _, per_a = label(root, mot, anchors)
            # RT returns a forearm path at an anchor iff the leg is free or crosses only the plane
            expected_frac.append(float(np.mean([(not h) or set(h) == {"partition"} for h in per_a])))
            labels.append(lab)
            blockers.append(names)
    data["test_label"] = np.asarray(labels)
    data["test_pid"] = np.repeat(np.arange(len(test)), cfg["k_test_motions_per_placement"])
    n_val = rtc.get("validation_onsets_per_carrier", 0)
    keep = {}
    for fc in carriers:
        bg = rt_v3.bg_cfr(fc, desc, offsets)
        gain = loc._regime_gain_db(fc, cfg["antenna_regime"])
        fcfg = FairnessConfig(regime=cfg["antenna_regime"])
        data[f"eirp0_{fc}"] = fcfg.conducted_power_dbm + normalize(fcfg, fc).bs.gain_db
        nf2 = normalize(FairnessConfig(regime="F2"), fc).bs
        data[f"f2_tx_gain_db_{fc}"] = nf2.gain_db
        data[f"f2_eirp0_{fc}"] = FairnessConfig(regime="F2").conducted_power_dbm + nf2.gain_db
        keep[fc] = (bg, gain)
        do_f2 = (not dry) and fc == f2c
        gain_f2 = 2.0 * nf2.gain_db
        sig, frac, sig_f2 = [], [], []
        pts = {m: [] for m in cfg["point_targets"]["models"]}
        nv = 0
        for ti, xy in enumerate(test):
            root = (xy[0], xy[1], geo.TARGET_HEIGHT_M)
            true_bin = range_m_to_bin(float(np.linalg.norm(np.asarray(root) - bs)), loc.NSC, loc.OCC_BW)
            _, r_hat, xr = static_ref(fc, "R1-A", root, gain, bg)
            checks["localization"].append({"fc_hz": fc, "placement": list(xy), "model": "R1-A",
                                           "r_hat": int(r_hat), "true_bin": int(true_bin)})
            if do_f2:
                _, r_hat2, _ = static_ref(fc, "R1-A", root, gain_f2, bg)
            for mi_, mot in enumerate(m_test[ti]):
                j = ti * len(m_test[ti]) + mi_
                s, fr, raw, _ = onset_record(fc, root, mot, gain, bg, r_hat)
                pre = times <= mot[3]
                e_post = float(np.sum(np.abs(s[~pre]) ** 2))
                checks["pre_onset_residual"].append(float(np.sum(np.abs(s[pre]) ** 2) / e_post) if e_post > 0 else 0.0)
                checks["reference_mismatch_rel"].append(float(abs(raw[0] - xr) ** 2 / abs(xr) ** 2))
                checks["label_vs_rt_anchor_fraction"].append(
                    {"fc_hz": fc, "onset": j, "rt": fr, "expected": expected_frac[j], "label": labels[j]})
                if labels[j] == 0 and nv < n_val:
                    d = direct_record(fc, root, mot, gain, bg, r_hat, xr)
                    m = ~pre
                    f_ = raw - xr
                    checks["fast_vs_direct_nmse"].append(
                        {"fc_hz": fc, "placement": list(xy), "motion": mot.tolist(),
                         "nmse": float(np.sum(np.abs(f_[m] - d[m]) ** 2) / np.sum(np.abs(d[m]) ** 2))})
                    nv += 1
                sig.append(s)
                frac.append(fr)
                if do_f2:
                    sig_f2.append(onset_record(fc, root, mot, gain_f2, bg, r_hat2)[0])
            for model in pts:
                stc, rh, xrp = static_ref(fc, model, root, gain, bg)
                checks["localization"].append({"fc_hz": fc, "placement": list(xy), "model": model,
                                               "r_hat": int(rh), "true_bin": int(true_bin)})
                pts[model].append(c2 * loc._body_vector(bg, np.repeat(stc[:, None], times.size, axis=1), rh) - xrp)
        data[f"test_sig_{fc}"], data[f"test_fafrac_{fc}"] = np.asarray(sig), np.asarray(frac)
        if do_f2:
            data[f"test_sig_F2_{fc}"] = np.asarray(sig_f2)
        for model, recs in pts.items():
            data[f"pt_{model}_{fc}"] = np.asarray(recs)
        print(f"rt {fc / 1e9:g} GHz done, {rt_calls[0]} RT calls, {time.time() - t_start:.0f} s", flush=True)

    for fc in carriers:   # G11: re-simulate and compare bit-for-bit
        bg, gain = keep[fc]
        root = (test[0][0], test[0][1], geo.TARGET_HEIGHT_M)
        _, r_hat, _ = static_ref(fc, "R1-A", root, gain, bg)
        for j in range(cfg["g11_onsets_per_carrier"]):
            s2_ = onset_record(fc, root, m_test[0][j], gain, bg, r_hat)[0]
            ref = data[f"test_sig_{fc}"][j]
            checks["determinism"].append({"fc_hz": fc, "onset": j, "bit_identical": bool(np.array_equal(s2_, ref)),
                                          "max_abs_diff": float(np.max(np.abs(s2_ - ref)))})
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "rt_cache.npz", **data)
    sha = hashlib.sha256((out / "rt_cache.npz").read_bytes()).hexdigest()
    (out / "rt_cache.npz.sha256").write_text(f"{sha}  rt_cache.npz\n")
    agree = [abs(c["rt"] - c["expected"]) < 1e-12 for c in checks["label_vs_rt_anchor_fraction"]]
    meta = rv.manifest(cfg, "rt", {"dry_run": dry, "rt_calls": rt_calls[0], "wall_s": time.time() - t_start,
                                   "n_test_placements_run": len(test), "n_eligible_offset_grid": len(elig),
                                   "rt_cache_sha256": sha, "onset_blockers": blockers,
                                   "label_counts": {LABELS[k]: int(np.sum(np.asarray(labels) == k)) for k in LABELS},
                                   "label_rt_agreement": {"agree": int(sum(agree)), "n": len(agree)},
                                   "checks": checks})
    (out / "rt_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"rt_calls": rt_calls[0], "wall_s": time.time() - t_start, "labels": meta["label_counts"],
                      "label_rt_agreement": meta["label_rt_agreement"],
                      "fast_vs_direct_nmse": [c["nmse"] for c in checks["fast_vs_direct_nmse"]],
                      "determinism": all(c["bit_identical"] for c in checks["determinism"])}))


# --------------------------------------------------------------------------- #
# Detectors (template banks and slow-time energy) with an optional reference error
# --------------------------------------------------------------------------- #
class Det:
    def __init__(self, name, bank, s2):
        self.name, self.bank, self.s2 = name, bank, s2

    def stat(self, X):
        if self.bank is None:
            return np.sum(np.abs(X) ** 2, axis=0) / self.s2
        return core.window_stat(self.bank, X, self.s2)

    def miss(self, sig, onset, thr, amps, X, cfg):
        if self.bank is None:
            return energy_miss(sig, onset, self.s2, thr, amps, X, cfg)
        return core.miss_curve(sig, onset, self.bank, self.s2, thr, amps, X, cfg)


def make_dets(fc, cfg, design, s2):
    L = cfg["slow_time"]["window_samples"]
    Bb = core.bank_from_params(fc, core.baseline_params(cfg), cfg)
    d = {"baseline": Det("baseline", Bb, s2),
         "base_plus_dc": Det("base_plus_dc", np.vstack([Bb, np.ones((1, L), complex)]), s2),
         "redesigned": Det("redesigned", core.bank_from_params(fc, np.asarray(design["redesigned_params"]), cfg), s2),
         "energy": Det("energy", None, s2)}
    pc = design["per_carrier_params"].get(f"{fc / 1e9:g}")
    if pc is not None:
        d["per_carrier"] = Det("per_carrier", core.bank_from_params(fc, np.asarray(pc), cfg), s2)
    return d


def null_stats(det, L, k, n, ss, chunk=100_000):
    rng = np.random.default_rng(ss)
    out = np.empty(n)
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        X = core._cn(rng, (L, m), det.s2)
        if k:
            X = X + core._cn(rng, (1, m), det.s2 / k)
        out[i:i + m] = det.stat(X)
    return out


def event_fa(det, thr, cfg, ss, chunk_windows=50_000):
    """As core.event_false_alarms, for any detector statistic (perfect reference)."""
    st = cfg["slow_time"]
    L, H = st["window_samples"], st["hop_samples"]
    hours = cfg["detector"]["fa_stream_hours"]
    n_win = int(round(hours * 3600.0 * st["prf_hz"] / H))
    rng = np.random.default_rng(ss)
    tail = core._cn(rng, (L - H,), det.s2)
    prev, crossings, events, done = False, 0, 0, 0
    while done < n_win:
        m = min(chunk_windows, n_win - done)
        x = np.concatenate([tail, core._cn(rng, (m * H,), det.s2)])
        idx = np.arange(m)[:, None] * H + np.arange(L)[None, :]
        c = det.stat(x[idx].T) > thr
        crossings += int(c.sum())
        events += int(c[0] and not prev) + int(np.sum(c[1:] & ~c[:-1]))
        prev = bool(c[-1])
        tail = x[-(L - H):]
        done += m
    return {"hours": hours, "windows": n_win, "window_crossings": crossings, "events": events,
            "events_per_hour": events / hours, "window_pfa": crossings / n_win}


def _calibrate_job(args):
    """Thresholds for one (carrier, detector) at every K; event FA and reference-error Pfa at K=inf."""
    fc, det_name, ci, di, cfg, out = args
    t0 = time.time()
    z = np.load(Path(out) / "rt_cache.npz")
    design = json.loads((Path(cfg["v3"]["design_run"]) / "analysis.json").read_text())
    s2 = float(z["s2"])
    det = make_dets(fc, cfg, design, s2)[det_name]
    ss = streams(cfg["master_seed"])
    d, L = cfg["detector"], cfg["slow_time"]["window_samples"]
    ks = cfg["v3"]["reference_frames"]
    nfc, nd, nk = len(cfg["carriers_hz"]), len(DETS), len(ks)
    j0 = (ci * nd + di) * nk
    cal_k = ss["calibration"].spawn(nfc * nd * nk)[j0:j0 + nk]
    val_k = ss["validation"].spawn(nfc * nd * nk)[j0:j0 + nk]
    ref_k = ss["pfa_ref"].spawn(nfc * nd * nk)[j0:j0 + nk]
    fa_k = ss["fa_stream"].spawn(nfc * nd)[ci * nd + di]
    pfa = d["pfa_per_window"]
    res = {}
    for i, k in enumerate(ks):
        thr = float(np.quantile(null_stats(det, L, k, d["n_calibration_windows"], cal_k[i]), 1.0 - pfa))
        fa = int(np.sum(null_stats(det, L, k, d["n_validation_windows"], val_k[i]) > thr))
        lo, hi = core.wilson(fa, d["n_validation_windows"])
        res[kname(k)] = {"threshold": thr, "calibration_N": d["n_calibration_windows"],
                         "validation_N": d["n_validation_windows"], "false_alarms": fa,
                         "pfa": fa / d["n_validation_windows"], "wilson_ci95": [lo, hi], "accept": bool(lo <= pfa <= hi)}
    thr_inf = res["inf"]["threshold"]
    res["inf"]["events"] = event_fa(det, thr_inf, cfg, fa_k)
    n_ref = cfg["v3"]["pfa_ref_check_windows"]
    res["inf"]["pfa_with_reference_error"] = {}
    for i, k in enumerate(ks):
        if k is None:
            continue
        fa = int(np.sum(null_stats(det, L, k, n_ref, ref_k[i]) > thr_inf))
        res["inf"]["pfa_with_reference_error"][kname(k)] = {"pfa": fa / n_ref, "wilson_ci95": list(core.wilson(fa, n_ref)),
                                                            "ratio_to_nominal": fa / n_ref / pfa}
    print(f"[cal {fc / 1e9:g} GHz {det_name}] {time.time() - t0:.0f} s", flush=True)
    return fc, det_name, res


def _curves_job(args):
    """Miss curves for one (carrier, deadline, K) over every detector; at 200 ms / K=inf also the
    point targets, glass-only onsets and the F2 spot check."""
    fc, ci, deadline, k, thr, cfg, out = args
    t0 = time.time()
    cfg = copy.deepcopy(cfg)
    cfg["slow_time"]["deadline_s"] = deadline
    z = np.load(Path(out) / "rt_cache.npz")
    design = json.loads((Path(cfg["v3"]["design_run"]) / "analysis.json").read_text())
    s2 = float(z["s2"])
    dets = make_dets(fc, cfg, design, s2)
    names = [d for d in DETS if d in dets]
    ss = streams(cfg["master_seed"])
    det, st = cfg["detector"], cfg["slow_time"]
    lim = cfg["point_targets"]["max_eirp_above_nominal_db"]
    sig, lab = z[f"test_sig_{fc}"], z["test_label"]
    vis, glass = lab == 0, lab == 1
    on = z["m_test"].reshape(-1, 4)[:, 3]
    nfc = len(cfg["carriers_hz"])
    noise_ss = ss["noise"].spawn(nfc)[ci].spawn(len(sig))
    rerr_ss = ss["ref_error"].spawn(nfc)[ci].spawn(len(sig))
    T, n = det["n_noise_trials"], st["record_samples"]
    Bb = dets["baseline"].bank

    def noise(i):
        r = np.random.default_rng(noise_ss[i])
        X = (r.standard_normal((n, T)) + 1j * r.standard_normal((n, T))) * math.sqrt(s2 / 2)
        if k is not None:
            X = X + core._cn(np.random.default_rng(rerr_ss[i]), (1, T), s2) / math.sqrt(k)
        return X

    def run(sigs, idx, sel, grid0):
        """Curves for onsets idx (records sigs[idx]); refine around the crossing of the mean over sel."""
        def evaluate(grids):
            P = {d: np.zeros((len(idx), len(grids[d]))) for d in grids}
            for jj, i in enumerate(idx):
                X = noise(i)
                for d, g in grids.items():
                    if len(g):
                        P[d][jj] = dets[d].miss(sigs[i], on[i], thr[d], 10 ** (np.asarray(g) / 20), X, cfg)
            return P
        coarse = grid0
        Pc = evaluate({d: coarse for d in names})
        while any(Pc[d][sel].mean(0)[-1] > 0.10 for d in names) and coarse[-1] < lim:
            ext = coarse[-1] + det["eirp_grid_step_db"] * np.arange(1, 9)
            Pe = evaluate({d: ext for d in names})
            Pc = {d: np.concatenate([Pc[d], Pe[d]], axis=1) for d in names}
            coarse = np.concatenate([coarse, ext])
        return rv._refine(Pc, evaluate, sel, coarse, names, cfg, extra_db=(lim,))[:2]

    def grid_from(sigs, idx):
        nf = np.array([core.noisefree_required_db(sigs[i], on[i], Bb, s2, thr["baseline"], cfg) for i in idx])
        nf = nf[np.isfinite(nf)]
        return float(np.percentile(nf, 90)) + np.arange(det["eirp_grid_span_db"][0],
                                                        det["eirp_grid_span_db"][1] + 1e-9, det["eirp_grid_step_db"])

    allidx = np.arange(len(sig))
    grids, P = run(sig, allidx, vis, grid_from(sig, np.flatnonzero(vis)))
    res = {"fc": fc, "deadline": deadline, "K": kname(k), "grids": grids, "P": P}
    if deadline == 0.2 and k is None:
        gi = np.flatnonzero(glass)
        if gi.size:
            gg, gP = run(sig, gi, np.ones(gi.size, bool), grid_from(sig, gi))
            res["glass"] = {"idx": gi, "grids": gg, "P": gP}
        res["capture_loss_db"] = {d: core.capture_loss_db(dets[d].bank, sig[vis], on[vis], cfg).tolist()
                                  for d in names if dets[d].bank is not None}
        res["constant_share"] = core.constant_term_share(sig[vis], on[vis], cfg).tolist()
        pt_grid = np.append(grid_from(sig, np.flatnonzero(vis)), lim)
        kper = cfg["k_test_motions_per_placement"]
        res["point_targets"] = {}
        for m in cfg["point_targets"]["models"]:
            recs = z[f"pt_{m}_{fc}"]
            Pm = np.array([dets["baseline"].miss(recs[i // kper], on[i], thr["baseline"], 10 ** (pt_grid / 20),
                                                 noise(i), cfg) for i in allidx])
            res["point_targets"][m] = {"eirp_db_re_nominal": pt_grid.tolist(), "mean_miss": Pm.mean(0).tolist(),
                                       "max_residual_signal_energy": float(np.max(np.abs(recs) ** 2))}
        if f"test_sig_F2_{fc}" in z.files:
            sf2 = z[f"test_sig_F2_{fc}"]
            g2, P2 = run(sf2, allidx, vis, grid_from(sf2, np.flatnonzero(vis)))
            res["f2_spot"] = {"grids": g2, "P": P2}
    print(f"[curves {fc / 1e9:g} GHz {deadline * 1e3:g} ms K={kname(k)}] {time.time() - t0:.0f} s", flush=True)
    return res


def _analyze_stage(cfg, out, dry):
    t_start = time.time()
    fcs = cfg["carriers_hz"]
    with ProcessPoolExecutor(min(32, len(fcs) * len(DETS))) as ex:
        cal_jobs = [(fc, d, ci, di, cfg, str(out)) for ci, fc in enumerate(fcs) for di, d in enumerate(DETS)
                    if d != "per_carrier" or f"{fc / 1e9:g}" in json.loads(
                        (Path(cfg["v3"]["design_run"]) / "analysis.json").read_text())["per_carrier_params"]]
        cal = {}
        for fc, d, r in ex.map(_calibrate_job, cal_jobs):
            cal.setdefault(fc, {})[d] = r
    curve_jobs = [(fc, ci, dl, k, {d: cal[fc][d][kname(k)]["threshold"] for d in cal[fc]}, cfg, str(out))
                  for ci, fc in enumerate(fcs) for dl in cfg["v3"]["deadlines_s"] for k in cfg["v3"]["reference_frames"]]
    with ProcessPoolExecutor(min(32, len(curve_jobs))) as ex:
        res = list(ex.map(_curves_job, curve_jobs))
    curves, extras = {}, {}
    for r in res:
        tag = f"{r['fc']}_{r['deadline'] * 1e3:g}ms_K{r['K']}"
        for d in r["grids"]:
            curves[f"grid_{tag}_{d}"], curves[f"P_{tag}_{d}"] = r["grids"][d], r["P"][d]
        if "glass" in r:
            curves[f"glass_idx_{r['fc']}"] = r["glass"]["idx"]
            for d in r["glass"]["grids"]:
                curves[f"glass_grid_{r['fc']}_{d}"] = r["glass"]["grids"][d]
                curves[f"glass_P_{r['fc']}_{d}"] = r["glass"]["P"][d]
        if "f2_spot" in r:
            for d in r["f2_spot"]["grids"]:
                curves[f"F2spot_grid_{r['fc']}_{d}"] = r["f2_spot"]["grids"][d]
                curves[f"F2spot_P_{r['fc']}_{d}"] = r["f2_spot"]["P"][d]
        if "point_targets" in r:
            extras[f"{r['fc'] / 1e9:g}"] = {k_: r[k_] for k_ in ("capture_loss_db", "constant_share", "point_targets")}
    np.savez_compressed(out / "miss_curves.npz", **curves)
    res_json = {"manifest": rv.manifest(cfg, "analyze", {"wall_s": time.time() - t_start,
                                                         "miss_curves_sha256": hashlib.sha256(
                                                             (out / "miss_curves.npz").read_bytes()).hexdigest()}),
                "calibration": {f"{fc / 1e9:g}": cal[fc] for fc in fcs}, "per_carrier_extras": extras}
    (out / "analysis.json").write_text(json.dumps(res_json, indent=2))
    print(f"analyze done, {time.time() - t_start:.0f} s")


# --------------------------------------------------------------------------- #
# Stage numbers (CPU only, deterministic)
# --------------------------------------------------------------------------- #
def build_numbers(cfg, out):
    a = json.loads((out / "analysis.json").read_text())
    cv = np.load(out / "miss_curves.npz")
    z = rt_small(out)
    rtm = json.loads((out / "rt_manifest.json").read_text())
    ss = streams(cfg["master_seed"])
    bc = cfg["bootstrap"]
    pc = bc["ci_percent"]
    fcs = cfg["carriers_hz"]
    f0 = fcs[0]
    lab, pid = z["test_label"], z["test_pid"]
    vis, glass = lab == 0, lab == 1
    n_place = len(z["test_xy"])
    W = core.resample_weights(ss["bootstrap"], n_place, bc["n_resamples"])
    shift = {fc: float(z["f1_rx_gain_db"]) - float(z[f"f2_tx_gain_db_{fc}"]) for fc in fcs}
    e0 = {fc: float(z[f"eirp0_{fc}"]) for fc in fcs}
    lim = cfg["point_targets"]["max_eirp_above_nominal_db"]
    req, boot = {}, {}

    def put(key, g, P, sel, fc, idx_pid=pid):
        S, C = core.placement_sums(P, sel, idx_pid, n_place)
        pt = core.crossing(S.sum(0) / C.sum(), g)
        req[key] = None if pt is None else e0[fc] + pt
        boot[key] = e0[fc] + core.boot_crossings(S, C, g, W)

    for fc in fcs:
        for dl in cfg["v3"]["deadlines_s"]:
            for k in cfg["v3"]["reference_frames"]:
                for d in DETS:
                    t = f"{fc}_{dl * 1e3:g}ms_K{kname(k)}_{d}"
                    if f"grid_{t}" in cv.files:
                        put((d, kname(k), dl, fc), cv[f"grid_{t}"], cv[f"P_{t}"], vis, fc)

    def paired(k1, k2, s=0.0):
        if req.get(k1) is None or req.get(k2) is None:
            return {"value_db": None, "ci95_db": [None, None]}
        return {"value_db": req[k1] - req[k2] + s, "ci95_db": core.pct_ci(boot[k1] - boot[k2] + s, pc)}

    rows, sav, extra = [], [], []
    for (d, kk, dl, fc), v in req.items():
        for reg, s in (("F1", 0.0), ("F2", shift[fc])):
            rows.append({"regime": reg, "detector": d, "K": kk, "deadline_s": dl, "carrier_ghz": fc / 1e9,
                         "required_eirp_dbm": None if v is None else v + s,
                         "ci95_dbm": core.pct_ci(boot[(d, kk, dl, fc)] + s, pc),
                         "n_resamples_without_crossing": int(np.sum(~np.isfinite(boot[(d, kk, dl, fc)])))})
        if d != "baseline":
            sav.append({"detector": d, "K": kk, "deadline_s": dl, "carrier_ghz": fc / 1e9,
                        **paired(("baseline", kk, dl, fc), (d, kk, dl, fc)),
                        "note": "positive = detector needs less EIRP than the baseline bank (same K, deadline)"})
        if fc != f0 and (d, kk, dl, f0) in req:
            for reg in ("F1", "F2"):
                s = 0.0 if reg == "F1" else shift[fc] - shift[f0]
                extra.append({"regime": reg, "detector": d, "K": kk, "deadline_s": dl, "carrier_ghz": fc / 1e9,
                              **paired((d, kk, dl, fc), (d, kk, dl, f0), s),
                              "lambda2_term_db": 20 * math.log10(fc / f0)})
    # glass-only onsets with transmission (200 ms, K = inf)
    glass_rows = []
    for fc in fcs:
        if f"glass_idx_{fc}" not in cv.files:
            continue
        gi = cv[f"glass_idx_{fc}"]
        for d in DETS:
            if f"glass_grid_{fc}_{d}" not in cv.files:
                continue
            g, P = cv[f"glass_grid_{fc}_{d}"], cv[f"glass_P_{fc}_{d}"]
            put(("glass", d, fc), g, P, np.ones(len(gi), bool), fc, pid[gi])
            jl = int(np.argmin(np.abs(g - lim)))
            glass_rows.append({"carrier_ghz": fc / 1e9, "detector": d, "n_glass_only_onsets": int(len(gi)),
                               "n_placements": int(len(set(pid[gi].tolist()))),
                               "required_eirp_dbm": req[("glass", d, fc)],
                               "ci95_dbm": core.pct_ci(boot[("glass", d, fc)], pc),
                               "n_resamples_without_crossing": int(np.sum(~np.isfinite(boot[("glass", d, fc)]))),
                               "search_limit_dbm": e0[fc] + lim,
                               "n_detected_at_search_limit": int(np.sum(P[:, jl] <= 0.10)),
                               "n_detected_at_visible_requirement": None if req.get((d, "inf", 0.2, fc)) is None else
                               int(np.sum(np.array([np.interp(req[(d, "inf", 0.2, fc)] - e0[fc], g, P[j])
                                                    for j in range(len(gi))]) <= 0.10)),
                               "extra_vs_visible_same_carrier": paired(("glass", d, fc), (d, "inf", 0.2, fc))})
    spot = {}
    f2c = cfg["f2_spot_check_carrier_hz"]
    for d in DETS:
        if f"F2spot_grid_{f2c}_{d}" in cv.files:
            g, P = cv[f"F2spot_grid_{f2c}_{d}"], cv[f"F2spot_P_{f2c}_{d}"]
            S, C = core.placement_sums(P, vis, pid, n_place)
            pt = core.crossing(S.sum(0) / C.sum(), g)
            direct = None if pt is None else float(z[f"f2_eirp0_{f2c}"]) + pt
            sh = None if req.get((d, "inf", 0.2, f2c)) is None else req[(d, "inf", 0.2, f2c)] + shift[f2c]
            spot[d] = {"direct_f2_dbm": direct, "shifted_f2_dbm": sh,
                       "difference_db": None if None in (direct, sh) else direct - sh}
    cal_rows, ev_rows, pref_rows = [], [], []
    for fc in fcs:
        for d, byk in a["calibration"][f"{fc / 1e9:g}"].items():
            for kk, c in byk.items():
                cal_rows.append({"carrier_ghz": fc / 1e9, "detector": d, "K": kk,
                                 **{x: c[x] for x in ("threshold", "calibration_N", "validation_N", "false_alarms",
                                                      "pfa", "wilson_ci95", "accept")}})
            ev_rows.append({"carrier_ghz": fc / 1e9, "detector": d, **byk["inf"]["events"]})
            for kk, v in byk["inf"]["pfa_with_reference_error"].items():
                pref_rows.append({"carrier_ghz": fc / 1e9, "detector": d, "K": kk, **v})
    ex = a["per_carrier_extras"]
    cap_rows = [{"carrier_ghz": float(f), "detector": d, "mean": float(np.mean(v)), "p90": float(np.percentile(v, 90))}
                for f, x in ex.items() for d, v in x["capture_loss_db"].items()]
    ct_rows = [{"carrier_ghz": float(f), "median_share": float(np.median(x["constant_share"])),
                "floor_db_from_median": float(-10 * np.log10(np.median(x["constant_share"])))} for f, x in ex.items()]
    pt_rows = [{"carrier_ghz": float(f), "model": m, "max_tested_eirp_dbm": e0[float(f) * 1e9] + max(v["eirp_db_re_nominal"]),
                "miss_min": min(v["mean_miss"]), "miss_max": max(v["mean_miss"]),
                "max_residual_signal_energy": v["max_residual_signal_energy"]}
               for f, x in ex.items() for m, v in x["point_targets"].items()]
    loc_rows = rtm["checks"]["localization"]
    nm = rtm["checks"]["fast_vs_direct_nmse"]
    numbers = {
        "required_eirp_dbm": rows, "savings_vs_baseline_db": sav, "extra_eirp_vs_3p5_db": extra,
        "glass_only_with_transmission": glass_rows, "f2_spot_check": spot,
        "false_alarm_calibration": {"cells": cal_rows, "n_not_accepted": int(sum(not r["accept"] for r in cal_rows))},
        "event_false_alarms": ev_rows, "pfa_with_reference_error_at_perfect_reference_thresholds": pref_rows,
        "test_capture_loss_db": cap_rows, "constant_term": ct_rows, "point_targets": pt_rows,
        "onsets": {"n_total": int(len(lab)), "n_visible": int(vis.sum()), "n_glass_only": int(glass.sum()),
                   "n_other_blocked": int(np.sum(lab == 2)), "label_rt_agreement": rtm["label_rt_agreement"]},
        "placements": {"n_test": n_place, "n_eligible_offset_grid": rtm["n_eligible_offset_grid"],
                       "k_test_motions": cfg["k_test_motions_per_placement"]},
        "localization": {"n": len(loc_rows), "within_one_bin": int(sum(abs(r["r_hat"] - r["true_bin"]) <= 1
                                                                       for r in loc_rows)),
                         "exact": int(sum(r["r_hat"] == r["true_bin"] for r in loc_rows))},
        "fast_vs_direct_nmse": {"n_onsets": len(nm), "max": max((r["nmse"] for r in nm), default=None)},
        "antennas": {"f2_tx_gain_db": {f"{fc / 1e9:g}": float(z[f"f2_tx_gain_db_{fc}"]) for fc in fcs},
                     "f2_receive_gain_shift_db": {f"{fc / 1e9:g}": shift[fc] for fc in fcs}},
        "bootstrap": {"n_resamples": bc["n_resamples"], "ci_percent": pc, "unit": "placement",
                      "paired": "one resample index set shared by all cells"},
        "rt_cache_sha256": rtm["rt_cache_sha256"]}
    return {"release": "rerun-v3", "primary_estimand": "required EIRP: lowest EIRP with mean deadline miss <= 0.10 "
            "over visible onsets (200 noise trials; 1 dB grid refined to 0.25 dB)",
            "sources": {"analysis_sha256": hashlib.sha256((out / "analysis.json").read_bytes()).hexdigest(),
                        "miss_curves_sha256": hashlib.sha256((out / "miss_curves.npz").read_bytes()).hexdigest()},
            "numbers": numbers}


def _numbers_stage(cfg, out, dry):
    res = build_numbers(cfg, out)
    res["manifest"] = rv.manifest(cfg, "numbers", {"dry_run": dry})
    again = build_numbers(cfg, out)
    res["numbers"]["determinism_numbers_stage"] = json.dumps(again["numbers"], sort_keys=True) == json.dumps(
        {k: v for k, v in res["numbers"].items() if k != "determinism_numbers_stage"}, sort_keys=True)
    (out / "manuscript_numbers.json").write_text(json.dumps(res, indent=2))
    print("numbers written; deterministic:", res["numbers"]["determinism_numbers_stage"])


def _gates_stage(cfg, out, dry):
    """G1-G9 from validation_gates (out/gates_rt), G10 localization and G11 determinism of this run."""
    rv._gates_stage(cfg, out, dry)
    p = out / "gates.json"
    g = json.loads(p.read_text())
    g["release"] = "rerun-v3"
    p.write_text(json.dumps(g, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description="rerun v3 (review-3 production run)")
    ap.add_argument("stage", choices=("rt", "analyze", "numbers", "gates", "summary"))
    ap.add_argument("--config", default="configs/rerun_v3.yaml")
    ap.add_argument("--out", default="results/rerun_v3/full")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = core.load_cfg(a.config)
    if a.dry_run:
        cfg = copy.deepcopy(cfg)
        cfg["carriers_hz"] = cfg["dry_run"]["carriers_hz"]
    if a.stage == "summary":
        print("rt_summary.npz:", write_rt_summary(a.out))
        return
    {"rt": _rt_stage, "analyze": _analyze_stage, "numbers": _numbers_stage,
     "gates": _gates_stage}[a.stage](cfg, Path(a.out), a.dry_run)


if __name__ == "__main__":
    main()
