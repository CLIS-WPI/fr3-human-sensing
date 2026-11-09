"""Rerun v2: deadline detection with sliding windows, random motions, disjoint dev/test.

Stages
  rt       (Sionna, GPU) noise-free detector-input slow-time records for every development
           and test onset (R1-A), stationary point targets (R0, R1-P), the direct F2 spot
           check, localization rows (G10), direct-RT checks of the fast method and a
           bit-for-bit re-simulation (G11).
  analyze  (numpy) banks from development data, coverage criterion, per-window thresholds,
           event-level false alarms, deadline-miss curves (1 dB grid refined to 0.25 dB at
           the crossing), hidden-onset status, point-target miss. Writes analysis.json and
           miss_curves.npz.
  numbers  (numpy, laptop) placement bootstrap with paired intervals; writes
           manuscript_numbers.json.
  gates    merges G1-G9 (validation_gates) with G10/G11 from this run into gates.json.

Target paths are direct only (RCSSolver: los=True, no specular reflection, no refraction);
room clutter is LoS + single specular reflections (PathSolver). See docs/RERUN_V2_METHOD.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fr3isac import geometry as geo
from fr3isac.experiments import rerun_v2_core as core

BANKS = ("baseline", "redesigned", "per_carrier")
LEGACY = {"baseline": "D0", "redesigned": "Dfix", "per_carrier": "D1"}  # keys used by tools/


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:  # noqa: BLE001 - git may be absent
        return None


def manifest(cfg, stage, extra=None) -> dict:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    dirty = _git("status", "--porcelain", "--", "src", "configs", "tools")
    return {"stage": stage, "utc": datetime.now(UTC).isoformat(),
            "config_version": cfg["version"],
            "config_hash": hashlib.sha256(blob.encode()).hexdigest(),
            "git_sha": _git("rev-parse", "HEAD") or os.environ.get("FR3ISAC_GIT_SHA", "unknown"),
            "git_dirty_src_configs_tools": None if dirty is None else bool(dirty),
            "container_image": os.environ.get("FR3ISAC_CONTAINER_IMAGE", "unknown"),
            "container_digest": os.environ.get("FR3ISAC_CONTAINER_DIGEST", "unknown"),
            "master_seed": cfg["master_seed"],
            "scene_hash": hashlib.sha256(Path(cfg["scene"]).read_bytes()).hexdigest(),
            "target_model": "R1-A torso+right_forearm (geometry.SEGMENT_OFFSETS), onset_arm_trajectory",
            "rcs_model": {"torso_m2": cfg["rt"]["torso_rcs_m2"], "forearm_m2": cfg["rt"]["forearm_rcs_m2"]},
            "solver": cfg["rt"], **(extra or {})}


def _carriers(cfg, dry):
    return cfg["dry_run"]["carriers_hz"] if dry else cfg["carriers_hz"]


def _split(cfg, dry):
    elig = core.eligible_grid(geo.load_descriptor(cfg["scene"]), cfg["grid_step_m"])
    dev, test = core.split_placements(elig, cfg["n_test_placements"])
    core.assert_disjoint(dev, test)
    n = cfg["dry_run"]["n_test_placements"]
    test_run = test[:: max(1, len(test) // n)][:n] if dry else test
    return elig, dev, test, test_run


def _motions(cfg, dev, test):
    ss = core.seed_sequences(cfg["master_seed"])
    m_dev = core.motions_per_placement(ss["dev_motions"], dev, cfg["k_dev_motions_per_placement"], cfg)
    m_test = core.motions_per_placement(ss["test_motions"], test, cfg["k_test_motions_per_placement"], cfg)
    return m_dev, m_test


def _visible(root, motion, times, descriptor):
    bs = geo.bs_position(descriptor)
    blk = geo.blockers(descriptor)
    for s in geo.onset_arm_trajectory(root, times, *motion):
        fa = np.asarray(s.forearm_position, dtype=float)
        if any(geo.segment_aabb_intersect(bs, fa, o["center_m"], o["size_m"]) for o in blk):
            return False
    return True


def _lower95(phat, n, z=1.6448536269514722):
    """One-sided 95% Wilson lower bound (the G10 rule of the original pipeline)."""
    d = 1 + z * z / n
    c = phat + z * z / (2 * n)
    h = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (c - h) / d


# --------------------------------------------------------------------------- #
# Stage rt
# --------------------------------------------------------------------------- #
def _rt_stage(cfg, out, dry):
    from sionna.rt.rcs import RCSSolver

    from fr3isac.config import FairnessConfig
    from fr3isac.experiments import localization as loc
    from fr3isac.fairness.normalize import normalize
    from fr3isac.sensing.rt_channel import detector_input_scale, thermal_noise_variance
    from fr3isac.sensing.rt_localize import range_m_to_bin
    from fr3isac.sensing.rt_slowtime import interp_complex_series, path_records, synthesize_cfr

    t_start = time.time()
    rt_calls = [0]
    elig, dev, test, test_run = _split(cfg, dry)
    assert elig == loc.eligible_scenarios(step_m=cfg["grid_step_m"])[0], "eligibility mismatch"
    m_dev, m_test = _motions(cfg, dev, test)
    test_idx = [test.index(p) for p in test_run]
    times = core.record_times(cfg)
    st = cfg["slow_time"]
    dt_anchor = st["window_samples"] / st["prf_hz"] / st["anchors_per_window"]
    offsets = (np.arange(loc.NSC) - loc.NSC // 2) * loc.SCS
    c2 = detector_input_scale(loc.P_TX_W, loc.NSC)
    desc = geo.load_descriptor(cfg["scene"])
    bs = geo.bs_position(desc)
    f2c = cfg["f2_spot_check_carrier_hz"]
    assert abs(10 * math.log10(loc.P_TX_W * 1e3) - FairnessConfig().conducted_power_dbm) < 1e-9

    def static_ref(fc, model, root, gain, bg):
        rt_calls[0] += 1
        stc = loc._solve_cfr(fc, model, geo.static_sample(root), offsets, gain)
        r_hat, _ = loc.localize_body_bin(bg[:, None] + stc[:, None], bg[:, None], loc.OCC_BW,
                                         desc["range_search_min_m"], desc["range_search_max_m"])
        # static reference: one complex value at the range bin (constant in slow time)
        return stc, r_hat, complex(c2 * loc._body_vector(bg, stc, r_hat)[0])

    def onset_record(fc, root, motion, gain, bg, r_hat, xr):
        """Fast method; returns (record, fraction of anchors with a forearm path, raw)."""
        A, f, phi, t0 = motion
        anchors = t0 + np.arange(int(math.ceil((times[-1] - t0) / dt_anchor)) + 1) * dt_anchor
        ta_amp, fa_amp = [], []
        for ta in anchors:
            s = geo.onset_arm_trajectory(root, [ta], A, f, phi, t0)[0]
            rt_calls[0] += 1
            paths = RCSSolver(deterministic=True)(loc.build_room(fc, loc._targets_for("R1-A", s)),
                                                  max_depth=1, los=True, specular_reflection=False,
                                                  refraction=False, seed=1)
            r_t = float(np.linalg.norm(np.asarray(s.torso_position) - bs))
            r_f = float(np.linalg.norm(np.asarray(s.forearm_position) - bs))
            a_t = a_f = 0.0 + 0.0j
            for rec in path_records(paths):
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
        frac = float(np.mean(np.abs(np.asarray(fa_amp)) > 0))
        raw = c2 * loc._body_vector(bg, cfr, r_hat)
        # noise-free pre-motion reference from the same synthesis (sample 0 precedes every
        # onset); the RT-static reference ``xr`` differs only by float32-delay residue
        return raw - raw[0], frac, raw

    def direct_record(fc, root, motion, gain, bg, r_hat, xr):
        cols = []
        for s in geo.onset_arm_trajectory(root, times, *motion):
            rt_calls[0] += 1
            cols.append(loc._solve_cfr(fc, "R1-A", s, offsets, gain))
        return c2 * loc._body_vector(bg, np.stack(cols, axis=1), r_hat) - xr

    data = {"times": times,
            "s2": thermal_noise_variance(loc.SCS, noise_figure_db=loc.NOISE_FIGURE_DB) / loc.NSC,
            "f1_rx_gain_db": FairnessConfig(regime="F1").bs_rx_gain_db}
    checks = {"pre_onset_residual": [], "fast_vs_direct_nmse": [], "localization": [],
              "determinism": []}
    n_val = cfg["rt"].get("validation_onsets_per_carrier", 0)
    keep = {}
    for fc in cfg["carriers_hz"]:
        need_test = fc in _carriers(cfg, dry)
        bg = loc._bg_cfr(fc, offsets)
        gain = loc._regime_gain_db(fc, cfg["antenna_regime"])
        # nominal EIRP = conducted power + the transmit gain actually simulated (the F1
        # element-count quantization gives 10.21 dBi, so 40.21 dBm, not the 40.0 dBm target)
        fcfg = FairnessConfig(regime=cfg["antenna_regime"])
        data[f"eirp0_{fc}"] = fcfg.conducted_power_dbm + normalize(fcfg, fc).bs.gain_db
        nf2 = normalize(FairnessConfig(regime="F2"), fc).bs
        data[f"f2_tx_gain_db_{fc}"] = nf2.gain_db
        data[f"f2_eirp0_{fc}"] = FairnessConfig(regime="F2").conducted_power_dbm + nf2.gain_db
        keep[fc] = (bg, gain)
        sig, vis = [], []
        for pi, xy in enumerate(dev):
            root = (xy[0], xy[1], geo.TARGET_HEIGHT_M)
            _, r_hat, xr = static_ref(fc, "R1-A", root, gain, bg)
            for mot in m_dev[pi]:
                sig.append(onset_record(fc, root, mot, gain, bg, r_hat, xr)[0])
                vis.append(_visible(root, mot, times, desc))
        data[f"dev_sig_{fc}"], data[f"dev_vis_{fc}"] = np.asarray(sig), np.asarray(vis)
        if not need_test:
            print(f"rt {fc / 1e9:g} GHz (dev only) done, {rt_calls[0]} RT calls, "
                  f"{time.time() - t_start:.0f} s", flush=True)
            continue
        do_f2 = (not dry) and fc == f2c
        sig, vis, pid, frac, sig_f2 = [], [], [], [], []
        pts = {m: [] for m in cfg["point_targets"]["models"]}
        gain_f2 = 2.0 * nf2.gain_db  # same array transmits and receives under F2
        nv = 0
        for ti in test_idx:
            xy = test[ti]
            root = (xy[0], xy[1], geo.TARGET_HEIGHT_M)
            true_bin = range_m_to_bin(float(np.linalg.norm(np.asarray(root) - bs)), loc.NSC, loc.OCC_BW)
            _, r_hat, xr = static_ref(fc, "R1-A", root, gain, bg)
            checks["localization"].append({"fc_hz": fc, "placement": list(xy), "model": "R1-A",
                                           "r_hat": int(r_hat), "true_bin": int(true_bin)})
            if do_f2:
                _, r_hat2, xr2 = static_ref(fc, "R1-A", root, gain_f2, bg)
            for mot in m_test[ti]:
                s, fr, raw = onset_record(fc, root, mot, gain, bg, r_hat, xr)
                t0 = mot[3]
                pre = times <= t0
                e_post = float(np.sum(np.abs(s[~pre]) ** 2))
                checks["pre_onset_residual"].append(
                    float(np.sum(np.abs(s[pre]) ** 2) / e_post) if e_post > 0 else 0.0)
                # synthesized rest-pose return vs RT static solve (both noise-free)
                checks.setdefault("reference_mismatch_rel", []).append(
                    float(abs(raw[0] - xr) ** 2 / abs(xr) ** 2))
                v = _visible(root, mot, times, desc)
                if v and nv < n_val:
                    d = direct_record(fc, root, mot, gain, bg, r_hat, xr)
                    m = ~pre
                    f_ = raw - xr
                    checks["fast_vs_direct_nmse"].append(
                        {"fc_hz": fc, "placement": list(xy), "motion": mot.tolist(),
                         "nmse": float(np.sum(np.abs(f_[m] - d[m]) ** 2) / np.sum(np.abs(d[m]) ** 2))})
                    nv += 1
                sig.append(s)
                vis.append(v)
                pid.append(ti)
                frac.append(fr)
                if do_f2:
                    sig_f2.append(onset_record(fc, root, mot, gain_f2, bg, r_hat2, xr2)[0])
            for model in pts:
                stc, rh, xrp = static_ref(fc, model, root, gain, bg)
                checks["localization"].append({"fc_hz": fc, "placement": list(xy), "model": model,
                                               "r_hat": int(rh), "true_bin": int(true_bin)})
                rec = c2 * loc._body_vector(bg, np.repeat(stc[:, None], times.size, axis=1), rh) - xrp
                pts[model].append(rec)
        data[f"test_sig_{fc}"], data[f"test_vis_{fc}"] = np.asarray(sig), np.asarray(vis)
        data[f"test_pid_{fc}"], data[f"test_fafrac_{fc}"] = np.asarray(pid), np.asarray(frac)
        if do_f2:
            data[f"test_sig_F2_{fc}"] = np.asarray(sig_f2)
        for model, recs in pts.items():
            data[f"pt_{model}_{fc}"] = np.asarray(recs)
        print(f"rt {fc / 1e9:g} GHz done, {rt_calls[0]} RT calls, {time.time() - t_start:.0f} s", flush=True)

    # G11: re-simulate a few test onsets per carrier and compare bit-for-bit
    for fc in _carriers(cfg, dry):
        bg, gain = keep[fc]
        ti = test_idx[0]
        root = (test[ti][0], test[ti][1], geo.TARGET_HEIGHT_M)
        _, r_hat, xr = static_ref(fc, "R1-A", root, gain, bg)
        for j in range(cfg["g11_onsets_per_carrier"]):
            s2_ = onset_record(fc, root, m_test[ti][j], gain, bg, r_hat, xr)[0]
            ref = data[f"test_sig_{fc}"][j]
            checks["determinism"].append({"fc_hz": fc, "onset": j, "bit_identical": bool(np.array_equal(s2_, ref)),
                                          "max_abs_diff": float(np.max(np.abs(s2_ - ref)))})

    data["m_dev"], data["m_test"] = m_dev, m_test
    data["dev_xy"], data["test_xy"] = np.asarray(dev), np.asarray(test)
    data["test_idx"] = np.asarray(test_idx)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "rt_cache.npz", **data)
    wall = time.time() - t_start
    meta = manifest(cfg, "rt", {"dry_run": dry, "rt_calls": rt_calls[0], "wall_s": wall,
                                "n_dev_placements": len(dev), "n_test_placements_run": len(test_idx),
                                "rt_cache_sha256": hashlib.sha256((out / "rt_cache.npz").read_bytes()).hexdigest(),
                                "checks": checks})
    (out / "rt_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"rt_calls": rt_calls[0], "wall_s": wall,
                      "max_pre_onset_residual": max(checks["pre_onset_residual"], default=None),
                      "max_reference_mismatch_rel": max(checks.get("reference_mismatch_rel", [0.0])),
                      "fast_vs_direct_nmse": [c["nmse"] for c in checks["fast_vs_direct_nmse"]],
                      "determinism": all(c["bit_identical"] for c in checks["determinism"])}))


# --------------------------------------------------------------------------- #
# Stage analyze
# --------------------------------------------------------------------------- #
def _curves(sigs, ons, banks, thr, s2, coarse, noise_ss, cfg):
    """Miss curves on a coarse grid refined to the fine step around each bank's crossing.

    Noise: one continuous record per (onset, trial) from ``noise_ss[i]``; regenerated
    identically for every pass and shared by all banks (paired)."""
    det, st = cfg["detector"], cfg["slow_time"]
    T, n = det["n_noise_trials"], st["record_samples"]

    def noise(i):
        r = np.random.default_rng(noise_ss[i])
        return (r.standard_normal((n, T)) + 1j * r.standard_normal((n, T))) * math.sqrt(s2 / 2)

    def evaluate(grids):
        P = {b: np.zeros((len(sigs), len(grids[b]))) for b in grids}
        for i in range(len(sigs)):
            X = noise(i)
            for b, g in grids.items():
                if len(g):
                    P[b][i] = core.miss_curve(sigs[i], ons[i], banks[b], s2, thr[b], 10 ** (np.asarray(g) / 20), X, cfg)
        return P

    Pc = evaluate({b: coarse for b in banks})
    return Pc, evaluate


def _refine(Pc, evaluate, vis, coarse, banks, cfg, extra_db=()):
    det = cfg["detector"]
    fine, cr0 = {}, {}
    for b in banks:
        c = core.crossing(Pc[b][vis].mean(0), coarse)
        cr0[b] = c
        if c is None:
            fine[b] = list(extra_db)
            continue
        f = c + np.arange(-det["eirp_fine_halfwidth_db"], det["eirp_fine_halfwidth_db"] + 1e-9,
                          det["eirp_fine_step_db"])
        f = [x for x in f if np.min(np.abs(coarse - x)) > 1e-9]
        fine[b] = f + list(extra_db)
    Pf = evaluate(fine)
    grids, P = {}, {}
    for b in banks:
        g = np.concatenate([coarse, np.asarray(fine[b], dtype=float)])
        M = np.concatenate([Pc[b], Pf[b]], axis=1)
        o = np.argsort(g, kind="stable")
        grids[b], P[b] = g[o], M[:, o]
    return grids, P, cr0


def _analyze_stage(cfg, out, dry):
    t_start = time.time()
    z = np.load(out / "rt_cache.npz")
    ss = core.seed_sequences(cfg["master_seed"])
    m_dev, m_test = z["m_dev"], z["m_test"]
    dev_on = m_dev.reshape(-1, 4)[:, 3]
    st, det = cfg["slow_time"], cfg["detector"]
    s2 = float(z["s2"])
    all_fc, run_fc = cfg["carriers_hz"], _carriers(cfg, dry)
    rb = cfg["banks"]["redesigned"]
    lim_db = cfg["point_targets"]["max_eirp_above_nominal_db"]

    # ---- banks (development data only) -------------------------------------
    cand = core.candidate_params(ss["design_candidates"], cfg)
    caps, dev_items = {}, {}
    for fc in all_fc:
        vis = z[f"dev_vis_{fc}"]
        sigs = z[f"dev_sig_{fc}"][vis]
        dev_items[fc] = (sigs, dev_on[vis])
        caps[fc] = core.capture_matrix(core.bank_from_params(fc, cand, cfg), sigs, dev_on[vis], cfg)
    cov = core.coverage_search([caps[fc] for fc in all_fc], rb["coverage_tolerance_db"],
                               rb["n_templates"], rb["max_templates_search"])
    params = {"baseline": core.baseline_params(cfg), "redesigned": cand[cov["chosen"]]}
    percar, percar_cov = {}, {}
    for fc in run_fc:
        c = core.coverage_search([caps[fc]], rb["coverage_tolerance_db"], rb["n_templates"],
                                 rb["max_templates_search"])
        percar[fc] = cand[c["chosen"]]
        percar_cov[f"{fc / 1e9:g}"] = {k: v for k, v in c.items() if k != "chosen"}
    coverage = {"criterion": f"P90 capture loss on development onsets within {rb['coverage_tolerance_db']} dB "
                             "of the full candidate pool, at every carrier",
                "n_candidates": int(cand.shape[0]), "n_templates": rb["n_templates"],
                "per_carrier": {}, "smallest_size_meeting": cov["smallest_size_meeting"],
                "searched_up_to": cov["searched_up_to"], "per_carrier_variant": percar_cov}
    for i, fc in enumerate(all_fc):
        sigs, on = dev_items[fc]
        bl = core.capture_loss_db(core.bank_from_params(fc, params["baseline"], cfg), sigs, on, cfg)
        coverage["per_carrier"][f"{fc / 1e9:g}"] = {
            "n_dev_onsets_visible": int(len(sigs)),
            "pool_p90_loss_db": cov["pool_p90_loss_db"][i],
            "redesigned_p90_loss_db": cov["bank_p90_loss_db"][i],
            "baseline_p90_loss_db": float(np.percentile(bl, 90)),
            "meets": bool(cov["bank_p90_loss_db"][i] <= cov["pool_p90_loss_db"][i] + rb["coverage_tolerance_db"])}
    coverage["meets_all"] = all(v["meets"] for v in coverage["per_carrier"].values())

    def bank(name, fc):
        return core.bank_from_params(fc, percar[fc] if name == "per_carrier" else params[name], cfg)

    # ---- per carrier ------------------------------------------------------------
    cells, curves = [], {}
    nb = len(BANKS)
    cal_kids = ss["calibration"].spawn(len(all_fc) * nb)
    val_kids = ss["validation"].spawn(len(all_fc) * nb)
    fa_kids = ss["fa_stream"].spawn(len(all_fc) * nb)
    noise_par = ss["noise_trials"].spawn(len(all_fc))
    pt_models = cfg["point_targets"]["models"]
    f2c = cfg["f2_spot_check_carrier_hz"]
    for ci, fc in enumerate(all_fc):
        if fc not in run_fc:
            continue
        eirp0 = float(z[f"eirp0_{fc}"])
        sig, vis = z[f"test_sig_{fc}"], z[f"test_vis_{fc}"]
        pid_global = z[f"test_pid_{fc}"]
        order = {p: i for i, p in enumerate(z["test_idx"].tolist())}
        pid = np.array([order[p] for p in pid_global])
        on = m_test[z["test_idx"]].reshape(-1, 4)[:, 3]
        frac = z[f"test_fafrac_{fc}"]
        B = {b: bank(b, fc) for b in BANKS}
        cal = {}
        for bi, b in enumerate(BANKS):
            j = ci * nb + bi
            cal[b] = core.calibrate(B[b], s2, cfg, cal_kids[j], val_kids[j])
            cal[b]["events"] = core.event_false_alarms(B[b], s2, cal[b]["threshold"], cfg, fa_kids[j])
        thr = {b: cal[b]["threshold"] for b in BANKS}
        cap = {b: core.capture_loss_db(B[b], sig[vis], on[vis], cfg) for b in BANKS}
        share = core.constant_term_share(sig[vis], on[vis], cfg)
        floor = -10 * np.log10(1 - share)
        nf = np.array([core.noisefree_required_db(s, t0, B["baseline"], s2, thr["baseline"], cfg)
                       for s, t0 in zip(sig[vis], on[vis], strict=True)])
        base = float(np.percentile(nf, 90))
        coarse = base + np.arange(det["eirp_grid_span_db"][0], det["eirp_grid_span_db"][1] + 1e-9,
                                  det["eirp_grid_step_db"])
        noise_ss = noise_par[ci].spawn(len(sig))
        Pc, evaluate = _curves(sig, on, B, thr, s2, coarse, noise_ss, cfg)
        grids, P, cr0 = _refine(Pc, evaluate, vis, coarse, BANKS, cfg, extra_db=(lim_db,))
        # point targets: same chain, same noise, baseline bank; coarse grid + search limit
        pt_grid = np.append(coarse, lim_db)
        PT = {}
        kper = len(sig) // len(z["test_idx"])
        for m in pt_models:
            recs = z[f"pt_{m}_{fc}"]
            Pm = np.zeros((len(sig), pt_grid.size))
            for i in range(len(sig)):
                r = np.random.default_rng(noise_ss[i])
                n_ = st["record_samples"]
                X = (r.standard_normal((n_, det["n_noise_trials"]))
                     + 1j * r.standard_normal((n_, det["n_noise_trials"]))) * math.sqrt(s2 / 2)
                Pm[i] = core.miss_curve(recs[i // kper], on[i], B["baseline"], s2, thr["baseline"],
                                        10 ** (pt_grid / 20), X, cfg)
            PT[m] = {"eirp_tested_dbm": (eirp0 + pt_grid).tolist(), "mean_miss": Pm.mean(0).tolist(),
                     "max_residual_signal_energy": float(np.max(np.abs(recs) ** 2))}
        hid = ~vis
        hidden = {"n_hidden": int(hid.sum()),
                  "placements": sorted({int(p) for p in pid[hid]}),
                  "no_target_path": int(np.sum(hid & (frac == 0))),
                  "partial_or_weak_path": int(np.sum(hid & (frac > 0))),
                  "forearm_path_anchor_fraction": frac[hid].tolist()}
        if hid.any():
            nfh = np.array([core.noisefree_required_db(s, t0, B["baseline"], s2, thr["baseline"], cfg)
                            for s, t0 in zip(sig[hid], on[hid], strict=True)])
            hidden["baseline_noisefree_required_dbm"] = [None if not np.isfinite(x) else eirp0 + x for x in nfh]
            hidden["within_search_limit"] = int(np.sum(np.isfinite(nfh) & (nfh <= lim_db)))
            hidden["mean_miss_at_search_limit"] = {
                b: float(P[b][hid][:, int(np.argmin(np.abs(grids[b] - lim_db)))].mean()) for b in BANKS}
            if (~hid).any():
                hidden["median_extra_noisefree_vs_visible_db"] = float(
                    np.median(nfh[np.isfinite(nfh)]) - np.median(nf)) if np.isfinite(nfh).any() else None
        cell = {"carrier_ghz": fc / 1e9, "eirp0_dbm": eirp0, "n_onsets": int(len(sig)),
                "n_visible": int(vis.sum()), "n_placements": int(len(z["test_idx"])),
                "k_test_motions": int(kper), "search_limit_dbm": eirp0 + lim_db,
                "constant_term": {"median_share": float(np.median(share)),
                                  "p90_floor_db": float(np.percentile(floor, 90)),
                                  "median_floor_db": float(np.median(floor))},
                "hidden": hidden, "point_targets": PT, "banks": {}}
        for b in BANKS:
            cr = core.crossing(P[b][vis].mean(0), grids[b])
            cell["banks"][b] = {"calibration": cal[b], "coarse_crossing_db": cr0[b],
                                "required_eirp_dbm": None if cr is None else eirp0 + cr,
                                "test_capture_loss_db": {"mean": float(cap[b].mean()),
                                                         "p90": float(np.percentile(cap[b], 90)),
                                                         "max": float(cap[b].max())}}
            curves[f"grid_{fc}_{b}"], curves[f"P_{fc}_{b}"] = grids[b], P[b]
        curves[f"vis_{fc}"], curves[f"pid_{fc}"], curves[f"eirp0_{fc}"] = vis, pid, eirp0
        # direct F2 spot check (same noise as F1 at this carrier)
        if f"test_sig_F2_{fc}" in z.files and fc == f2c:
            sf2 = z[f"test_sig_F2_{fc}"]
            e2 = float(z[f"f2_eirp0_{fc}"])
            nf2 = np.array([core.noisefree_required_db(s, t0, B["baseline"], s2, thr["baseline"], cfg)
                            for s, t0 in zip(sf2[vis], on[vis], strict=True)])
            c2_ = float(np.percentile(nf2, 90)) + np.arange(det["eirp_grid_span_db"][0],
                                                            det["eirp_grid_span_db"][1] + 1e-9, det["eirp_grid_step_db"])
            Pc2, ev2 = _curves(sf2, on, B, thr, s2, c2_, noise_ss, cfg)
            g2, P2, _ = _refine(Pc2, ev2, vis, c2_, BANKS, cfg)
            for b in BANKS:
                curves[f"grid_F2spot_{fc}_{b}"], curves[f"P_F2spot_{fc}_{b}"] = g2[b], P2[b]
            curves[f"eirp0_F2spot_{fc}"] = e2
        cells.append(cell)
        print(f"analyze {fc / 1e9:g} GHz done, {time.time() - t_start:.0f} s", flush=True)

    np.savez_compressed(out / "miss_curves.npz", **curves)
    f2_shift = {f"{fc / 1e9:g}": float(z["f1_rx_gain_db"]) - float(z[f"f2_tx_gain_db_{fc}"]) for fc in all_fc}
    res = {"manifest": manifest(cfg, "analyze", {"dry_run": dry, "wall_s": time.time() - t_start,
                                                 "miss_curves_sha256": hashlib.sha256(
                                                     (out / "miss_curves.npz").read_bytes()).hexdigest()}),
           "coverage": coverage, "f2_receive_gain_shift_db": f2_shift,
           "f2_tx_gain_db": {f"{fc / 1e9:g}": float(z[f"f2_tx_gain_db_{fc}"]) for fc in all_fc},
           "redesigned_params": params["redesigned"].tolist(),
           "per_carrier_params": {f"{fc / 1e9:g}": percar[fc].tolist() for fc in run_fc},
           "cells": cells}
    (out / "analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"wall_s": time.time() - t_start, "coverage_meets_all": coverage["meets_all"],
                      "smallest_size": coverage["smallest_size_meeting"]}))


# --------------------------------------------------------------------------- #
# Stage numbers (CPU only)
# --------------------------------------------------------------------------- #
def build_numbers(cfg, out):
    a = json.loads((out / "analysis.json").read_text())
    cv = np.load(out / "miss_curves.npz")
    ss = core.seed_sequences(cfg["master_seed"])
    bcfg = cfg["bootstrap"]
    cells = {c["carrier_ghz"]: c for c in a["cells"]}
    fcs = sorted(cells)
    n_place = cells[fcs[0]]["n_placements"]
    W = core.resample_weights(ss["bootstrap"], n_place, bcfg["n_resamples"])
    shift = {float(k): v for k, v in a["f2_receive_gain_shift_db"].items()}
    ci = bcfg["ci_percent"]
    req, boot = {}, {}
    for f in fcs:
        fc = f * 1e9
        vis, pid, e0 = cv[f"vis_{fc}"], cv[f"pid_{fc}"], float(cv[f"eirp0_{fc}"])
        for b in BANKS:
            g, P = cv[f"grid_{fc}_{b}"], cv[f"P_{fc}_{b}"]
            S, C = core.placement_sums(P, vis, pid, n_place)
            pt = core.crossing(S.sum(0) / C.sum(), g)
            bt = core.boot_crossings(S, C, g, W)
            for reg, d in (("F1", 0.0), ("F2", shift[f])):
                req[(reg, b, f)] = None if pt is None else e0 + pt + d
                boot[(reg, b, f)] = e0 + bt + d
    rows = []
    for (reg, b, f), v in req.items():
        rows.append({"regime": reg, "detector": LEGACY[b], "bank": b, "fc_hz": f * 1e9, "carrier_ghz": f,
                     "required_eirp_dbm": v, "ci95_db": core.pct_ci(boot[(reg, b, f)], ci),
                     "n_resamples_without_crossing": int(np.sum(~np.isfinite(boot[(reg, b, f)]))),
                     "estimand": "min EIRP with mean visible-onset deadline miss <= 0.10"
                                 + ("" if reg == "F1" else "; F2 = F1 + (G_rx,F1 - G_F2)")})

    def paired(k1, k2):
        """value(k1) - value(k2) with a paired bootstrap interval."""
        if req[k1] is None or req[k2] is None:
            return None, [None, None]
        return req[k1] - req[k2], core.pct_ci(boot[k1] - boot[k2], ci)

    sav = []
    for f in fcs:
        for reg in ("F1", "F2"):
            for b, comp in (("redesigned", "Dfix-D0"), ("per_carrier", "D1-D0")):
                p, c = paired((reg, "baseline", f), (reg, b, f))
                sav.append({"regime": reg, "comparison": comp, "bank": b, "fc_hz": f * 1e9, "carrier_ghz": f,
                            "saving_db": p, "ci95_db": c, "note": "positive = bank needs less EIRP than baseline"})
    extra = []
    for reg in ("F1", "F2"):
        for b in BANKS:
            for f in fcs[1:]:
                p, c = paired((reg, b, f), (reg, b, fcs[0]))
                extra.append({"regime": reg, "detector": LEGACY[b], "bank": b, "fc_hz": f * 1e9,
                              "extra_eirp_db": p, "ci95_db": c, "lambda2_term_db": 20 * math.log10(f / fcs[0]),
                              "note": f"paired over placements vs {fcs[0]:g} GHz"})
    spot = {}
    f2c = cfg["f2_spot_check_carrier_hz"]
    if f"eirp0_F2spot_{f2c}" in cv.files:
        f = f2c / 1e9
        vis, pid = cv[f"vis_{f2c}"], cv[f"pid_{f2c}"]
        for b in BANKS:
            g, P = cv[f"grid_F2spot_{f2c}_{b}"], cv[f"P_F2spot_{f2c}_{b}"]
            S, C = core.placement_sums(P, vis, pid, n_place)
            pt = core.crossing(S.sum(0) / C.sum(), g)
            direct = None if pt is None else float(cv[f"eirp0_F2spot_{f2c}"]) + pt
            spot[b] = {"direct_f2_dbm": direct, "shifted_f2_dbm": req[("F2", b, f)],
                       "difference_db": None if None in (direct, req[("F2", b, f)]) else direct - req[("F2", b, f)]}
    ra_rows, cal_rows, ev_rows, cap_rows, pt_rows, ct_rows = [], [], [], [], [], []
    for f in fcs:
        c = cells[f]
        for b in BANKS:
            k = c["banks"][b]
            cal = k["calibration"]
            cal_rows.append({"carrier_ghz": f, "bank": b, "detector": LEGACY[b], "threshold": cal["threshold"],
                             "calibration_N": cal["calibration_N"], "validation_N": cal["validation_N"],
                             "false_alarm_count": cal["false_alarms"], "empirical_pfa": cal["pfa"],
                             "wilson_ci95": cal["wilson_ci95"], "accept": cal["accept"]})
            for kk, v in cal["reference_averaging"].items():
                ra_rows.append({"carrier_ghz": f, "bank": b, "K_reference_frames": int(kk), **v})
            ev_rows.append({"carrier_ghz": f, "bank": b, **cal["events"]})
            cap_rows.append({"carrier_ghz": f, "bank": b, **k["test_capture_loss_db"]})
        for m, v in c["point_targets"].items():
            pt_rows.append({"carrier_ghz": f, "model": m, "max_tested_eirp_dbm": max(v["eirp_tested_dbm"]),
                            "min_tested_eirp_dbm": min(v["eirp_tested_dbm"]),
                            "miss_min": min(v["mean_miss"]), "miss_max": max(v["mean_miss"]),
                            "max_residual_signal_energy": v["max_residual_signal_energy"]})
        ct_rows.append({"carrier_ghz": f, **c["constant_term"]})
    c0 = cells[fcs[0]]
    hid = {f"{f:g}": cells[f]["hidden"] for f in fcs}
    numbers = {
        "required_eirp_dbm_operational": rows, "detector_savings_db": sav, "extra_eirp_vs_3p5_db": extra,
        "f2_spot_check": spot, "false_alarm_calibration": {"cells": cal_rows,
                                                           "all_accept": all(r["accept"] for r in cal_rows)},
        "event_false_alarms": ev_rows, "reference_averaging": {"rows": ra_rows},
        "test_capture_loss_db": cap_rows, "constant_term_floor": ct_rows, "coverage": a["coverage"],
        "point_targets": pt_rows,
        "hidden_forearm": {"placements": len(c0["hidden"]["placements"]), "onsets": c0["hidden"]["n_hidden"],
                           "n_onsets_total": c0["n_onsets"], "per_carrier": hid},
        "placements": {"n_selected": c0["n_placements"], "k_test_motions": c0["k_test_motions"],
                       "n_visible_onsets": c0["n_visible"]},
        "antennas": {"f2_tx_gain_db": a["f2_tx_gain_db"], "f2_receive_gain_shift_db": a["f2_receive_gain_shift_db"]},
        "bootstrap": {"n_resamples": bcfg["n_resamples"], "ci_percent": ci,
                      "unit": "placement", "paired": "one resample index set shared by all cells"}}
    return {"release": "rerun-v2", "primary_estimand": "operational required EIRP (mean visible-onset "
            "deadline miss <= 0.10; 200 noise trials; 1 dB grid refined to 0.25 dB at the crossing)",
            "sources": {"analysis_sha256": hashlib.sha256((out / "analysis.json").read_bytes()).hexdigest(),
                        "miss_curves_sha256": hashlib.sha256((out / "miss_curves.npz").read_bytes()).hexdigest()},
            "numbers": numbers}


def _numbers_stage(cfg, out, dry):
    res = build_numbers(cfg, out)
    res["manifest"] = manifest(cfg, "numbers", {"dry_run": dry})
    again = build_numbers(cfg, out)  # determinism of the CPU stage (part of G11)
    res["numbers"]["determinism_numbers_stage"] = json.dumps(again["numbers"], sort_keys=True) == \
        json.dumps({k: v for k, v in res["numbers"].items() if k != "determinism_numbers_stage"}, sort_keys=True)
    (out / "manuscript_numbers.json").write_text(json.dumps(res, indent=2))
    print("numbers written; deterministic:", res["numbers"]["determinism_numbers_stage"])


# --------------------------------------------------------------------------- #
# Stage gates (merge G1-G9 with G10/G11 of this run)
# --------------------------------------------------------------------------- #
def _gates_stage(cfg, out, dry):
    g = json.loads((out / "gates_rt" / "gates.json").read_text())["gates"]
    rt = json.loads((out / "rt_manifest.json").read_text())["checks"]
    rows = rt["localization"]
    hits = [abs(r["r_hat"] - r["true_bin"]) <= 1 for r in rows]
    exact = [r["r_hat"] == r["true_bin"] for r in rows]
    ph = float(np.mean(hits))
    lb = _lower95(ph, len(rows))
    g["G10"] = {"status": "PASS" if lb >= 0.90 else "FAIL", "n": len(rows),
                "hits_within_1_bin": int(sum(hits)), "exact_bin": int(sum(exact)),
                "phat": ph, "lower95_one_sided": lb,
                "expected": "one-sided 95% lower bound of P(|bin err|<=1) >= 0.90 (noise-free static return)"}
    det = rt["determinism"]
    num = json.loads((out / "manuscript_numbers.json").read_text())["numbers"]
    ok = all(d["bit_identical"] for d in det) and num.get("determinism_numbers_stage", False)
    g["G11"] = {"status": "PASS" if ok else "FAIL", "rt_resimulation": det,
                "numbers_stage_repeat_identical": num.get("determinism_numbers_stage"),
                "expected": "re-simulated records bit-identical; repeated numbers stage identical"}
    payload = {"release": "rerun-v2", "manifest": manifest(cfg, "gates", {"dry_run": dry}), "gates": g}
    (out / "gates.json").write_text(json.dumps(payload, indent=2))
    for k, v in g.items():
        print(k, v["status"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="rerun v2 (deadline detection, disjoint dev/test)")
    ap.add_argument("stage", choices=("rt", "analyze", "numbers", "gates"))
    ap.add_argument("--config", default="configs/rerun_v2.yaml")
    ap.add_argument("--out", default="results/rerun_v2/full")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = core.load_cfg(a.config)
    {"rt": _rt_stage, "analyze": _analyze_stage, "numbers": _numbers_stage,
     "gates": _gates_stage}[a.stage](cfg, Path(a.out), a.dry_run)


if __name__ == "__main__":
    main()
