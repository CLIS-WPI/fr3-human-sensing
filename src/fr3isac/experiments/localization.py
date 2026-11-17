"""Pilot localization and detector pipeline on the Sionna RT solver.

Builds the sensing channel with background clutter and a moving target, localizes
the body range from the data, forms the detector input tensor, and calibrates the
detection thresholds on null windows. Provides the detector banks (fixed and
carrier-adapted) and the required-EIRP margin used by the pilot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mitsuba as mi
import numpy as np
from sionna.rt import (
    ITURadioMaterial,
    PathSolver,
    PlanarArray,
    Receiver,
    Scene,
    SceneObject,
    Transmitter,
)
from sionna.rt.rcs import ConstantRCSSensingTarget, RCSSolver, TR38901SensingTarget

from fr3isac.config import SPEED_OF_LIGHT_M_S, FairnessConfig
from fr3isac.fairness.normalize import normalize
from fr3isac.geometry import (
    CPI_S,
    DEV_MOTIONS,
    NSLOW,
    PRF_SLOW_HZ,
    ROI_X,
    ROI_Y,
    SEGMENT_OFFSETS,
    TARGET_HEIGHT_M,
    TEST_MOTIONS,
    analytic_arm_trajectory,
)
from fr3isac.geometry import (
    bs_position as _bs_position,
)
from fr3isac.geometry import (
    load_descriptor as _descriptor,
)
from fr3isac.geometry import (
    segment_aabb_intersect as _segment_aabb_intersect,
)
from fr3isac.geometry import (
    static_sample as _static_sample,
)
from fr3isac.sensing.rt_channel import (
    add_noise_frequency_domain,
    cfr_passband,
    detector_input_scale,
    sensing_path_arrays,
    thermal_noise_variance,
)
from fr3isac.sensing.rt_localize import localize_body_bin, range_m_to_bin
from fr3isac.sensing.rt_slowtime import interp_complex_series, path_records, synthesize_cfr
from fr3isac.sensing.transient_detector import limb_chirp_template

C = SPEED_OF_LIGHT_M_S
CARRIERS = (3.5e9, 12e9, 15e9, 28e9)
REGIMES = ("F1", "F2")
MODELS = ("R0", "R1-P", "R1-A")
NSC, SCS = 816, 120e3
OCC_BW = NSC * SCS
K_ANCHORS = 8
PFA = 1e-3
NOISE_FIGURE_DB = 7.0
P_TX_W = 1.0  # conducted power (30 dBm), split over active subcarriers (C2)
SCENE_JSON = Path("scenes/tier1_primary/scene.json")


D0_AMPLITUDES = (0.15, 0.22, 0.28)
D0_OMEGAS = (2 * np.pi * 0.6, 2 * np.pi * 0.8, 2 * np.pi * 1.0)
D0_PHASES = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
D1_AMPLITUDES = tuple(round(0.15 + 0.0025 * i, 4) for i in range(49))     # 0.15..0.27
D1_OMEGAS = tuple(2 * np.pi * (0.46 + 0.01 * i) for i in range(29))       # 2pi*0.46..2pi*0.74
D1_PHASES = tuple(np.linspace(0, 2 * np.pi, 48, endpoint=False))
D1_TOP_K = 64

# Human ROI (scenes/tier1_primary/scene.json).


def _p3(v):
    return mi.Point3f(float(v[0]), float(v[1]), float(v[2]))






def _box(name, center, size):
    tf = mi.ScalarTransform4f.translate([float(c) for c in center]) @ mi.ScalarTransform4f.scale(
        [float(s) / 2.0 for s in size])
    return SceneObject(mi_mesh=mi.load_dict({"type": "cube", "to_world": tf}), name=name)


def _array():
    return PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")


def build_room(fc_hz, targets=()):
    d = _descriptor()
    sc = Scene()
    mats = {m: ITURadioMaterial(name=m, itu_type=m, thickness=0.1) for m in d["materials"]}
    for m in mats.values():
        sc.add(m)
    for obj in d["objects"]:
        o = _box(obj["name"], obj["center_m"], obj["size_m"])
        o.radio_material = mats[obj["material"]]
        sc.add(o)
    bs = d["bs"]["position_m"]
    sc.add(Transmitter("bs", position=mi.Point3f(*[float(x) for x in bs])))
    sc.add(Receiver("bs_sense", position=mi.Point3f(*[float(x) for x in bs])))
    sc.tx_array = _array()
    sc.rx_array = _array()
    sc.frequency = fc_hz
    for t in targets:
        sc.add(t)
    return sc


# --------------------------------------------------------------------------- #
# Scenario eligibility (defined before outcomes; every exclusion logged)
# --------------------------------------------------------------------------- #


def eligible_scenarios(step_m=1.0):
    """Grid over the human ROI, keep BS-LoS scenarios; log every exclusion."""
    d = _descriptor()
    bs = np.asarray(d["bs"]["position_m"], dtype=float)
    blockers = [o for o in d["objects"] if o["name"] not in ("floor", "ceiling")]
    eligible, excluded = [], []
    x = ROI_X[0]
    while x <= ROI_X[1] + 1e-9:
        y = ROI_Y[0]
        while y <= ROI_Y[1] + 1e-9:
            target = np.array([x, y, TARGET_HEIGHT_M])
            reason = None
            for o in blockers:
                if _segment_aabb_intersect(bs, target, o["center_m"], o["size_m"]):
                    reason = f"blocked_by:{o['name']}"
                    break
            if reason is None:
                eligible.append((float(x), float(y)))
            else:
                excluded.append({"position": [float(x), float(y)], "reason": reason})
            y += step_m
        x += step_m
    return eligible, excluded


# --------------------------------------------------------------------------- #
# RT CFR (background + target) and the fast slow-time synthesis
# --------------------------------------------------------------------------- #
def _bg_cfr(fc_hz, offsets_hz):
    paths = PathSolver(deterministic=True)(build_room(fc_hz), max_depth=1, los=True,
                                           specular_reflection=True, diffuse_reflection=False,
                                           refraction=False, diffraction=False, seed=1)
    arr = sensing_path_arrays(paths)
    return cfr_passband(arr["a"], arr["tau"], fc_hz, offsets_hz)


def _targets_for(model, s):
    if model == "R0":
        return [TR38901SensingTarget("human", object_type="human", model_type=2,
                                     position=_p3(s.torso_position), random_sigma_s=False,
                                     random_xpr=False, random_phases=False)]
    if model == "R1-P":
        return [ConstantRCSSensingTarget("point", sigma=1.0, position=_p3(s.torso_position))]
    return [ConstantRCSSensingTarget("torso", sigma=0.7, position=_p3(s.torso_position)),
            ConstantRCSSensingTarget("right_forearm", sigma=0.3, position=_p3(s.forearm_position))]


def _solve_cfr(fc_hz, model, s, offsets_hz, gain_db):
    paths = RCSSolver(deterministic=True)(build_room(fc_hz, _targets_for(model, s)), max_depth=1,
                                          los=True, specular_reflection=False, refraction=False, seed=1)
    arr = sensing_path_arrays(paths)
    return cfr_passband(arr["a"], arr["tau"], fc_hz, offsets_hz) * (10.0 ** (gain_db / 20.0))




def _r1a_fast(fc_hz, root, motion, offsets_hz, gain_db, k_anchors=K_ANCHORS):
    """Fast R1-A slow-time CFR: RT amplitudes at anchors + geometric phase."""
    amp, omega = motion
    w = 2 * np.pi * omega
    times = np.arange(NSLOW) * (CPI_S / NSLOW)
    anchor_times = np.linspace(0.0, CPI_S, k_anchors, endpoint=False)
    torso_amps, fa_amps = [], []
    for ta in anchor_times:
        s = analytic_arm_trajectory(root, np.array([ta]), amplitude_m=amp, omega_rad_s=w)[0]
        paths = RCSSolver(deterministic=True)(build_room(fc_hz, _targets_for("R1-A", s)),
                                              max_depth=1, los=True, specular_reflection=False,
                                              refraction=False, seed=1)
        bs = _bs_position()
        r_t = float(np.linalg.norm(np.asarray(s.torso_position) - bs))
        r_f = float(np.linalg.norm(np.asarray(s.forearm_position) - bs))
        torso_amps.append(0.0 + 0.0j)
        fa_amps.append(0.0 + 0.0j)
        for rec in path_records(paths):
            if abs(rec["tau"] - 2 * r_t / C) <= abs(rec["tau"] - 2 * r_f / C):
                torso_amps[-1] += rec["a"]
            else:
                fa_amps[-1] += rec["a"]
    torso_amps = np.asarray(torso_amps) * (10.0 ** (gain_db / 20.0))
    fa_amps = np.asarray(fa_amps) * (10.0 ** (gain_db / 20.0))
    samples = analytic_arm_trajectory(root, times, amplitude_m=amp, omega_rad_s=w)
    bs = _bs_position()
    r_torso = np.array([float(np.linalg.norm(np.asarray(s.torso_position) - bs)) for s in samples])
    r_fa = np.array([float(np.linalg.norm(np.asarray(s.forearm_position) - bs)) for s in samples])
    a_torso = interp_complex_series(anchor_times, torso_amps, times)
    a_fa = interp_complex_series(anchor_times, fa_amps, times)
    return synthesize_cfr(fc_hz, offsets_hz, np.vstack([a_torso, a_fa]), np.vstack([r_torso, r_fa]))


# --------------------------------------------------------------------------- #
# Detector banks / statistic / thresholds
# --------------------------------------------------------------------------- #
def _regime_gain_db(fc_hz, regime):
    cfg = FairnessConfig(regime=regime)
    return normalize(cfg, fc_hz).bs.gain_db + cfg.bs_rx_gain_db


def _regime_eirp_dbm(fc_hz, regime):
    cfg = FairnessConfig(regime=regime)
    return normalize(cfg, fc_hz).bs.eirp_dbm


def _bank(fc_hz, amplitudes, omegas, phases):
    return np.asarray([limb_chirp_template(fc_hz, a, w, p, NSLOW, CPI_S)
                       for a in amplitudes for w in omegas for p in phases])


def _scores(diff, templates, sigma2, eps=1e-30):
    norm = np.sum(np.abs(templates) ** 2, axis=1)
    return np.abs(templates.conj() @ diff) ** 2 / np.maximum(norm * max(sigma2, eps), eps)


def _statistic(diff, templates, sigma2):
    return float(np.max(_scores(diff, templates, sigma2)))


def _capture(signal, templates, eps=1e-30):
    """Normalized matched-filter capture in [0, 1]: max_t |<t,s>|^2/(||t||^2||s||^2)."""
    s = np.asarray(signal, dtype=np.complex128)
    denom = np.sum(np.abs(templates) ** 2, axis=1) * max(float(np.sum(np.abs(s) ** 2)), eps)
    corr = np.abs(templates.conj() @ s) ** 2 / np.maximum(denom, eps)
    return float(np.clip(np.max(corr), 0.0, 1.0))


def _body_vector(bg_cfr, cfr, r_hat):
    cfr = np.asarray(cfr)
    if cfr.ndim == 1:
        return np.full(NSLOW, np.fft.ifft(bg_cfr + cfr, axis=0)[r_hat])
    return np.fft.ifft(bg_cfr[:, None] + cfr, axis=0)[r_hat, :]


def _null_threshold(bank, x_ref, sigma2, rng, n_null):
    stats = np.empty(n_null)
    for i in range(n_null):
        x = add_noise_frequency_domain(x_ref, sigma2, rng)
        stats[i] = _statistic(x - x_ref, bank, sigma2)
    return float(np.quantile(stats, 1.0 - PFA))


def _cp95_lower(p, n, z=1.6448536269514722):
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float((centre - half) / denom)


# --------------------------------------------------------------------------- #
# C2 link-budget audit (issue #6)
# --------------------------------------------------------------------------- #
def link_budget_audit(fc_hz, regime, root=(5.0, 4.0, TARGET_HEIGHT_M)):
    offsets = (np.arange(NSC) - NSC // 2) * SCS
    cfg = FairnessConfig(regime=regime)
    fair = normalize(cfg, fc_hz)
    gain = fair.bs.gain_db + cfg.bs_rx_gain_db
    sigma2 = thermal_noise_variance(SCS, noise_figure_db=NOISE_FIGURE_DB)
    s = _static_sample(root)
    cfr = _solve_cfr(fc_hz, "R1-P", s, offsets, gain)  # unit-RCS point at root
    path_gain = float(np.mean(np.abs(cfr) ** 2))  # per-subcarrier received power (P_tx=1)
    rng_m = float(np.linalg.norm(np.asarray(root) - np.asarray(_descriptor()["bs"]["position_m"])))
    lam = C / fc_hz
    radar = lam**2 / ((4 * np.pi) ** 3 * rng_m**4)  # sigma=1, P_tx=1
    proc_gain_db = 10 * np.log10(NSLOW)
    snr_db = 10 * np.log10(path_gain / sigma2) + proc_gain_db
    return {
        "fc_hz": fc_hz, "regime": regime,
        "conducted_power_dbm": cfg.conducted_power_dbm,
        "bs_tx_gain_db": fair.bs.gain_db, "bs_rx_gain_db": cfg.bs_rx_gain_db,
        "eirp_dbm": fair.bs.eirp_dbm,
        "target_range_m": rng_m, "forearm_rcs_m2": 0.3, "torso_rcs_m2": 0.7,
        "rt_path_gain": path_gain, "radar_eq_gain": radar,
        "rt_vs_radar_db": 10 * np.log10(path_gain / radar),
        "noise_var_per_sc": sigma2, "noise_figure_db": NOISE_FIGURE_DB,
        "processing_gain_db": proc_gain_db, "snr_db_after_processing": snr_db,
    }


# --------------------------------------------------------------------------- #
# Pilot
# --------------------------------------------------------------------------- #
def run_pilot(carriers=CARRIERS, regimes=REGIMES, scenarios=None,
              dev_motions=DEV_MOTIONS, test_motions=TEST_MOTIONS, n_null=4000, seed=0):
    offsets = (np.arange(NSC) - NSC // 2) * SCS
    sigma2 = thermal_noise_variance(SCS, noise_figure_db=NOISE_FIGURE_DB)
    c2_scale = detector_input_scale(P_TX_W, NSC)
    sigma2_body = sigma2 / NSC  # noise after coherent range compression (C2)
    if scenarios is None:
        elig, _ = eligible_scenarios()
        idx = np.linspace(0, len(elig) - 1, min(4, len(elig))).round().astype(int)
        scenarios = [elig[i] for i in idx]
    loc_rows, det_rows, margin_rows, inr_rows = [], [], [], []
    capture = []
    rng = np.random.default_rng(seed)

    for fc in carriers:
        bg = _bg_cfr(fc, offsets)
        d0 = _bank(fc, D0_AMPLITUDES, D0_OMEGAS, D0_PHASES)
        cand = _bank(fc, D1_AMPLITUDES, D1_OMEGAS, D1_PHASES)

        for regime in regimes:
            gain = _regime_gain_db(fc, regime)
            eirp0 = _regime_eirp_dbm(fc, regime)
            tx0 = FairnessConfig(regime=regime).conducted_power_dbm

            dev_scores = np.zeros(cand.shape[0])
            dev_store = []
            for xy in scenarios:
                root = (float(xy[0]), float(xy[1]), TARGET_HEIGHT_M)
                static = _solve_cfr(fc, "R1-A", _static_sample(root), offsets, gain)
                r_hat, _ = localize_body_bin(bg[:, None] + static[:, None], bg[:, None],
                                             OCC_BW, 1.0, 12.0)
                x_ref = c2_scale * _body_vector(bg, static, r_hat)
                for motion in dev_motions:
                    x = c2_scale * _body_vector(bg, _r1a_fast(fc, root, motion, offsets, gain), r_hat)
                    dev_scores += _scores(x - x_ref, cand, sigma2_body).ravel()
                    dev_store.append(x - x_ref)
            d1 = cand[np.argsort(dev_scores)[-D1_TOP_K:]]
            for sv in dev_store:
                for label, bank in (("D0", d0), ("D1", d1)):
                    capture.append({"fc_hz": fc, "regime": regime, "split": "dev",
                                    "detector": label,
                                    "capture_db": 10 * np.log10(_capture(sv, bank) + 1e-30)})

            for xy in scenarios:
                root = (float(xy[0]), float(xy[1]), TARGET_HEIGHT_M)
                true_bin = range_m_to_bin(float(np.linalg.norm(np.asarray(root) - _bs_position())), NSC, OCC_BW)
                for model in MODELS:
                    static = _solve_cfr(fc, model, _static_sample(root), offsets, gain)
                    r_hat, _ = localize_body_bin(bg[:, None] + static[:, None], bg[:, None],
                                                 OCC_BW, 1.0, 12.0)
                    loc_rows.append({"fc_hz": fc, "regime": regime, "model": model, "root": root,
                                     "true_bin": true_bin, "r_hat": r_hat,
                                     "abs_bin_err": abs(r_hat - true_bin)})
                    x_ref = c2_scale * _body_vector(bg, static, r_hat)
                    thr = {d: _null_threshold(b, x_ref, sigma2_body, rng, n_null)
                           for d, b in (("D0", d0), ("D1", d1))}
                    if model != "R1-A":
                        for detector, t in thr.items():
                            det_rows.append({"fc_hz": fc, "regime": regime, "model": model,
                                             "detector": detector, "miss": 1.0, "threshold": t})
                            margin_rows.append({"fc_hz": fc, "regime": regime, "model": model,
                                                "detector": detector, "required_eirp_dbm": None,
                                                "required_tx_dbm": None, "miss": 1.0,
                                                "note": "no independent limb motion"})
                        continue
                    test_frames = [_r1a_fast(fc, root, m, offsets, gain) for m in test_motions]
                    for detector, bank in (("D0", d0), ("D1", d1)):
                        t = thr[detector]
                        stats = np.asarray([
                            _statistic(c2_scale * _body_vector(bg, fr, r_hat) - x_ref, bank, sigma2_body)
                            for fr in test_frames])
                        miss = float(np.mean(stats < t))
                        for fr in test_frames:
                            sv = c2_scale * _body_vector(bg, fr, r_hat) - x_ref
                            capture.append({"fc_hz": fc, "regime": regime, "split": "test",
                                            "detector": detector,
                                            "capture_db": 10 * np.log10(_capture(sv, bank) + 1e-30)})
                        s10 = float(np.quantile(stats, 0.10))
                        extra_db = 10 * np.log10(t / s10) if s10 > 0 else float("inf")
                        det_rows.append({"fc_hz": fc, "regime": regime, "model": model,
                                         "detector": detector, "miss": miss, "threshold": t,
                                         "stats": stats.tolist()})
                        margin_rows.append({"fc_hz": fc, "regime": regime, "model": model,
                                            "detector": detector, "miss": miss,
                                            "required_eirp_dbm": eirp0 + extra_db,
                                            "required_tx_dbm": tx0 + extra_db})
                        inr = s10 / t - 1.0
                        inr_rows.append({"fc_hz": fc, "regime": regime, "model": model,
                                         "detector": detector,
                                         "max_tolerable_inr_db": (10 * np.log10(inr) if inr > 0 else None)})

    hits = [r["abs_bin_err"] <= 1 for r in loc_rows]
    phat = float(np.mean(hits)) if hits else 0.0
    gate10 = {"n": len(loc_rows), "hits_within_1_bin": int(np.sum(hits)), "phat": phat,
              "cp95_lower": _cp95_lower(phat, len(loc_rows)) if loc_rows else 0.0}
    gate10["status"] = "PASS" if gate10["cp95_lower"] >= 0.90 else "FAIL"

    # D0 vs D1 required-EIRP difference per carrier x regime (not pooled)
    diffs = []
    for fc in carriers:
        for regime in regimes:
            d0e = [r["required_eirp_dbm"] for r in margin_rows
                   if r["fc_hz"] == fc and r["regime"] == regime and r["model"] == "R1-A"
                   and r["detector"] == "D0" and r["required_eirp_dbm"] is not None]
            d1e = [r["required_eirp_dbm"] for r in margin_rows
                   if r["fc_hz"] == fc and r["regime"] == regime and r["model"] == "R1-A"
                   and r["detector"] == "D1" and r["required_eirp_dbm"] is not None]
            if d0e and d1e:
                per_scen = np.array(d1e) - np.array(d0e)
                diffs.append({"fc_hz": fc, "regime": regime, "n": int(per_scen.size),
                              "median_diff_db": float(np.median(per_scen)),
                              "min_diff_db": float(per_scen.min()),
                              "max_diff_db": float(per_scen.max())})

    cap_table = []
    for fc in carriers:
        for split in ("dev", "test"):
            for label in ("D0", "D1"):
                vals = [c["capture_db"] for c in capture
                        if c["fc_hz"] == fc and c["split"] == split and c["detector"] == label]
                if vals:
                    cap_table.append({"fc_hz": fc, "split": split, "detector": label,
                                      "n": len(vals), "mean_capture_db": float(np.mean(vals)),
                                      "p10_capture_db": float(np.quantile(vals, 0.10))})

    return {"gate_G10": gate10, "localization": loc_rows, "detector": det_rows,
            "capture": capture, "capture_table": cap_table,
            "margin": margin_rows, "max_inr": inr_rows, "d0_d1_eirp_diff": diffs,
            "link_budget": [link_budget_audit(fc, rg) for fc in carriers for rg in regimes],
            "config": {"nsc": NSC, "scs": SCS, "nslow": NSLOW, "prf_slow_hz": PRF_SLOW_HZ,
                       "cpi_s": CPI_S, "pfa": PFA, "k_anchors": K_ANCHORS,
                       "carriers_hz": list(carriers), "regimes": list(regimes),
                       "scenarios": [list(s) for s in scenarios],
                       "dev_motions": [list(m) for m in dev_motions],
                       "test_motions": [list(m) for m in test_motions]}}


def determinism_gate(out_dir="results/commag"):
    tmp = Path(out_dir) / "_det"
    tmp.mkdir(parents=True, exist_ok=True)
    for tag in ("a", "b"):
        subprocess.run([sys.executable, "-m", "fr3isac.experiments.localization", "--emit",
                        str(tmp / f"{tag}.json"), "--tiny"], check=True)
    a = json.loads((tmp / "a.json").read_text())
    b = json.loads((tmp / "b.json").read_text())
    keys_ok = sorted(a) == sorted(b)
    rows_ok = (len(a["detector"]) == len(b["detector"])
               and len(a["localization"]) == len(b["localization"])
               and len(a["margin"]) == len(b["margin"]))
    max_abs = 0.0
    for section, keys in (("detector", ("miss", "threshold")),
                          ("margin", ("required_eirp_dbm",))):
        for ra, rb in zip(a[section], b[section], strict=False):
            for k in keys:
                if ra.get(k) is not None and rb.get(k) is not None:
                    max_abs = max(max_abs, abs(float(ra[k]) - float(rb[k])))
    for ra, rb in zip(a["localization"], b["localization"], strict=False):
        max_abs = max(max_abs, abs(float(ra["r_hat"]) - float(rb["r_hat"])))
    status = "PASS" if keys_ok and rows_ok and max_abs <= 1e-9 else "FAIL"
    return {"status": status, "keys_match": keys_ok, "row_counts_match": rows_ok,
            "max_abs_diff": max_abs}


def hand_calculation(root=(2.0, 1.5, 1.4), fc=12e9, regime="F1", motion=(0.18, 0.9),
                     pilot_json="results/commag/localization.json"):
    """Fully worked single-cell chain (item 1b) with a residual decomposition."""
    offsets = (np.arange(NSC) - NSC // 2) * SCS
    sigma2 = thermal_noise_variance(SCS, noise_figure_db=NOISE_FIGURE_DB)
    sigma2_body = sigma2 / NSC
    c2 = detector_input_scale(P_TX_W, NSC)
    bs = _bs_position()
    cfg = FairnessConfig(regime=regime)
    fair = normalize(cfg, fc)
    gain = _regime_gain_db(fc, regime)
    amp, omega = motion
    lam = C / fc

    bg = _bg_cfr(fc, offsets)
    static = _solve_cfr(fc, "R1-A", _static_sample(root), offsets, gain)
    r_hat, _ = localize_body_bin(bg[:, None] + static[:, None], bg[:, None], OCC_BW, 1.0, 12.0)
    x_ref = c2 * _body_vector(bg, static, r_hat)
    diff = c2 * _body_vector(bg, _r1a_fast(fc, root, motion, offsets, gain), r_hat) - x_ref

    fa = np.asarray(root, dtype=float) + np.asarray(SEGMENT_OFFSETS["right_forearm"], dtype=float)
    range_m = float(np.linalg.norm(fa - bs))
    g_tx = 10 ** (fair.bs.gain_db / 10.0)
    g_rx = 10 ** (cfg.bs_rx_gain_db / 10.0)
    h2 = g_tx * g_rx * 0.3 * lam**2 / ((4 * np.pi) ** 3 * range_m**4)
    p_split = (P_TX_W / NSC) * h2
    snr_sc_db = 10 * np.log10(p_split / sigma2)
    n_coh = NSC * NSLOW
    pred_db = snr_sc_db + 10 * np.log10(n_coh)

    matched_db = 10 * np.log10(float(np.sum(np.abs(diff) ** 2)) / sigma2_body)
    d0 = _bank(fc, D0_AMPLITUDES, D0_OMEGAS, D0_PHASES)
    bank_db = 10 * np.log10(_statistic(diff, d0, sigma2_body))

    times = np.arange(NSLOW) * (CPI_S / NSLOW)
    samples = analytic_arm_trajectory(root, times, amplitude_m=amp, omega_rad_s=2 * np.pi * omega)
    r_t = np.array([float(np.linalg.norm(np.asarray(s.forearm_position) - bs)) for s in samples])
    cos_theta = abs(float((fa - bs)[0])) / range_m  # viewing projection of the x-motion

    def _cap(templates):
        return _capture(diff, templates)

    exact = np.exp(-1j * 4 * np.pi / lam * (r_t - range_m))[None, :]
    proj = np.exp(-1j * 4 * np.pi / lam * (amp * cos_theta) * np.sin(2 * np.pi * omega * times))[None, :]
    cap_exact = _cap(exact)
    cap_proj = _cap(proj)
    cap_bank = _cap(d0)

    measured = None
    try:
        d = json.loads(Path(pilot_json).read_text())
        rows = [r for r in d["detector"] if r["fc_hz"] == fc and r["regime"] == regime
                and r["model"] == "R1-A" and r["detector"] == "D0"]
        if rows:
            measured = float(rows[0]["stats"][0])
    except FileNotFoundError:
        pass

    def db(x):
        return 10 * np.log10(x) if x > 0 else None

    return {
        "cell": {"scenario_root": list(root), "fc_hz": fc, "regime": regime,
                 "motion_amplitude_m": amp, "motion_omega_hz": omega},
        "range_forearm_m": range_m, "forearm_rcs_m2": 0.3, "lambda_m": lam,
        "conducted_power_dbm": cfg.conducted_power_dbm, "eirp_dbm": fair.bs.eirp_dbm,
        "tx_gain_db": fair.bs.gain_db, "rx_gain_db": cfg.bs_rx_gain_db,
        "receive_antenna_gain_db_by_regime": {"F1": FairnessConfig(regime="F1").bs_rx_gain_db,
                                              "F2": FairnessConfig(regime="F2").bs_rx_gain_db},
        "channel_gain_h2": h2, "per_subcarrier_signal_w": p_split,
        "per_subcarrier_noise_w": sigma2, "per_subcarrier_snr_db": snr_sc_db,
        "coherent_samples": n_coh, "coherent_gain_db": 10 * np.log10(n_coh),
        "predicted_postproc_snr_db": pred_db, "predicted_statistic": 10 ** (pred_db / 10.0),
        "bank_statistic_db": bank_db, "matched_statistic_db": matched_db,
        "measured_statistic": measured,
        "residual_db": (pred_db - 10 * np.log10(measured)) if measured else None,
        "decomposition_db": {
            "analytic_vs_matched": pred_db - matched_db,
            "template_capture_loss": -db(cap_bank),
            "viewing_geometry_projection": -db(cap_proj),
            "capture_bank_vs_projection_remainder": db(cap_proj / cap_bank) if cap_bank else None,
            "exact_geometric_capture_db": db(cap_exact),
            "combined_capture_bank_db": db(cap_bank),
        },
        "viewing_cos_theta": cos_theta,
        "onset_fraction_of_cycle": omega * CPI_S,
        "note": "residual = analytic_vs_matched + template_capture_loss; capture loss is "
                "dominated by the viewing-geometry projection (cos_theta); no 64x factor",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="pilot localization and detector pipeline")
    parser.add_argument("--out", default="results/commag")
    parser.add_argument("--emit")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--hand-calc", action="store_true")
    args = parser.parse_args(argv)
    if args.hand_calc:
        hc = hand_calculation()
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "hand_calc_12ghz_f1.json").write_text(json.dumps(hc, indent=2))
        print(json.dumps(hc, indent=2))
        return
    if args.audit:
        elig, excl = eligible_scenarios()
        print("eligible:", elig)
        print("excluded:", excl)
        for row in [link_budget_audit(fc, rg) for fc in CARRIERS for rg in REGIMES]:
            print(json.dumps(row))
        return
    if args.tiny:
        res = run_pilot(carriers=CARRIERS[:1], regimes=REGIMES[:1],
                        scenarios=eligible_scenarios()[0][:1],
                        dev_motions=DEV_MOTIONS[:1], test_motions=TEST_MOTIONS[:1], n_null=200)
    else:
        res = run_pilot()
        res["gate_G11"] = determinism_gate(args.out)
    if args.emit:
        Path(args.emit).write_text(json.dumps(res, sort_keys=True, indent=2))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "localization.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    print("G10:", res["gate_G10"]["status"], res["gate_G10"])
    if "gate_G11" in res:
        print("G11:", res["gate_G11"]["status"], res["gate_G11"])


if __name__ == "__main__":
    main()
