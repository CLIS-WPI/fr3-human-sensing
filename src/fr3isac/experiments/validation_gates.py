"""Validation gate harness on the Sionna RT solver.

Runs gates G1-G11 against the real RT solver and writes ``gates.json`` plus one plot
per gate. No localization, detector or production run is performed here. Gate status
is ``PASS`` / ``FAIL`` / ``INCONCLUSIVE`` and is never fabricated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from sionna.rt.rcs import (
    ConstantCPM,
    ConstantRCSSensingTarget,
    RCSSolver,
    ScatteringModel,
    SensingTarget,
)

from fr3isac.config import TargetConfig
from fr3isac.rcs.models import frequency_bound_rcs
from fr3isac.sensing.rt_channel import (
    add_noise_frequency_domain,
    cfr_passband,
    radar_equation_power,
    sensing_path_arrays,
    thermal_noise_variance,
)
from fr3isac.sensing.rt_slowtime import (
    interpolate_cfr,
    reconstruction_nmse,
    slow_time_cfr,
    track_frames,
)
from fr3isac.targets.builders import (
    analytic_arm_trajectory,
    build_human_target,
    build_point_target,
    segment_sigmas,
)

C = 299792458.0
CARRIERS = (3.5e9, 12e9, 15e9, 28e9)
NSC, SCS = 64, 120e3
NOISE_FIGURE_DB = 7.0


def _monostatic(R, fc, pattern="iso", sigma=1.0, velocity=None, rcs_callable=None):
    sc = Scene()
    sc.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    sc.add(Receiver("rx", position=mi.Point3f(0, 0, 0)))
    sc.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern=pattern, polarization="V")
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern=pattern, polarization="V")
    sc.frequency = fc
    if rcs_callable is None:
        sc.add(ConstantRCSSensingTarget(
            "t", sigma=sigma, position=mi.Point3f(float(R), 0, 0),
            velocity=(mi.Vector3f(*velocity) if velocity else None)))
    else:
        sm = ScatteringModel(lcs_positions=mi.Point3f(0, 0, 0), rcs=rcs_callable, cpm=ConstantCPM())
        sc.add(SensingTarget("t", scattering_model=sm, position=mi.Point3f(float(R), 0, 0),
                             length=0.5, width=0.5, height=1.75))
    return sc


def _solve(sc):
    return RCSSolver(deterministic=True)(sc, max_depth=1, los=True, specular_reflection=False,
                                         refraction=False, seed=1)


def _first_power(paths):
    a = sensing_path_arrays(paths)
    return float(np.abs(a["a"][0]) ** 2) if a["a"].size else 0.0


def _offsets():
    return (np.arange(NSC) - NSC // 2) * SCS


def _plot(out_dir, name, draw):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    draw(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def gate_g1(out_dir):
    rows, ok = [], True
    for fc in CARRIERS:
        R = 5.0
        a = sensing_path_arrays(_solve(_monostatic(R, fc)))
        H = cfr_passband(a["a"], a["tau"], fc, _offsets())
        ph = np.unwrap(np.angle(H))
        tau_est = -np.polyfit(_offsets(), ph, 1)[0] / (2 * np.pi)
        tau_true = 2 * R / C
        derr = tau_est - tau_true
        res = 1.0 / (NSC * SCS)
        lam = C / fc
        dR = lam / 100.0
        H1 = cfr_passband(a["a"], a["tau"], fc, np.array([0.0]))[0]
        a2 = sensing_path_arrays(_solve(_monostatic(R + dR, fc)))
        H2 = cfr_passband(a2["a"], a2["tau"], fc, np.array([0.0]))[0]
        dphase = float(np.angle(H2 * np.conj(H1)))
        predicted = float(np.angle(np.exp(-1j * 4 * np.pi * dR / lam)))
        perr = abs(dphase - predicted)
        ok &= (abs(derr) / res <= 0.1) and (perr <= 0.05)
        rows.append({"fc_hz": fc, "delay_err_s": derr, "delay_err_over_res": abs(derr) / res,
                     "dphase_rad": dphase, "dphase_pred_rad": predicted, "dphase_err_rad": perr})

    def draw(ax):
        fc = CARRIERS[0]
        a = sensing_path_arrays(_solve(_monostatic(5.0, fc)))
        H = cfr_passband(a["a"], a["tau"], fc, _offsets())
        ax.plot(_offsets() / 1e6, np.unwrap(np.angle(H)), "o-", ms=3, label="RT phase")
        ax.set_xlabel("subcarrier offset (MHz)")
        ax.set_ylabel("unwrapped phase (rad)")
        ax.set_title("G1 phase/delay (3.5 GHz)")

    return {"status": "PASS" if ok else "FAIL", "measured": rows,
            "expected": "delay_err<=0.1 bin; dphase_err<=0.05 rad",
            "plot": _plot(out_dir, "g1_phase_delay", draw)}


def gate_g2(out_dir):
    rows, ok = [], True
    for fc in CARRIERS:
        R = 5.0
        p = _first_power(_solve(_monostatic(R, fc)))
        exp = radar_equation_power(1.0, fc, R, 1.0)
        gdb = 10 * np.log10(p / exp)
        ok &= abs(gdb) <= 0.5
        rows.append({"fc_hz": fc, "received": p, "radar_eq": exp, "gain_err_db": gdb})

    def draw(ax):
        fc = CARRIERS[0]
        Rs = np.linspace(3, 30, 12)
        meas = [10 * np.log10(_first_power(_solve(_monostatic(R, fc)))) for R in Rs]
        ana = [10 * np.log10(radar_equation_power(1.0, fc, R, 1.0)) for R in Rs]
        ax.plot(Rs, meas, "o", ms=4, label="RT")
        ax.plot(Rs, ana, "-", label="radar eq")
        ax.set_xlabel("range (m)")
        ax.set_ylabel("RX power (dB)")
        ax.set_title("G2 radar equation (3.5 GHz)")
        ax.legend(fontsize=7)

    return {"status": "PASS" if ok else "FAIL", "measured": rows,
            "expected": "|gain_err|<=0.5 dB", "plot": _plot(out_dir, "g2_radar_equation", draw)}


def gate_g3(out_dir):
    fc = CARRIERS[0]
    Rs = np.logspace(np.log10(3), np.log10(30), 12)
    p = np.array([_first_power(_solve(_monostatic(R, fc))) for R in Rs])
    slope = float(np.polyfit(np.log10(Rs), 10 * np.log10(p), 1)[0])
    ok = abs(slope + 40.0) <= 1.0

    def draw(ax):
        ax.loglog(Rs, p, "o-", ms=4)
        ax.set_xlabel("range (m)")
        ax.set_ylabel("RX power")
        ax.set_title(f"G3 R^-4 slope = {slope:.2f} dB/dec (expect -40)")

    return {"status": "PASS" if ok else "FAIL", "measured": {"slope_db_per_decade": slope},
            "expected": "-40 +/- 1 dB/decade", "plot": _plot(out_dir, "g3_r4_slope", draw)}


def gate_g4(out_dir):
    fc = CARRIERS[0]
    a = sensing_path_arrays(_solve(_monostatic(5.0, fc)))
    H = cfr_passband(a["a"], a["tau"], fc, _offsets())
    sigma2 = thermal_noise_variance(SCS, noise_figure_db=NOISE_FIGURE_DB)
    rng = np.random.default_rng(0)
    resid = np.array([add_noise_frequency_domain(H, sigma2, rng) - H for _ in range(4000)])
    var_meas = float(np.var(resid.real) + np.var(resid.imag))
    var_err = abs(var_meas - sigma2) / sigma2

    def total_noise(bandwidth_hz):
        return (bandwidth_hz / SCS) * thermal_noise_variance(SCS, noise_figure_db=NOISE_FIGURE_DB)

    b2_db = 10 * np.log10(total_noise(2 * NSC * SCS) / total_noise(NSC * SCS))
    ok = var_err <= 0.02 and abs(b2_db - 3.01) <= 0.1

    def draw(ax):
        bws = np.array([20, 40, 80, 160, 320]) * 1e6
        v = [total_noise(b) for b in bws]
        ax.semilogy(bws / 1e6, v, "o-")
        ax.set_xlabel("bandwidth (MHz)")
        ax.set_ylabel("total in-band noise")
        ax.set_title("G4 noise vs bandwidth")

    return {"status": "PASS" if ok else "FAIL",
            "measured": {"var_meas": var_meas, "var_model": sigma2, "var_rel_err": var_err,
                         "bandwidth_x2_db": b2_db},
            "expected": "var within 2%; 2x bandwidth = 3.01 +/- 0.1 dB",
            "plot": _plot(out_dir, "g4_noise", draw)}


def gate_g5(out_dir):
    rows, ok = [], True
    for fc in (3.5e9, 28e9):
        p1 = _first_power(_solve(_monostatic(5.0, fc, rcs_callable=frequency_bound_rcs(1.0))))
        p10 = _first_power(_solve(_monostatic(5.0, fc, rcs_callable=frequency_bound_rcs(10.0))))
        g = 10 * np.log10(p10 / p1)
        ok &= abs(g - 10.0) <= 0.2
        rows.append({"fc_hz": fc, "sigma_x10_db": g})

    def draw(ax):
        sig = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
        fc = 3.5e9
        p = [_first_power(_solve(_monostatic(5.0, fc, rcs_callable=frequency_bound_rcs(s)))) for s in sig]
        ax.plot(sig, 10 * np.log10(np.array(p) / p[2]), "o-")
        ax.set_xscale("log")
        ax.set_xlabel("sigma (m^2)")
        ax.set_ylabel("power rel. to sigma=1 (dB)")
        ax.set_title("G5 RCS scaling")

    return {"status": "PASS" if ok else "FAIL", "measured": rows,
            "expected": "sigma x10 = +10.0 +/- 0.2 dB", "plot": _plot(out_dir, "g5_rcs", draw)}


def gate_g6(out_dir):
    v, bin_hz = 1.0, 1.0 / (NSC * 1e-3)
    rows, ok = [], True
    for fc in CARRIERS:
        a = sensing_path_arrays(_solve(_monostatic(5.0, fc, velocity=(v, 0, 0))))
        meas = float(np.asarray(a["doppler"]).mean())
        ana = -2 * v / (C / fc)
        ok &= abs(meas - ana) <= bin_hz
        rows.append({"fc_hz": fc, "doppler_hz": meas, "analytic_hz": ana, "abs_err_hz": abs(meas - ana)})

    def draw(ax):
        fcs = np.array(CARRIERS) / 1e9
        m = [r["doppler_hz"] for r in rows]
        an = [r["analytic_hz"] for r in rows]
        ax.plot(fcs, m, "o", ms=5, label="RT")
        ax.plot(fcs, an, "-", label="-2v/lambda")
        ax.set_xlabel("carrier (GHz)")
        ax.set_ylabel("Doppler (Hz)")
        ax.set_title("G6 Doppler (v=1 m/s)")
        ax.legend(fontsize=7)

    return {"status": "PASS" if ok else "FAIL", "measured": rows,
            "expected": "within 1 bin; negative (receding)", "plot": _plot(out_dir, "g6_doppler", draw)}


def gate_g7(out_dir):
    fc, R = 3.5e9, 5.0
    p_free = _first_power(_solve(_monostatic(R, fc)))
    sc = _monostatic(R, fc)
    mesh = mi.load_dict({
        "type": "cube",
        "to_world": mi.ScalarTransform4f.translate([R / 2, 0, 0]) @ mi.ScalarTransform4f.scale([0.05, 2.0, 2.0]),
    })
    plate = SceneObject(mi_mesh=mesh, name="plate")
    plate.radio_material = ITURadioMaterial("metal_plate", "metal", thickness=0.05)
    sc.add(plate)
    paths = _solve(sc)
    a = sensing_path_arrays(paths)
    p_block = float(np.abs(a["a"][0]) ** 2) if a["a"].size else 0.0
    drop_db = 10 * np.log10(p_free / p_block) if p_block > 0 else float("inf")
    ok = (p_block == 0.0) or (drop_db >= 20.0)

    def draw(ax):
        ax.bar(["free", "blocked"], [10 * np.log10(p_free + 1e-30),
                                     (10 * np.log10(p_block) if p_block > 0 else -300)])
        ax.set_ylabel("RX power (dB)")
        ax.set_title(f"G7 blockage (drop {drop_db:.1f} dB)")

    return {"status": "PASS" if ok else "FAIL",
            "measured": {"p_free": p_free, "p_blocked": p_block, "drop_db": drop_db,
                         "blocked_paths": int(a["a"].size)},
            "expected": "target path drops >=20 dB or disappears",
            "plot": _plot(out_dir, "g7_blockage", draw)}


def _peak_doppler(x, prf_hz, oversample=8):
    x = np.asarray(x, dtype=np.complex128)
    nfft = oversample * x.size
    X = np.fft.fftshift(np.fft.fft(x * np.hanning(x.size), n=nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / prf_hz))
    return float(f[np.argmax(np.abs(X))])


def _targets_scene(targets, fc=3.5e9):
    sc = Scene()
    sc.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    sc.add(Receiver("rx", position=mi.Point3f(0, 0, 0)))
    sc.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.frequency = fc
    for t in targets:
        sc.add(t)
    return sc


def gate_g6e2e(out_dir):
    """End-to-end: Doppler peak measured from the slow-time detector tensor."""
    v, n_frames, prf = 1.0, 128, 1280.0
    dt = 1.0 / prf
    rows, ok, spectra = [], True, None
    for fc in CARRIERS:
        lam = C / fc
        frames = [_solve(_monostatic(5.0 + v * n * dt, fc)) for n in range(n_frames)]
        H = slow_time_cfr(frames, fc, _offsets())
        x = H.sum(axis=0)
        fd = _peak_doppler(x, prf)
        ana = -2 * v / lam
        bin_hz = prf / n_frames
        ok &= abs(fd - ana) <= 1.0 * bin_hz and fd < 0
        rows.append({"fc_hz": fc, "doppler_hz": fd, "analytic_hz": ana,
                     "abs_err_hz": abs(fd - ana), "bin_hz": bin_hz})
        if fc == CARRIERS[0]:
            spectra = (x, prf)

    def draw(ax):
        x, p = spectra
        nfft = 8 * x.size
        X = np.fft.fftshift(np.fft.fft(x * np.hanning(x.size), n=nfft))
        f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / p))
        ax.plot(f, 20 * np.log10(np.abs(X) + 1e-30))
        ax.axvline(-2 * 1.0 / (C / CARRIERS[0]), color="r", ls="--", label="-2v/lambda")
        ax.set_xlim(-60, 60)
        ax.set_xlabel("Doppler (Hz)")
        ax.set_ylabel("|X| (dB)")
        ax.set_title("G6-e2e slow-time Doppler (3.5 GHz)")
        ax.legend(fontsize=7)

    return {"status": "PASS" if ok else "FAIL", "measured": rows,
            "expected": "peak = -2v/lambda within 1 bin; negative",
            "plot": _plot(out_dir, "g6e2e_doppler", draw)}


def gate_g6b(out_dir):
    """R1-A forearm path phase vs analytic kinematics; static R0/R1-P cancel."""
    fc = 3.5e9
    root = (5.0, 0.0, 0.0)
    n_frames, prf = 128, 1280.0
    dt = 1.0 / prf
    times = np.arange(n_frames) * dt
    sigmas = segment_sigmas(1.0, ["torso", "right_forearm"])
    frames, tau_ana, v_fa = [], [], []
    for t in times:
        s = analytic_arm_trajectory(root, np.array([t]), amplitude_m=0.25,
                                    omega_rad_s=2 * np.pi * 0.5)[0]
        sc = _targets_scene([
            build_point_target("torso", s.torso_position, sigmas["torso"]),
            build_point_target("right_forearm", s.forearm_position, sigmas["right_forearm"]),
        ])
        frames.append(_solve(sc))
        tau_ana.append(2 * float(np.linalg.norm(s.forearm_position)) / C)
        v_fa.append(s.forearm_velocity[0])
    tracks, events = track_frames(frames)
    fa_track = max(tracks, key=lambda tr: tr["key"][1][-1])
    ns = sorted(fa_track["samples"])
    tau_meas = np.array([fa_track["samples"][n]["tau"] for n in ns])
    tau_ana = np.array(tau_ana)[ns]
    lam = C / fc
    phi_meas = np.unwrap(-2 * np.pi * fc * tau_meas)
    phi_ana = np.unwrap(-2 * np.pi * fc * tau_ana)
    resid = phi_meas - phi_ana
    resid = resid - resid.mean()
    nmse = float(np.sum(resid ** 2) / np.sum((phi_ana - phi_ana.mean()) ** 2))
    t_s = np.array(ns) * dt
    fd_meas = np.gradient(phi_meas, t_s) / (2 * np.pi)
    fd_ana = -2 * np.array(v_fa)[ns] / lam
    corr = float(np.corrcoef(fd_meas, fd_ana)[0, 1])

    cfg = TargetConfig(model="native", rcs_tier="R0", random_sigma_s=False,
                       random_xpr=False, random_phases=False)
    static_scenes = {
        "R0": _targets_scene([build_human_target("human", root, cfg)]),
        "R1-P": _targets_scene([build_point_target("point", root, 1.0)]),
    }
    cancel = {}
    for name, sc in static_scenes.items():
        fr = [_solve(sc) for _ in range(8)]
        Hs = slow_time_cfr(fr, fc, _offsets())
        cancel[name] = float(np.sum(np.abs(Hs - Hs.mean(axis=1, keepdims=True)) ** 2)
                             / np.sum(np.abs(Hs) ** 2))

    ok = nmse <= 0.02 and corr > 0.99 and all(v <= 1e-6 for v in cancel.values())

    def draw(ax):
        ax.plot(t_s * 1e3, phi_meas - phi_meas[0], "o-", ms=3, label="RT forearm")
        ax.plot(t_s * 1e3, phi_ana - phi_ana[0], "--", label="analytic")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("phase (rad)")
        ax.set_title(f"G6b R1-A forearm phase (NMSE {nmse:.2e})")
        ax.legend(fontsize=7)

    return {"status": "PASS" if ok else "FAIL",
            "measured": {"phase_nmse": nmse, "doppler_corr": corr,
                         "static_cancel_nmse": cancel, "n_tracks": len(tracks),
                         "events": [list(e) for e in events[:20]]},
            "expected": "phase NMSE<=0.02; corr>0.99; static cancel NMSE<=1e-6",
            "plot": _plot(out_dir, "g6b_forearm_phase", draw)}


def gate_g6c(out_dir):
    """G6c: noise-free held-out motion at 28 GHz does not alias at PRF = 1280 Hz."""
    from fr3isac.sensing.rt_slowtime import synthesize_cfr

    fc, prf, nslow, dt = 28e9, 1280.0, 128, 1.0 / 1280.0
    v, R0 = 3.3, 5.0  # held-out peak radial velocity (m/s)
    amp = sensing_path_arrays(_solve(_monostatic(R0, fc)))["a"]
    amp = complex(amp[0]) if amp.size else 0.0 + 0.0j
    times = np.arange(nslow) * dt
    ranges = R0 + v * times
    H = synthesize_cfr(fc, _offsets(), np.full((1, nslow), amp), ranges[None, :])
    f_peak = _peak_doppler(H.sum(axis=0), prf)
    truth = -2 * v / (C / fc)
    bin_hz = prf / nslow
    # legacy 480 Hz rate would alias this Doppler (Nyquist 240 Hz)
    H_lo = synthesize_cfr(fc, _offsets(), np.full((1, nslow), amp),
                          (R0 + (v * prf / 480.0) * times)[None, :])
    f_alias = _peak_doppler(H_lo.sum(axis=0), 480.0)
    ok = abs(f_peak - truth) <= 1.0 * bin_hz

    def draw(ax):
        x = H.sum(axis=0)
        nfft = 8 * x.size
        X = np.fft.fftshift(np.fft.fft(x * np.hanning(x.size), n=nfft))
        f = np.fft.fftshift(np.fft.fftfreq(nfft, d=dt))
        ax.plot(f, 20 * np.log10(np.abs(X) + 1e-30))
        ax.axvline(truth, color="r", ls="--", label="truth -2v/lambda")
        ax.set_xlim(-700, 700)
        ax.set_xlabel("Doppler (Hz)")
        ax.set_ylabel("|X| (dB)")
        ax.set_title(f"G6c 28 GHz no aliasing (err {abs(f_peak - truth):.1f} Hz)")
        ax.legend(fontsize=7)

    return {"status": "PASS" if ok else "FAIL",
            "measured": {"prf_hz": prf, "nslow": nslow, "bin_hz": bin_hz,
                         "velocity_mps": v, "doppler_peak_hz": f_peak, "truth_hz": truth,
                         "abs_err_hz": abs(f_peak - truth),
                         "nyquist_hz": prf / 2,
                         "legacy_480hz_peak_hz": f_alias,
                         "legacy_aliases": abs(f_alias - truth) > 1.0 * bin_hz},
            "expected": "peak within 1 bin of -2v/lambda at PRF 1280 Hz",
            "plot": _plot(out_dir, "g6c_no_aliasing", draw)}


def gate_g8(out_dir):
    """Single metal wall: image-method reflected path into the detector input."""
    fc = 3.5e9
    rx = (0.0, 0.0, 0.5)
    wall_center_x, half_thickness = 2.0, 0.05
    plane_x = wall_center_x - half_thickness
    sc = Scene()
    sc.add(Transmitter("tx", position=mi.Point3f(0.0, 0.0, 0.0)))
    sc.add(Receiver("rx", position=mi.Point3f(*rx)))
    sc.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.frequency = fc
    wall = mi.load_dict({
        "type": "cube",
        "to_world": mi.ScalarTransform4f.translate([wall_center_x, 0, 0])
        @ mi.ScalarTransform4f.scale([half_thickness, 2.0, 2.0]),
    })
    obj = SceneObject(mi_mesh=wall, name="wall")
    obj.radio_material = ITURadioMaterial("metal_wall", "metal", thickness=0.05)
    sc.add(obj)
    paths = PathSolver(deterministic=True)(sc, max_depth=1, los=True, specular_reflection=True,
                                           diffuse_reflection=False, refraction=False,
                                           diffraction=False, seed=1)
    a = sensing_path_arrays(paths)
    tau_image = float(np.linalg.norm(np.array([2 * plane_x, 0.0, 0.0]) - np.array(rx)) / C)
    tau_meas = float(a["tau"][np.argmin(np.abs(a["tau"] - tau_image))]) if a["tau"].size else float("nan")
    err = abs(tau_meas - tau_image)
    H = cfr_passband(a["a"], a["tau"], fc, _offsets())
    ok = a["tau"].size >= 1 and err <= 1e-11

    def draw(ax):
        ax.stem(a["tau"] * 1e9, 20 * np.log10(np.abs(a["a"]) + 1e-30))
        ax.axvline(tau_image * 1e9, color="r", ls="--", label="image method")
        ax.set_xlabel("delay (ns)")
        ax.set_ylabel("path |a| (dB)")
        ax.set_title(f"G8 wall reflection (err {err * 1e9:.4f} ns)")
        ax.legend(fontsize=7)

    return {"status": "PASS" if ok else "FAIL",
            "measured": {"tau_meas_ns": tau_meas * 1e9, "tau_image_ns": tau_image * 1e9,
                         "abs_err_ns": err * 1e9, "n_paths": int(a["tau"].size),
                         "cfr_power": float(np.sum(np.abs(H) ** 2))},
            "expected": "specular wall path delay matches image method <=0.01 ns",
            "plot": _plot(out_dir, "g8_wall_image", draw)}


def gate_g9(out_dir):
    """C3 convergence: fast reconstruction NMSE + Doppler stability across dt."""
    fc, v, n, prf, dt, fine = 3.5e9, 1.0, 32, 1280.0, 1.0 / 1280.0, 4
    t_anchor = np.arange(n) * dt
    t_ref = np.arange(n * fine) * dt / fine
    H_ref = slow_time_cfr([_solve(_monostatic(5.0 + v * t, fc)) for t in t_ref], fc, _offsets())
    tracks, events = track_frames([_solve(_monostatic(5.0 + v * t, fc)) for t in t_anchor])
    H_fast = interpolate_cfr(tracks, t_anchor, t_ref, fc, _offsets())
    mask = (np.arange(n * fine) % fine) != 0
    nmse = reconstruction_nmse(H_fast[:, mask], H_ref[:, mask])
    peaks = []
    for f in (1, 2, 4):
        tt = np.arange(n * f) * dt / f
        HH = slow_time_cfr([_solve(_monostatic(5.0 + v * t, fc)) for t in tt], fc, _offsets())
        peaks.append(_peak_doppler(HH.sum(axis=0), prf * f))
    bin_hz = prf / n
    stable = (max(peaks) - min(peaks)) <= bin_hz
    ok = nmse <= 0.01 and stable

    def draw(ax):
        ax.plot(t_ref * 1e3, np.unwrap(np.angle(H_ref[32, :])), "-", label="reference")
        ax.plot(t_ref * 1e3, np.unwrap(np.angle(H_fast[32, :])), "--", label="fast")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("phase (rad)")
        ax.set_title(f"G9 fast vs reference (NMSE {nmse:.2e})")
        ax.legend(fontsize=7)

    return {"status": "PASS" if ok else "FAIL",
            "measured": {"reconstruction_nmse": nmse, "doppler_peaks_hz": peaks,
                         "doppler_spread_hz": max(peaks) - min(peaks), "bin_hz": bin_hz,
                         "n_tracks": len(tracks), "events": [list(e) for e in events[:20]]},
            "expected": "NMSE<=0.01; Doppler stable across dt,dt/2,dt/4 within 1 bin",
            "plot": _plot(out_dir, "g9_fast_reference", draw)}


GATES = {"G1": gate_g1, "G2": gate_g2, "G3": gate_g3, "G4": gate_g4,
         "G5": gate_g5, "G6": gate_g6, "G6-e2e": gate_g6e2e, "G6b": gate_g6b,
         "G6c": gate_g6c, "G7": gate_g7, "G8": gate_g8, "G9": gate_g9}


def run_gates(out_dir):
    out = Path(out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    results = {}
    for name, fn in GATES.items():
        try:
            results[name] = fn(out / "plots")
        except Exception as exc:  # noqa: BLE001
            results[name] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        print(f"{name}: {results[name]['status']}")
    payload = {"release": "v1.0-commag", "gates": results}
    (out / "gates.json").write_text(json.dumps(payload, indent=2))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="validation gate harness")
    parser.add_argument("--out", default="results/commag")
    args = parser.parse_args(argv)
    run_gates(args.out)


if __name__ == "__main__":
    main()
