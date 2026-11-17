"""Figs. 2-4 from the rerun-v3 production run (results/rerun_v3/full/manuscript_numbers.json).

Fig. 2: (a) peak limb Doppler vs carrier (motion class); (b) template capture of the baseline
        bank and the baseline + one constant (zero-Doppler) template, on the v3 test motions
        (same seeds as the run), purely radial, static return removed, best in-deadline window.
Fig. 3: required EIRP at every carrier, K = inf, 200 ms; baseline / redesigned bank; fixed gain
        (solid) and aperture-based gain (dashed); 95% placement-bootstrap intervals.
Fig. 4: (a) redesigned-bank saving vs reference averaging K, one line per carrier (paired
        intervals); (b) extra EIRP of the glass-blocked onsets over visible onsets (redesigned).

Usage (repository root, in the container):
    python manuscript/commag/scripts/make_figures_v3.py --out manuscript/commag/figures_v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from make_figures_tutorial import radial_records

from fr3isac import geometry as geo
from fr3isac.experiments import rerun_v2_core as core
from fr3isac.experiments import rerun_v3 as v3

NAVY, ORANGE, GREY = "#1F3A5F", "#D9822B", "#8A8F98"
C = core.C
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif", "pdf.fonttype": 42})
FC = [3.5, 7.5, 12.0, 15.0, 28.0]
KS = ["64", "256", "1024", "inf"]
KX = {"64": 64, "256": 256, "1024": 1024, "inf": 8192}       # inf drawn at 8192 on the log axis
CARRIER_COL = dict(zip(FC, ["#1F3A5F", "#4F6D8F", "#8A8F98", "#E0A56A", "#D9822B"]))   # navy -> orange


def _save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def v3_test_motions(cfg):
    _, test = v3.test_placements(cfg, geo.load_descriptor(cfg["scene"]))
    return core.motions_per_placement(v3.streams(cfg["master_seed"])["test_motions"], test,
                                      cfg["k_test_motions_per_placement"], cfg).reshape(-1, 4)


def fig2(cfg, out):
    mc, st = cfg["motion_class"], cfg["slow_time"]
    prf, cpi = st["prf_hz"], st["window_samples"] / st["prf_hz"]
    f = np.linspace(2e9, 30e9, 300)
    vmin = mc["amplitude_m"][0] * 2 * np.pi * mc["rate_hz"][0]
    vmax = mc["amplitude_m"][1] * 2 * np.pi * mc["rate_hz"][1]
    LBL = dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8)  # carrier lines pass behind labels
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.15))
    for ax in (a1, a2):
        ax.axvspan(7.125, 24.25, color="#F3EFE0", zorder=0)
        ax.axvspan(7.125, 8.4, color="#E3D7AE", zorder=0)
        ax.axvspan(14.8, 15.35, color="#E3D7AE", zorder=0)
    a1.fill_between(f / 1e9, 2 * vmin * f / C, 2 * vmax * f / C, color="#4C72B0", alpha=0.35,
                    label="peak limb Doppler (test motions)")
    a1.axhline(prf / 2, ls="--", lw=0.8, color="k")
    a1.text(2.5, prf / 2 * 1.32, f"Nyquist of {prf:g} Hz sensing rate", fontsize=6.3, bbox=LBL)
    a1.axhline(1 / cpi, ls=":", lw=0.8, color="k")
    a1.text(16, 1 / cpi * 1.12, f"Doppler-bin spacing ({1 / cpi:g} Hz)", fontsize=6.3, bbox=LBL)
    for fc in cfg["carriers_hz"]:
        a1.axvline(fc / 1e9, lw=0.5, color=CARRIER_COL[fc / 1e9])
    a1.set_yscale("log")
    a1.set_ylim(1, 2000)
    a1.set_xlim(2, 30)
    a1.set_xlabel("carrier frequency (GHz)")
    a1.set_ylabel("Doppler (Hz)")
    a1.set_title("(a) Same motion, more Doppler at higher carrier")
    a1.text(21, 1150, "FR3 (7-24 GHz)", fontsize=6.3, ha="center", bbox=LBL)
    a1.legend(loc="lower right", fontsize=6.3, frameon=True, framealpha=1, facecolor="white", edgecolor="none")

    motions = v3_test_motions(cfg)
    L = st["window_samples"]
    grid = np.array([3.5, 5, 7.5, 10, 12, 15, 18, 21, 24, 28]) * 1e9
    res = {}
    for name in ("baseline", "base_plus_dc"):
        mean_db, p10_db = [], []
        for fc in grid:
            bank = core.bank_from_params(fc, core.baseline_params(cfg), cfg)
            if name == "base_plus_dc":
                bank = np.vstack([bank, np.ones((1, L), complex)])
            cap = 10 ** (-core.capture_loss_db(bank, radial_records(fc, motions, cfg), motions[:, 3], cfg) / 10)
            mean_db.append(10 * np.log10(cap.mean()))
            p10_db.append(10 * np.log10(np.percentile(cap, 10)))
        res[name] = (mean_db, p10_db)
    for name, col, lab in (("baseline", NAVY, "baseline"), ("base_plus_dc", ORANGE, "+ zero-Doppler")):
        m, p = res[name]
        a2.plot(grid / 1e9, m, "o-", ms=3, lw=1, color=col, label=f"{lab}, mean")
        a2.plot(grid / 1e9, p, "s--", ms=3, lw=1, color=col, mfc="white", label=f"{lab}, 10th pct.")
    a2.set_xlim(2, 30)
    a2.set_xlabel("carrier frequency (GHz)")
    a2.set_ylabel("template capture (dB)")
    a2.set_title("(b) Template capture: baseline bank vs. + zero-Doppler template")
    a2.set_ylim(-15.5, -0.3)
    a2.set_yticks([-15, -10, -5])
    a2.text(19.5, -1.4, "dark: WRC-27 candidates", fontsize=6.0, ha="center",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    fig.tight_layout(w_pad=1.5)
    # legend below panel (b), added after layout so the panel sizes stay unchanged
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), fontsize=5.8, ncol=2, columnspacing=1.2,
              handlelength=1.6, handletextpad=0.4, frameon=False)
    _save(fig, out, "fig2_microdoppler_scaling")
    return {f"{g / 1e9:g}": {k: (round(v[0][i], 2), round(v[1][i], 2)) for k, v in res.items()}
            for i, g in enumerate(grid)}, len(motions)


def load(path):
    n = json.loads(Path(path).read_text())["numbers"]
    req = {(r["regime"], r["detector"], r["K"], r["deadline_s"], r["carrier_ghz"]): r for r in n["required_eirp_dbm"]}
    sav = {(r["detector"], r["K"], r["deadline_s"], r["carrier_ghz"]): r for r in n["savings_vs_baseline_db"]}
    glass = {(r["detector"], r["carrier_ghz"]): r for r in n["glass_only_with_transmission"]}
    return req, sav, glass, n


def _err(v, lo, hi):
    return [[v - lo], [hi - v]]


def fig3(req, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    series = [("F1", "baseline", -0.21), ("F1", "redesigned", -0.07), ("F2", "baseline", 0.07),
              ("F2", "redesigned", 0.21)]
    lab = {"baseline": "baseline bank", "redesigned": "redesigned bank"}
    for reg, det, o in series:
        rows = [req[(reg, det, "inf", 0.2, f)] for f in FC]
        p = [r["required_eirp_dbm"] for r in rows]
        col = NAVY if det == "baseline" else ORANGE
        ax.errorbar([i + o for i in range(len(FC))], p,
                    yerr=[[a - r["ci95_dbm"][0] for a, r in zip(p, rows, strict=True)],
                          [r["ci95_dbm"][1] - a for a, r in zip(p, rows, strict=True)]],
                    fmt="o", ms=3.5, capsize=2, lw=0.9, color=col, mfc="white" if reg == "F1" else col,
                    ls="-" if reg == "F1" else "--",
                    label=f"{lab[det]}, {'fixed gain' if reg == 'F1' else 'aperture-based gain'}")
    ax.set_xticks(range(len(FC)))
    ax.set_xticklabels([f"{f:g} GHz" for f in FC])
    ax.set_ylabel("required EIRP (dBm)")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    vals = [r for k, r in req.items() if k[2] == "inf" and k[3] == 0.2 and k[1] in ("baseline", "redesigned")]
    ax.set_ylim(min(r["ci95_dbm"][0] for r in vals) - 3, max(r["ci95_dbm"][1] for r in vals) + 9)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    _save(fig, out, "fig3_required_eirp")


def fig4(sav, glass, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={"width_ratios": [1.35, 1]})
    for j, f in enumerate(FC):
        rows = [sav[("redesigned", k, 0.2, f)] for k in KS]
        x = np.array([KX[k] for k in KS]) * (1 + 0.035 * (j - 2))     # small dodge per carrier
        v = np.array([r["value_db"] for r in rows])
        lo = np.array([r["ci95_db"][0] for r in rows])
        hi = np.array([r["ci95_db"][1] for r in rows])
        a1.errorbar(x, v, yerr=[v - lo, hi - v], fmt="o-", ms=3, capsize=1.8, lw=1.0, color=CARRIER_COL[f],
                    label=f"{f:g} GHz")
    a1.axhline(0, color="k", lw=0.6)
    a1.axvline(256, ls=":", lw=0.9, color="#444")
    a1.text(256 * 1.08, 6.65, "K = 256 (0.2 s of frames)", fontsize=6.0, va="top", ha="left")
    a1.set_xscale("log")
    a1.set_xticks([KX[k] for k in KS])
    a1.set_xticklabels(["64", "256", "1024", r"$\infty$"])
    a1.minorticks_off()
    a1.set_xlabel("reference averaged over K frames")
    a1.set_ylabel("EIRP saved vs. baseline bank (dB)")
    a1.set_ylim(-3.6, 6.8)
    a1.set_title("(a) Redesigned-bank saving vs. reference quality")
    a1.legend(loc="lower right", frameon=False, fontsize=6.0, ncol=3, columnspacing=0.8, handlelength=1.4)

    rows = [glass[("redesigned", f)] for f in FC]
    v = np.array([r["extra_vs_visible_same_carrier"]["value_db"] for r in rows])
    lo = np.array([r["extra_vs_visible_same_carrier"]["ci95_db"][0] for r in rows])
    hi = np.array([r["extra_vs_visible_same_carrier"]["ci95_db"][1] for r in rows])
    a2.bar(range(len(FC)), v, width=0.6, color=ORANGE, alpha=0.85)
    a2.errorbar(range(len(FC)), v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", capsize=2.5, lw=0.8)
    for i, x in enumerate(v):
        a2.text(i, hi[i] + 0.8, f"{x:.1f}", ha="center", fontsize=6.5)
    a2.set_xticks(range(len(FC)))
    a2.set_xticklabels([f"{f:g}" for f in FC])
    a2.set_xlabel("carrier (GHz)")
    a2.set_ylabel("extra EIRP vs. visible onsets (dB)")
    a2.set_ylim(0, 38)
    n, npl = rows[0]["n_glass_only_onsets"], rows[0]["n_placements"]
    a2.set_title(f"(b) Forearm behind glass ({n} onsets, {npl} placements)")
    a2.text(0.02, 0.97, "redesigned bank; intervals rough\n(few placements)", transform=a2.transAxes,
            fontsize=6.0, va="top", color="#555")
    fig.tight_layout(w_pad=1.5)
    _save(fig, out, "fig4_design_choices")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rerun_v3.yaml")
    ap.add_argument("--numbers", default="results/rerun_v3/full/manuscript_numbers.json")
    ap.add_argument("--out", default="manuscript/commag/figures_v3")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = core.load_cfg(a.config)
    cap, n = fig2(cfg, out)
    print(f"template capture on {n} v3 test motions (mean, p10 dB):", cap)
    req, sav, glass, _ = load(a.numbers)
    fig3(req, out)
    fig4(sav, glass, out)
    print("wrote fig2-fig4 to", out)
