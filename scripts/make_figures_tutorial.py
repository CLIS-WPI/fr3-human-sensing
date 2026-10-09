"""Schematic/tutorial figure: limb micro-Doppler and template capture vs carrier.

Fig. 2  tutorial: limb micro-Doppler vs carrier for the held-out motion class, and
        noise-free capture of the fixed 36-template bank (D0) on that class.
Figs. 3-4 are produced by make_figures_results_v7.py from manuscript_numbers.json.

Usage (from the repository root):
    python scripts/make_figures.py --results results/p4_v3 --out paper/figures

The template-bank definition below mirrors fr3isac.sensing.transient_detector
(TEMPLATE_AMPLITUDES_M, TEMPLATE_OMEGAS_RAD_S, TEMPLATE_PHASES, limb_chirp_template)
and the motion parameters mirror fr3isac.experiments.production.FROZEN_SEQUENCES.
If either changes, update the constants here (or import them directly).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

C = 299_792_458.0
# Frozen detector / motion constants (mirrored from the repository).
AMPS = (0.15, 0.22, 0.28)
OMEGAS = (2 * np.pi * 0.6, 2 * np.pi * 0.8, 2 * np.pi * 1.0)
PHASES = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
SEQS = {"seq1": (0.15, 2 * np.pi * 0.6), "seq2": (0.22, 2 * np.pi * 0.8),
        "seq3": (0.28, 2 * np.pi * 1.0), "seq4": (0.18, 2 * np.pi * 1.2),
        "seq5": (0.25, 2 * np.pi * 0.9)}
NSLOW, CPI = 128, 0.100
FS = NSLOW / CPI
CARRIERS = [3.5e9, 12e9, 15e9, 28e9]
CAR_LBL = {3.5e9: "3.5 GHz", 12e9: "12 GHz", 15e9: "15 GHz", 28e9: "28 GHz"}
COLORS = {3.5e9: "#4C72B0", 12e9: "#DD8452", 15e9: "#55A868", 28e9: "#C44E52"}
CEILING = 0.10

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "pdf.fonttype": 42})


def boot_ci(df: pd.DataFrame, col: str = "deadline_miss", b: int = 2000, seed: int = 1):
    """Scenario-cluster bootstrap 95% CI for a proportion."""
    rng = np.random.default_rng(seed)
    g = df.groupby("scenario_id")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    idx = rng.integers(0, len(s), (b, len(s)))
    est = s[idx].sum(1) / c[idx].sum(1)
    return float(df[col].mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def bank(fc: float) -> np.ndarray:
    lam = C / fc
    t = np.arange(NSLOW) * (CPI / NSLOW)
    return np.array([np.exp(1j * (4 * np.pi / lam) * a * np.sin(w * t + p))
                     for a in AMPS for w in OMEGAS for p in PHASES])


def capture_loss_db(fc: float, draws: int = 200, seed: int = 0):
    """Noise-free best-template energy capture for the frozen motion class."""
    rng = np.random.default_rng(seed)
    tn = bank(fc)
    tn = tn / np.linalg.norm(tn, axis=1, keepdims=True)
    lam = C / fc
    t = np.arange(NSLOW) / FS
    vals = []
    for a, w in SEQS.values():
        for _ in range(draws):
            s = np.exp(1j * (4 * np.pi / lam) * a * np.sin(w * t + rng.uniform(0, 2 * np.pi)))
            s /= np.linalg.norm(s)
            vals.append(np.max(np.abs(tn.conj() @ s) ** 2))
    v = np.asarray(vals)
    return 10 * np.log10(v.mean()), 10 * np.log10(np.percentile(v, 10))


def box(ax, x, y, w, h, text, fc="#EEF2FA", ec="#33475B", fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#33475B"))



# Held-out test-motion class (held-out): amplitude U[0.10,0.35] m, rate U[0.5,1.5] Hz.
HO_AMP = (0.10, 0.35)
HO_RATE = (0.5, 1.5)
PRF = 1280.0


def capture_heldout_db(fc: float, draws: int = 1500, seed: int = 0):
    """Noise-free best-template capture of the fixed bank (D0) on held-out motions."""
    rng = np.random.default_rng(seed)
    tn = bank(fc)
    tn = tn / np.linalg.norm(tn, axis=1, keepdims=True)
    lam = C / fc
    t = np.arange(NSLOW) / FS
    v = []
    for _ in range(draws):
        a = rng.uniform(*HO_AMP); w = 2 * np.pi * rng.uniform(*HO_RATE)
        s = np.exp(1j * (4 * np.pi / lam) * a * np.sin(w * t + rng.uniform(0, 2 * np.pi)))
        s /= np.linalg.norm(s)
        v.append(np.max(np.abs(tn.conj() @ s) ** 2))
    v = np.asarray(v)
    return 10 * np.log10(v.mean()), 10 * np.log10(np.percentile(v, 10))


def fig2(out: Path):
    f = np.linspace(2e9, 30e9, 300)
    vmin = HO_AMP[0] * 2 * np.pi * HO_RATE[0]
    vmax = HO_AMP[1] * 2 * np.pi * HO_RATE[1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.15))
    for ax in (a1, a2):
        ax.axvspan(7.125, 24.25, color="#F3EFE0", zorder=0)
        ax.axvspan(7.125, 8.4, color="#E3D7AE", zorder=0)
        ax.axvspan(14.8, 15.35, color="#E3D7AE", zorder=0)
    a1.fill_between(f / 1e9, 2 * vmin * f / C, 2 * vmax * f / C, color="#4C72B0", alpha=0.35,
                    label="peak limb Doppler (test motions)")
    a1.axhline(PRF / 2, ls="--", lw=0.8, color="k")
    a1.text(2.5, PRF / 2 * 1.12, "Nyquist of 1280 Hz sensing rate", fontsize=6.3)
    a1.axhline(1 / CPI, ls=":", lw=0.8, color="k")
    a1.text(16, 1 / CPI * 1.12, "Doppler resolution (10 Hz)", fontsize=6.3)
    for fc in CARRIERS:
        a1.axvline(fc / 1e9, lw=0.5, color=COLORS[fc])
    a1.set_yscale("log"); a1.set_ylim(1, 2000); a1.set_xlim(2, 30)
    a1.set_xlabel("carrier frequency (GHz)"); a1.set_ylabel("Doppler (Hz)")
    a1.set_title("(a) Same motion, more Doppler at higher carrier")
    a1.text(20.5, 1350, "FR3 (7-24 GHz)", fontsize=6.3, ha="center")
    a1.legend(loc="lower right", frameon=False, fontsize=6.3)
    grid = np.array([3.5, 5, 7.5, 10, 12, 15, 18, 21, 24, 28]) * 1e9
    m, p10 = zip(*(capture_heldout_db(fc) for fc in grid))
    a2.plot(grid / 1e9, m, "o-", ms=3, lw=1, color="#33475B", label="mean")
    a2.plot(grid / 1e9, p10, "s--", ms=3, lw=1, color="#C44E52", label="10th percentile")
    a2.set_xlim(2, 30)
    a2.set_xlabel("carrier frequency (GHz)"); a2.set_ylabel("template capture (dB)")
    a2.set_title("(b) Baseline 36-template bank: capture loss")
    a2.text(11.6, min(p10) + 0.3, "dark: WRC-27 candidates", fontsize=6.0, ha="center")
    a2.legend(loc="upper right", frameon=False)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "fig2_microdoppler_scaling.pdf", bbox_inches="tight")
    fig.savefig(out / "fig2_microdoppler_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return dict(zip([g / 1e9 for g in grid], zip(m, p10)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manuscript/commag/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = fig2(out)
    print("D0 capture on held-out motions (mean, p10 dB):",
          {k: (round(v[0], 2), round(v[1], 2)) for k, v in cap.items()})
