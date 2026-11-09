"""Fig. 2 (tutorial): limb micro-Doppler vs carrier and baseline-bank capture.

Both panels use the test-motion class of the rerun (configs/rerun_v2.yaml):
amplitude U[0.10, 0.35] m, rate U[0.5, 1.5] Hz, random initial phase, onset within
one 50 ms hop. Panel (b) evaluates the baseline 36-template bank on the drawn test
motions themselves (same seeds as the run), purely radial, with the static return
subtracted as in the detector, using the best in-deadline 100 ms window. Capture is
relative to a perfectly matched filter on that window.

Usage (repository root):
    python manuscript/commag/scripts/make_figures_tutorial.py --out manuscript/commag/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fr3isac import geometry as geo
from fr3isac.experiments import rerun_v2_core as core

C = core.C
COLORS = {3.5e9: "#4C72B0", 12e9: "#DD8452", 15e9: "#55A868", 28e9: "#C44E52"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "pdf.fonttype": 42})


def test_motions(cfg):
    elig = core.eligible_grid(geo.load_descriptor(cfg["scene"]), cfg["grid_step_m"])
    _, test = core.split_placements(elig, cfg["n_test_placements"])
    ss = core.seed_sequences(cfg["master_seed"])
    return core.motions_per_placement(ss["test_motions"], test,
                                      cfg["k_test_motions_per_placement"], cfg).reshape(-1, 4)


def radial_records(fc, motions, cfg):
    """exp(j 4 pi d(t)/lambda) - 1 for the onset model (unit forearm return, static removed)."""
    t = core.record_times(cfg)
    out = []
    for a, f, p, t0 in motions:
        tau = np.clip(t - t0, 0.0, None)
        d = np.where(t > t0, a * (np.sin(2 * np.pi * f * tau + p) - np.sin(p)), 0.0)
        out.append(np.exp(1j * 4 * np.pi * fc / C * d) - 1.0)
    return out


def fig2(cfg, out: Path):
    mc, st = cfg["motion_class"], cfg["slow_time"]
    prf, cpi = st["prf_hz"], st["window_samples"] / st["prf_hz"]
    f = np.linspace(2e9, 30e9, 300)
    vmin = mc["amplitude_m"][0] * 2 * np.pi * mc["rate_hz"][0]
    vmax = mc["amplitude_m"][1] * 2 * np.pi * mc["rate_hz"][1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.15))
    for ax in (a1, a2):
        ax.axvspan(7.125, 24.25, color="#F3EFE0", zorder=0)
        ax.axvspan(7.125, 8.4, color="#E3D7AE", zorder=0)
        ax.axvspan(14.8, 15.35, color="#E3D7AE", zorder=0)
    a1.fill_between(f / 1e9, 2 * vmin * f / C, 2 * vmax * f / C, color="#4C72B0", alpha=0.35,
                    label="peak limb Doppler (test motions)")
    a1.axhline(prf / 2, ls="--", lw=0.8, color="k")
    a1.text(2.5, prf / 2 * 1.12, f"Nyquist of {prf:g} Hz sensing rate", fontsize=6.3)
    a1.axhline(1 / cpi, ls=":", lw=0.8, color="k")
    a1.text(16, 1 / cpi * 1.12, f"Doppler resolution ({1 / cpi:g} Hz)", fontsize=6.3)
    for fc in cfg["carriers_hz"]:
        a1.axvline(fc / 1e9, lw=0.5, color=COLORS.get(fc, "k"))
    a1.set_yscale("log")
    a1.set_ylim(1, 2000)
    a1.set_xlim(2, 30)
    a1.set_xlabel("carrier frequency (GHz)")
    a1.set_ylabel("Doppler (Hz)")
    a1.set_title("(a) Same motion, more Doppler at higher carrier")
    a1.text(20.5, 1350, "FR3 (7-24 GHz)", fontsize=6.3, ha="center")
    a1.legend(loc="lower right", frameon=False, fontsize=6.3)

    motions = test_motions(cfg)
    grid = np.array([3.5, 5, 7.5, 10, 12, 15, 18, 21, 24, 28]) * 1e9
    mean_db, p10_db = [], []
    for fc in grid:
        bank = core.bank_from_params(fc, core.baseline_params(cfg), cfg)
        cap = 10 ** (-core.capture_loss_db(bank, radial_records(fc, motions, cfg), motions[:, 3], cfg) / 10)
        mean_db.append(10 * np.log10(cap.mean()))
        p10_db.append(10 * np.log10(np.percentile(cap, 10)))
    a2.plot(grid / 1e9, mean_db, "o-", ms=3, lw=1, color="#33475B", label="mean")
    a2.plot(grid / 1e9, p10_db, "s--", ms=3, lw=1, color="#C44E52", label="10th percentile")
    a2.set_xlim(2, 30)
    a2.set_xlabel("carrier frequency (GHz)")
    a2.set_ylabel("template capture (dB)")
    a2.set_title("(b) Baseline 36-template bank: capture loss")
    a2.text(11.6, min(p10_db) + 0.3, "dark: WRC-27 candidates", fontsize=6.0, ha="center")
    a2.legend(loc="upper right", frameon=False)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "fig2_microdoppler_scaling.pdf", bbox_inches="tight")
    fig.savefig(out / "fig2_microdoppler_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {g / 1e9: (m, p) for g, m, p in zip(grid, mean_db, p10_db, strict=True)}, len(motions)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rerun_v2.yaml")
    ap.add_argument("--out", default="manuscript/commag/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap, n = fig2(core.load_cfg(a.config), out)
    print(f"baseline capture on {n} test motions (mean, p10 dB):",
          {k: (round(v[0], 2), round(v[1], 2)) for k, v in cap.items()})
