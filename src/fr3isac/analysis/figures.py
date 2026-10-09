"""Paper figures (Fig. 3 and Fig. 4) from the manuscript numbers file.

Reads only ``manuscript_numbers.json`` and recomputes no statistic. Writes vector
PDF and 200-dpi PNG figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CARRIERS = [3.5e9, 12e9, 15e9, 28e9]
CAR_LBL = {3.5e9: "3.5", 12e9: "12", 15e9: "15", 28e9: "28"}
COLORS = {3.5e9: "#4C72B0", 12e9: "#DD8452", 15e9: "#55A868", 28e9: "#C44E52"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "pdf.fonttype": 42})


def _req(numbers, fc, model, detector, regime):
    return next((r for r in numbers["required_eirp_dbm"]
                 if r["fc_hz"] == fc and r["model"] == model
                 and r["detector"] == detector and r["regime"] == regime), None)


def _diff(numbers, fc, regime):
    return next((r for r in numbers["d1_minus_d0_eirp_db"]
                 if r["fc_hz"] == fc and r["regime"] == regime), None)


def _rcs(numbers, fc, detector, exp):
    return next((r for r in numbers["rcs_exponent_required_eirp_dbm"]
                 if r["fc_hz"] == fc and r["detector"] == detector
                 and r["rcs_exponent"] == exp), None)


def _capture_dev(numbers, detector, fc):
    return numbers["detector_banks"][f"{detector}_capture_loss_db"]["dev"][f"{fc/1e9:g}"]["mean_capture_loss_db"]


def _inr(numbers, fc, detector):
    return next((r for r in numbers["inr"] if r["fc_hz"] == fc and r["detector"] == detector), None)


def fig3(numbers, out):
    """Required EIRP vs carrier, R1-A, D0/D1, F1/F2, with 95% CIs."""
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    series = [("D0", "F1", "none", "-", -0.08, "#33475B"),
              ("D1", "F1", "full", "-", 0.08, "#C44E52"),
              ("D0", "F2", "none", "--", -0.08, "#33475B"),
              ("D1", "F2", "full", "--", 0.08, "#C44E52")]
    for det, reg, face, ls, dx, col in series:
        xs, ys, lo, hi = [], [], [], []
        for fc in CARRIERS:
            r = _req(numbers, fc, "R1-A", det, reg)
            if not r:
                continue
            xs.append(fc / 1e9 + dx)
            ys.append(r["point"])
            lo.append(r["point"] - r["ci95_low"])
            hi.append(r["ci95_high"] - r["point"])
        ax.errorbar(xs, ys, yerr=[lo, hi], ls=ls, color=col, marker="o",
                    markerfacecolor=(col if face == "full" else "none"),
                    markeredgecolor=col, ms=3.5, lw=1, capsize=2,
                    label=f"{det}, {reg}")
    ax.set_xticks([fc / 1e9 for fc in CARRIERS])
    ax.set_xticklabels([CAR_LBL[fc] for fc in CARRIERS])
    ax.set_xlabel("carrier (GHz)")
    ax.set_ylabel("required EIRP (dBm)\nfor miss $\\leq$ 0.10 in 200 ms")
    ax.set_title("R1-A required EIRP (95% CI)")
    ax.text(0.03, 0.03, "R0, R1-P: not achievable at any power", transform=ax.transAxes,
            fontsize=6.0, style="italic")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig3_required_eirp.pdf")
    fig.savefig(out / "fig3_required_eirp.png", dpi=200)
    plt.close(fig)


def fig4(numbers, out):
    """(a) D1-over-D0 saving + capture loss; (b) RCS-trend shift; (c) tolerable INR."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 2.3),
                                     gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    x = np.arange(len(CARRIERS))
    # (a) saving = -(D1 - D0); positive = D1 needs less EIRP
    for i, reg in enumerate(("F1", "F2")):
        vals, lo, hi = [], [], []
        for fc in CARRIERS:
            d = _diff(numbers, fc, reg)
            vals.append(-d["point"])
            lo.append(d["ci95_high"] - d["point"])
            hi.append(d["point"] - d["ci95_low"])
        a1.bar(x + (i - 0.5) * 0.36, vals, width=0.36, color=["#33475B", "#8899AA"][i],
               label=f"{reg}", yerr=[lo, hi], capsize=2, error_kw={"lw": 0.7})
    for j, fc in enumerate(CARRIERS):
        a1.plot(j, -_capture_dev(numbers, "D0", fc), marker="x", color="grey", ms=5)
        a1.plot(j, -_capture_dev(numbers, "D1", fc), marker="D", color="grey", ms=4)
    a1.axhline(0, color="k", lw=0.6)
    a1.set_xticks(x)
    a1.set_xticklabels([CAR_LBL[fc] for fc in CARRIERS])
    a1.set_xlabel("carrier (GHz)")
    a1.set_ylabel("EIRP saving D1 vs D0 (dB)")
    a1.set_title("(a) D1 advantage")
    a1.legend(frameon=False, fontsize=6)
    a1.grid(True, alpha=0.25, axis="y")
    a1.text(0.02, 0.95, "grey x: D0 capture loss\ngrey D: D1 capture loss", transform=a1.transAxes,
            fontsize=5.5, va="top", color="grey")
    # (b) RCS-trend shift of required EIRP (D1, F1) relative to f^0
    for exp, mk in ((-1, "s"), (1, "^")):
        ys = []
        for fc in CARRIERS:
            b0 = _rcs(numbers, fc, "D1", 0)
            be = _rcs(numbers, fc, "D1", exp)
            ys.append(be["required_eirp_dbm"] - b0["required_eirp_dbm"])
        a2.plot(x, ys, marker=mk, ls="none", color="#33475B", label=f"$f^{{{exp}}}$")
    a2.axhline(0, color="k", lw=0.6)
    a2.set_xticks(x)
    a2.set_xticklabels([CAR_LBL[fc] for fc in CARRIERS])
    a2.set_xlabel("carrier (GHz)")
    a2.set_ylabel("shift of required EIRP (dB)")
    a2.set_title("(b) RCS trend (D1, F1)")
    a2.legend(frameon=False, fontsize=6)
    a2.grid(True, alpha=0.25, axis="y")
    a2.text(0.03, 0.03, "3.5 GHz = 0 by construction", transform=a2.transAxes, fontsize=5.5,
            style="italic")
    # (c) largest tolerable INR + interference-off baseline
    for i, det in enumerate(("D0", "D1")):
        vals = []
        for fc in CARRIERS:
            r = _inr(numbers, fc, det)
            vals.append(r["max_tolerable_inr_db"] if r and r["max_tolerable_inr_db"] is not None else np.nan)
        a3.bar(x + (i - 0.5) * 0.36, vals, width=0.36, color=["#33475B", "#C44E52"][i], label=det)
    a3.axhline(0, color="k", lw=0.6)
    a3.set_xticks(x)
    a3.set_xticklabels([CAR_LBL[fc] for fc in CARRIERS])
    a3.set_xlabel("carrier (GHz)")
    a3.set_ylabel("tolerable INR (dB)")
    a3.set_title("(c) Interference tolerance")
    a3.legend(frameon=False, fontsize=6)
    a3.grid(True, alpha=0.25, axis="y")
    a3.text(0.03, 0.95, "baseline: interference off\n(0 dB)", transform=a3.transAxes,
            fontsize=5.5, va="top", style="italic")
    fig.tight_layout(w_pad=1.4)
    fig.savefig(out / "fig4_detector_robustness.pdf")
    fig.savefig(out / "fig4_detector_robustness.png", dpi=200)
    plt.close(fig)


def generate(numbers_path="results/commag/manuscript_numbers.json",
             out_dir="manuscript/commag/figures"):
    numbers = json.loads(Path(numbers_path).read_text())["numbers"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig3(numbers, out)
    fig4(numbers, out)
    manifest = {"source": numbers_path,
                "figures": ["fig3_required_eirp.pdf/png", "fig4_detector_robustness.pdf/png"],
                "caption_data": {"receive_gain_db": numbers["antennas"]["receive_gain_db"],
                                 "transmit_gain_db": numbers["antennas"]["transmit_gain_db"],
                                 "noise_figure_db": numbers["antennas"]["noise_figure_db"]}}
    (out / "m5_figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="paper figures")
    parser.add_argument("--numbers", default="results/commag/manuscript_numbers.json")
    parser.add_argument("--out", default="manuscript/commag/figures")
    args = parser.parse_args(argv)
    print("wrote:", generate(args.numbers, args.out)["figures"])


if __name__ == "__main__":
    main()
