"""Figs. 3-4 for manuscript v7 (operational estimand).

Values are read directly from results/commag/manuscript_numbers.json
(required_eirp_dbm_operational, detector_savings_db). F2 is stored in the JSON by
the receive-gain shift, so no hard-coded numbers remain. Colour = detector bank
(navy baseline D0, orange redesigned Dfix); line style/fill = antenna regime.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, ORANGE, GREY = "#1F3A5F", "#D9822B", "#8A8F98"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7})
FC = [3.5, 12.0, 15.0, 28.0]


def load(path):
    """Return eirp[(regime, detector, fc)] = (dBm, ci_lo, ci_hi) and
    saving[(regime, comparison, fc)] = (dB, ci_lo, ci_hi) from the JSON."""
    n = json.load(open(path))["numbers"]
    eirp = {(r["regime"], r["detector"], r["fc_hz"] / 1e9):
            (r["required_eirp_dbm"], r["ci95_db"][0], r["ci95_db"][1])
            for r in n["required_eirp_dbm_operational"]}
    saving = {(r["regime"], r["comparison"], r["fc_hz"] / 1e9):
              (r["saving_db"], r["ci95_db"][0], r["ci95_db"][1])
              for r in n["detector_savings_db"]}
    return eirp, saving


def fig3(eirp, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    offs = {("F1", "D0"): -0.21, ("F1", "Dfix"): -0.07, ("F2", "D0"): 0.07, ("F2", "Dfix"): 0.21}
    lab = {"D0": "baseline bank", "Dfix": "redesigned bank"}
    for (reg, det), o in offs.items():
        vals = [eirp[(reg, det, f)] for f in FC]
        p = [v[0] for v in vals]
        x = [i + o for i in range(4)]
        ax.errorbar(x, p, yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    fmt="o", ms=3.5, capsize=2, lw=0.9, color=NAVY if det == "D0" else ORANGE,
                    mfc="white" if reg == "F1" else (NAVY if det == "D0" else ORANGE),
                    ls="-" if reg == "F1" else "--",
                    label=f"{lab[det]}, {'fixed gain' if reg == 'F1' else 'aperture-based gain'}")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{f:g} GHz" for f in FC])
    ax.set_ylabel("required EIRP (dBm)")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    allv = [v for v in eirp.values() if v[0] is not None]
    ax.set_ylim(min(v[1] for v in allv) - 3, max(v[2] for v in allv) + 8)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig3_required_eirp.pdf", bbox_inches="tight")
    fig.savefig(out / "fig3_required_eirp.png", dpi=200, bbox_inches="tight")


def fig4(eirp, saving, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.4), gridspec_kw={"width_ratios": [1, 1.6]})
    p = [saving[("F1", "Dfix-D0", f)][0] for f in FC]
    lo = [saving[("F1", "Dfix-D0", f)][1] for f in FC]
    hi = [saving[("F1", "Dfix-D0", f)][2] for f in FC]
    a1.bar(range(4), p, color=ORANGE, width=0.6, alpha=0.85, label="shared redesigned bank")
    a1.errorbar(range(4), p, yerr=[[a - b for a, b in zip(p, lo)], [b - a for a, b in zip(p, hi)]],
                fmt="none", ecolor="k", capsize=2.5, lw=0.8)
    a1.plot(range(4), [saving[("F1", "D1-D0", f)][0] for f in FC], "D", color=GREY, ms=4,
            label="per-carrier bank")
    for i, v in enumerate(p):
        a1.text(i, max(hi[i], 0) + 0.3, f"{v:.1f}", ha="center", fontsize=6.5)
    a1.axhline(0, color="k", lw=0.6)
    a1.set_xticks(range(4)); a1.set_xticklabels([f"{f:g}" for f in FC])
    a1.set_xlabel("carrier (GHz)"); a1.set_ylabel("EIRP saved vs. baseline bank (dB)")
    a1.set_ylim(-1.5, 10); a1.legend(loc="upper left", frameon=False)
    a1.set_title("(a) Redesigned bank")
    designs = [("F1", "D0", "fixed gain / baseline", NAVY, "white"),
               ("F1", "Dfix", "fixed gain / redesigned", ORANGE, "white"),
               ("F2", "D0", "aperture gain / baseline", NAVY, NAVY),
               ("F2", "Dfix", "aperture gain / redesigned", ORANGE, ORANGE)]
    w = 0.18
    for k, (reg, det, lbl, ec, fc) in enumerate(designs):
        vals = [eirp[(reg, det, f)][0] - eirp[(reg, det, 3.5)][0] for f in FC[1:]]
        xs = [i + (k - 1.5) * w for i in range(3)]
        a2.bar(xs, vals, width=w, color=fc, edgecolor=ec, lw=1.0, label=lbl)
        for x, v in zip(xs, vals):
            a2.text(x, max(v, 0) + (0.9 if v < 0.5 else 0.4), f"{v:.1f}", ha="center", fontsize=5.6)
    for i, f in enumerate(FC[1:]):
        a2.plot([i - 0.42, i + 0.42], [20 * math.log10(f / 3.5)] * 2, ls=":", lw=1.0, color="#444")
    a2.plot([], [], ls=":", color="#444", lw=1.0, label=r"$\lambda^2$ term (fixed gain)")
    a2.axhline(0, color="k", lw=0.6)
    a2.set_xticks(range(3)); a2.set_xticklabels([f"{f:g} GHz" for f in FC[1:]])
    a2.set_ylabel("extra EIRP vs. 3.5 GHz (dB)"); a2.set_ylim(-1.5, 30)
    a2.set_title("(b) Cost of moving up from 3.5 GHz", pad=24)
    a2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.0,
              ncol=3, columnspacing=0.9, handlelength=1.6)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "fig4_design_choices.pdf", bbox_inches="tight")
    fig.savefig(out / "fig4_design_choices.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", default="results/commag/manuscript_numbers.json")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eirp, saving = load(a.numbers)
    fig3(eirp, out)
    fig4(eirp, saving, out)
    print("wrote fig3_required_eirp and fig4_design_choices from", a.numbers)
