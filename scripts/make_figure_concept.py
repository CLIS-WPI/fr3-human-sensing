"""Fig. 1 (conceptual, vector): one sensing task, three modeling choices, the design question.
No data are plotted; panel (c) is a schematic. Run: python make_figure_concept.py --out figures"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Polygon

NAVY, ORANGE, TEAL, GREY, LIGHT = "#1F3A5F", "#D9822B", "#2A9D8F", "#8A8F98", "#EEF2F7"
ORANGE_DARK = "#A65A12"
plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})


def person(ax, x, y, s=1.0, arm_angles=(-60, -20), hidden=False, alpha=1.0):
    c = GREY if not hidden else "#C9CDD3"
    ax.add_patch(Circle((x, y + 1.55 * s), 0.18 * s, color=c, alpha=alpha, zorder=3))
    ax.add_patch(FancyBboxPatch((x - 0.22 * s, y + 0.55 * s), 0.44 * s, 0.85 * s,
                                boxstyle="round,pad=0.02", color=c, alpha=alpha, zorder=3))
    ax.plot([x - 0.1 * s, x - 0.12 * s], [y + 0.55 * s, y], color=c, lw=3, alpha=alpha, zorder=2)
    ax.plot([x + 0.1 * s, x + 0.12 * s], [y + 0.55 * s, y], color=c, lw=3, alpha=alpha, zorder=2)
    sh = (x + 0.22 * s, y + 1.25 * s)
    for k, a in enumerate(arm_angles):
        r = np.deg2rad(a)
        e = (sh[0] + 0.55 * s * np.cos(r), sh[1] + 0.55 * s * np.sin(r))
        ax.plot([sh[0], e[0]], [sh[1], e[1]], color=ORANGE if not hidden else "#E9B98A",
                lw=2.4, alpha=(1.0 if k == len(arm_angles) - 1 else 0.35) * alpha, zorder=4,
                solid_capstyle="round")
    return sh


def panel_a(ax):
    ax.set_xlim(0, 10); ax.set_ylim(-1.4, 9); ax.axis("off")
    ax.text(0.1, 8.85, "(a) One sensing task", fontsize=9, fontweight="bold", va="top")
    ax.add_patch(Rectangle((0.3, 1.6), 9.4, 6.2, fill=False, ec="#B8BEC6", lw=0.8))
    ax.add_patch(Rectangle((0.35, 5.9), 0.45, 0.9, color=NAVY))
    for i in range(3):
        for j in range(2):
            ax.add_patch(Rectangle((0.4 + j * 0.2, 6.0 + i * 0.28), 0.12, 0.18, color="white"))
    ax.text(0.95, 7.1, "Base station", fontsize=7, color=NAVY)
    person(ax, 4.9, 2.0, s=1.7, arm_angles=(-75, -35))
    ax.annotate("", xy=(4.75, 4.45), xytext=(0.85, 6.3),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2))
    ax.annotate("", xy=(0.85, 6.0), xytext=(4.75, 4.15),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2, ls="--"))
    ax.text(1.4, 4.75, "sensing echo", fontsize=6.3, color=TEAL)
    ax.text(6.3, 3.35, "moving\nforearm", fontsize=6.5, color=ORANGE_DARK)
    ax.text(4.35, 2.9, "stationary\ntorso", fontsize=6.5, color="#555", ha="right")
    ax.add_patch(Rectangle((7.75, 1.6), 0.16, 4.0, color="#C9D3DE"))
    person(ax, 8.75, 1.9, s=1.0, arm_angles=(-60, -25), hidden=True)
    ax.text(8.75, 5.75, "alternative\noccluded\nplacement", fontsize=5.9, color="#777", ha="center", va="bottom")
    ax.add_patch(Rectangle((0.6, 2.0), 0.22, 0.38, color="#9AA5B1"))
    ax.text(0.95, 2.15, "communication user\n(context only)", fontsize=5.6, va="center", color="#555")
    ax.text(5.0, 1.0, "Same room. Same arm motion. Same deadline.", fontsize=7.4,
            ha="center", style="italic")
    ax.text(5.0, 0.3, "Detect motion onset within 200 ms", fontsize=8.2, ha="center",
            fontweight="bold")
    for k, (lab, col) in enumerate([("3.5 GHz", "#DCE6F2"), ("12 GHz", "#D8EFEA"),
                                    ("15 GHz", "#D8EFEA"), ("28 GHz", "#F6E3CF")]):
        ax.add_patch(FancyBboxPatch((0.6 + k * 2.35, -0.75), 2.0, 0.6,
                                    boxstyle="round,pad=0.04", fc=col, ec="#AAB"))
        ax.text(1.6 + k * 2.35, -0.45, lab, ha="center", va="center", fontsize=7)
    ax.plot([3.05, 7.65], [-0.95, -0.95], color=TEAL, lw=1)
    ax.text(5.35, -1.3, "FR3", ha="center", color=TEAL, fontsize=7.5, fontweight="bold")


def panel_b(ax):
    ax.set_xlim(0, 10); ax.set_ylim(-1.4, 9); ax.axis("off")
    ax.text(0.1, 8.85, "(b) Three modeling choices", fontsize=9, fontweight="bold", va="top")
    tops = [7.9, 4.95, 2.0]          # row title baselines
    for yt, title in zip(tops, ["Target", "Antenna", "Detector"]):
        ax.text(0.1, yt, title, fontsize=8, fontweight="bold", color=NAVY)
        ax.text(5.0, yt - 1.05, "vs", ha="center", fontsize=7.5, color="#666")
    for y in (5.35, 2.4):
        ax.plot([0.1, 9.9], [y, y], color="#D5DAE0", lw=0.6)
    # target row: icons between 6.6 and 7.7, caption at 5.85
    y = 6.55
    ax.add_patch(FancyBboxPatch((2.15, y), 0.6, 1.05, boxstyle="round,pad=0.02", color="#C8CDD4"))
    ax.add_patch(Circle((2.45, y + 0.55), 0.1, color=NAVY))
    ax.text(2.45, y - 0.2, "Stationary point:\nno independent limb motion", ha="center", va="top", fontsize=6.0)
    ax.add_patch(FancyBboxPatch((7.2, y), 0.6, 1.05, boxstyle="round,pad=0.02", color="#C8CDD4"))
    ax.plot([7.8, 8.4], [y + 0.85, y + 0.4], color=ORANGE, lw=2.2, alpha=0.35)
    ax.plot([7.8, 8.45], [y + 0.85, y + 0.8], color=ORANGE, lw=2.2)
    ax.text(7.7, y - 0.2, "Articulated target:\nmotion enters the echo", ha="center", va="top", fontsize=6.0)
    # antenna row: icons between 3.6 and 4.6, captions at 3.4
    y = 3.65
    for k in range(2):
        ax.add_patch(Rectangle((1.75 + k * 0.9, y), 0.5, 0.9, color=NAVY, alpha=0.45 + 0.4 * k))
    ax.text(2.45, y - 0.2, "Fixed gain\n(aperture shrinks with f)", ha="center", va="top", fontsize=6.0)
    for k, n in enumerate([3, 6]):
        x0 = 6.75 + k * 1.15
        ax.add_patch(Rectangle((x0, y), 0.9, 0.9, fill=False, ec=NAVY, lw=0.8))
        for gx in np.linspace(x0 + 0.1, x0 + 0.8, n):
            for gy in np.linspace(y + 0.1, y + 0.8, n):
                ax.add_patch(Rectangle((gx - 0.035, gy - 0.035), 0.07, 0.07, color=NAVY, lw=0))
    ax.text(7.75, y - 0.2, "Aperture-based gain scaling\n(idealized comparison)", ha="center", va="top", fontsize=6.0)
    # detector row: waves between 0.7 and 1.6, captions at 0.45
    y = 0.75
    t = np.linspace(0, 1, 200)
    for k in range(3):
        ax.plot(1.45 + 2.0 * t, y + 0.85 - k * 0.38 + 0.09 * np.sin(2 * np.pi * (2 + k) * t), color=NAVY, lw=0.9)
    ax.text(2.45, y - 0.2, "Baseline bank\n(coarse fixed grid)", ha="center", va="top", fontsize=6.0)
    # same count as the baseline (3): the redesign changes where templates sit, not how many
    shapes = (np.sin(2 * np.pi * (1.5 * t + 1.75 * t ** 2)),                       # up-chirp
              np.sin(2 * np.pi * 3 * t + 2.5 * np.sin(2 * np.pi * 0.8 * t)),        # swept (sinusoidal FM)
              np.sin(2 * np.pi * (5 * t - 1.5 * t ** 2)))                           # down-chirp
    for k, w in enumerate(shapes):
        ax.plot(6.7 + 2.0 * t, y + 0.85 - k * 0.38 + 0.09 * w, color=ORANGE, lw=0.9)
    ax.text(7.7, y - 0.2, "Redesigned bank\n(better-placed templates)", ha="center", va="top", fontsize=6.0)


def panel_c(ax_txt, ax):
    ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, "(c) The design question", fontsize=9, fontweight="bold", va="top",
                transform=ax_txt.transAxes)
    ax_txt.text(0.0, 0.55, "How much sensing EIRP\ndoes each band need?", fontsize=7.6,
                va="top", transform=ax_txt.transAxes)
    ax_txt.text(0.0, -0.02, "(conceptual illustration, not data)", fontsize=5.8, color="#777",
                va="top", transform=ax_txt.transAxes)
    f = np.linspace(0, 1, 100)
    ax.plot(f, 0.24 + 0.70 * f ** 1.4, color=NAVY, lw=1.6)
    ax.plot(f, 0.11 + 0.08 * f ** 8, color=ORANGE, lw=1.6, ls="--")  # flat across FR3, slight rise at the top end
    ax.text(0.03, 0.92, "fixed gain +\nbaseline bank", color=NAVY, fontsize=6.2, va="top")
    ax.text(0.98, 0.215, "aperture-based gain +\nredesigned bank", color=ORANGE_DARK, fontsize=5.6, fontweight="bold",
            ha="right", va="bottom")
    ax.set_xlabel("carrier frequency", fontsize=7); ax.set_ylabel("required EIRP", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)



def main(out: Path):
    fig = plt.figure(figsize=(7.16, 3.1))
    axa = fig.add_axes([0.0, 0.02, 0.37, 0.96])
    axb = fig.add_axes([0.375, 0.02, 0.36, 0.96])
    axct = fig.add_axes([0.765, 0.74, 0.23, 0.24])
    axc = fig.add_axes([0.79, 0.26, 0.20, 0.42])
    panel_a(axa); panel_b(axb); panel_c(axct, axc)
    fig.text(0.875, 0.04, "Interpret a band comparison\ntogether with its target, antenna,\nand detector assumptions.",
             ha="center", fontsize=6.4, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=NAVY, lw=0.7))
    for x in (0.372, 0.752):
        fig.add_artist(plt.Line2D([x, x], [0.05, 0.95], color="#D5DAE0", lw=0.7))
    fig.savefig(out / "fig1_concept.pdf", bbox_inches="tight")
    fig.savefig(out / "fig1_concept.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figures")
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True); main(out)
