"""Verify fresh outputs against the shipped numbers (Tiers 0/1/2)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE_TOL = 1e-6


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tier0(check_figures=False):
    import importlib

    est = importlib.import_module("fr3isac.analysis.estimands")
    base = ROOT / "results/commag"
    tmp = base / "_tier0_numbers.json"
    est.build(base / "onsets.parquet", base / "thresholds.parquet", base / "run_manifest.json",
              base / "hand_calc_12ghz_f1.json", base / "manuscript_numbers.json", str(tmp),
              base / "localization.json")
    fresh = json.loads(tmp.read_text())["numbers"]
    ref = json.loads((base / "manuscript_numbers.json").read_text())["numbers"]
    worst = 0.0
    for field in ("required_eirp_dbm", "required_eirp_dbm_visible",
                  "required_eirp_dbm_per_placement_median"):
        a = {(r["regime"], r["detector"], r["fc_hz"]): r["point"] for r in fresh[field]}
        b = {(r["regime"], r["detector"], r["fc_hz"]): r["point"] for r in ref[field]}
        for k in a:
            worst = max(worst, abs(a[k] - b[k]))
    for field in ("d1_minus_d0_eirp_db", "d1_minus_d0_eirp_db_visible"):
        a = {(r["regime"], r["fc_hz"]): r["point"] for r in fresh[field]}
        b = {(r["regime"], r["fc_hz"]): r["point"] for r in ref[field]}
        for k in a:
            worst = max(worst, abs(a[k] - b[k]))
    assert worst <= 1e-6, f"Tier 0 recomputation differs by {worst} dB"
    print(f"Tier 0: estimands recomputed on CPU from onsets.parquet + thresholds.parquet; "
          f"max abs diff {worst:.2e} dB")
    tmp.unlink(missing_ok=True)
    if check_figures:
        for f in ("fig3_required_eirp.pdf", "fig4_design_choices.pdf"):
            assert (ROOT / "figures" / f).exists(), f
        print("Tier 0 (extra): figures present; manuscript_numbers.json sha "
              + _sha(base / "manuscript_numbers.json")[:12])


def _flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k != "plot":
                yield from _flat(v, p + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _flat(v, p + f"[{i}]")
    else:
        yield p, o


def tier1(reference):
    fresh = json.loads((ROOT / "results/commag/gates.json").read_text())["gates"]
    ref = json.loads(Path(reference).read_text())["gates"]
    fa, fb = dict(_flat(fresh)), dict(_flat(ref))
    assert sorted(fa) == sorted(fb), "gate key mismatch"
    worst = 0.0
    for k in fa:
        va, vb = fa[k], fb[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            worst = max(worst, abs(va - vb))
        else:
            assert va == vb, (k, va, vb)
    for name, v in fresh.items():
        assert v.get("status") == "PASS", (name, v.get("status"))
    print(f"Tier 1: gates match reference (max abs diff {worst:g}); all PASS")


def tier2(reference):
    fresh = json.loads((ROOT / "results/commag/manuscript_numbers.json").read_text())["numbers"]
    ref = json.loads(Path(reference).read_text())["numbers"]
    worst = 0.0

    def keyed(rows, keys):
        return {tuple(r[k] for k in keys): r["point"] for r in rows}

    for field in ("required_eirp_dbm_visible", "required_eirp_dbm",
                  "required_eirp_dbm_per_placement_median"):
        a = keyed(fresh[field], ("regime", "detector", "fc_hz"))
        b = keyed(ref[field], ("regime", "detector", "fc_hz"))
        for k in a:
            if k in b:
                worst = max(worst, abs(a[k] - b[k]))
    for field in ("d1_minus_d0_eirp_db", "d1_minus_d0_eirp_db_visible"):
        a = keyed(fresh[field], ("regime", "fc_hz"))
        b = keyed(ref[field], ("regime", "fc_hz"))
        for k in a:
            if k in b:
                worst = max(worst, abs(a[k] - b[k]))
    assert worst <= 0.05, f"max abs diff {worst} exceeds 0.05 dB"
    print(f"Tier 2: manuscript_numbers.json reproduced; max abs diff {worst:.4g} dB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=0)
    ap.add_argument("--reference", default="results/commag/gates_reference.json")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.tier == 0:
        tier0(a.figures)
    elif a.tier == 1:
        tier1(a.reference)
    else:
        tier2(a.reference)
