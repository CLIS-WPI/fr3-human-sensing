"""Verify fresh outputs against the shipped numbers (Tiers 0/1/2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/rerun_v2/full"
GATE_TOL = 1e-6


def tier0():
    from fr3isac.experiments import rerun_v2, rerun_v2_core

    cfg = rerun_v2_core.load_cfg(ROOT / "configs/rerun_v2.yaml")
    fresh = rerun_v2.build_numbers(cfg, RUN)["numbers"]
    ref = json.loads((RUN / "manuscript_numbers.json").read_text())["numbers"]
    ref.pop("determinism_numbers_stage", None)
    fa, fb = dict(_flat(fresh)), dict(_flat(ref))
    assert sorted(fa) == sorted(fb), "Tier 0: key mismatch with the shipped numbers"
    worst = 0.0
    for k in fa:
        if isinstance(fa[k], float) and isinstance(fb[k], float):
            worst = max(worst, abs(fa[k] - fb[k]))
        else:
            assert fa[k] == fb[k], (k, fa[k], fb[k])
    assert worst <= 1e-9, f"Tier 0: rebuilt numbers differ by {worst}"
    print(f"Tier 0: manuscript_numbers.json rebuilt on CPU from miss_curves.npz (max abs diff {worst:.1e})")


def _flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k not in ("plot", "manifest", "utc"):
                yield from _flat(v, p + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _flat(v, p + f"[{i}]")
    else:
        yield p, o


def tier1(reference):
    fresh = json.loads((RUN / "gates.json").read_text())["gates"]
    ref = json.loads(Path(reference).read_text())["gates"]
    for name, v in fresh.items():
        assert v.get("status") == "PASS", (name, v.get("status"))
    worst = 0.0
    for name in [g for g in ref if g not in ("G10", "G11")]:
        fa, fb = dict(_flat(fresh[name])), dict(_flat(ref[name]))
        assert sorted(fa) == sorted(fb), f"gate key mismatch in {name}"
        for k in fa:
            if isinstance(fa[k], (int, float)) and isinstance(fb[k], (int, float)):
                worst = max(worst, abs(fa[k] - fb[k]))
    assert worst <= GATE_TOL, f"gate values differ by {worst}"
    print(f"Tier 1: all gates PASS; G1-G9 match the reference (max abs diff {worst:g})")


def tier2(reference):
    def req(path):
        n = json.loads(Path(path).read_text())["numbers"]
        return {(r["regime"], r["bank"], r["fc_hz"]): r["required_eirp_dbm"]
                for r in n["required_eirp_dbm_operational"]}

    a, b = req(RUN / "manuscript_numbers.json"), req(reference)
    worst = max(abs(a[k] - b[k]) for k in b if a.get(k) is not None and b[k] is not None)
    def creq(path):
        return {(r["detector"], r["carrier_ghz"]): r["required_eirp_dbm"]
                for r in json.loads(path.read_text())["rows"]}

    for name in ("control", "review2"):
        d = ROOT / "results/rerun_v2" / name
        if (d / f"{name}_reference.json").exists():
            ca, cb = creq(d / f"{name}.json"), creq(d / f"{name}_reference.json")
            worst = max(worst, max(abs(ca[k] - cb[k]) for k in cb if cb[k] is not None))
    assert worst <= 0.05, f"max abs diff {worst} exceeds 0.05 dB"
    print(f"Tier 2: required EIRP reproduced; max abs diff {worst:.4g} dB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=0)
    ap.add_argument("--reference", default=str(RUN / "gates_reference.json"))
    a = ap.parse_args()
    {0: lambda: tier0(), 1: lambda: tier1(a.reference), 2: lambda: tier2(a.reference)}[a.tier]()
