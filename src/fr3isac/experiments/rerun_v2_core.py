"""Rerun v2 core (numpy only, no ray tracer): data split, motions, windowed deadline
detection, thresholds, event-level false alarms and template-bank selection.

Constants come from ``configs/rerun_v2.yaml``; nothing scientific is hard-coded here.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from fr3isac import geometry as geo

C = 299_792_458.0
SEED_STREAMS = ("dev_motions", "test_motions", "design_candidates", "noise_trials",
                "calibration", "validation", "fa_stream", "point_targets", "bootstrap")


def load_cfg(path="configs/rerun_v2.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Seeds and data split
# --------------------------------------------------------------------------- #
def seed_sequences(master_seed: int) -> dict[str, np.random.SeedSequence]:
    """One independent SeedSequence per stream (distinct spawn keys => disjoint streams)."""
    return {name: np.random.SeedSequence(master_seed, spawn_key=(i,))
            for i, name in enumerate(SEED_STREAMS)}


def eligible_grid(descriptor: dict, step_m: float) -> list[tuple[float, float]]:
    """Grid over the human ROI kept iff the BS->torso segment is unobstructed.

    Same rule and iteration order as ``localization.eligible_scenarios``.
    """
    bs = geo.bs_position(descriptor)
    blk = geo.blockers(descriptor)
    out = []
    x = geo.ROI_X[0]
    while x <= geo.ROI_X[1] + 1e-9:
        y = geo.ROI_Y[0]
        while y <= geo.ROI_Y[1] + 1e-9:
            t = np.array([x, y, geo.TARGET_HEIGHT_M])
            if not any(geo.segment_aabb_intersect(bs, t, o["center_m"], o["size_m"]) for o in blk):
                out.append((float(x), float(y)))
            y += step_m
        x += step_m
    return out


def split_placements(eligible, n_test: int):
    """Test = the paper's evenly spaced subset; development = every other eligible placement."""
    idx = np.linspace(0, len(eligible) - 1, min(n_test, len(eligible))).round().astype(int)
    test = [eligible[i] for i in idx]
    tset = set(test)
    dev = [p for p in eligible if p not in tset]
    return dev, test


def assert_disjoint(dev, test):
    overlap = set(map(tuple, dev)) & set(map(tuple, test))
    if overlap:
        raise AssertionError(f"development/test placement overlap: {sorted(overlap)}")


# --------------------------------------------------------------------------- #
# Motions
# --------------------------------------------------------------------------- #
def draw_motions(ss: np.random.SeedSequence, n: int, cfg: dict) -> np.ndarray:
    """(n, 4) array of [amplitude_m, rate_hz, phase_rad, onset_s] from the stated class."""
    rng = np.random.default_rng(ss)
    mc, st = cfg["motion_class"], cfg["slow_time"]
    return np.column_stack([rng.uniform(*mc["amplitude_m"], n), rng.uniform(*mc["rate_hz"], n),
                            rng.uniform(*mc["phase_rad"], n), rng.uniform(*st["onset_window_s"], n)])


def motions_per_placement(ss: np.random.SeedSequence, placements, k: int, cfg: dict) -> np.ndarray:
    """(n_placements, k, 4); each placement gets its own child stream (order-independent)."""
    kids = ss.spawn(len(placements))
    return np.stack([draw_motions(kids[i], k, cfg) for i in range(len(placements))])


# --------------------------------------------------------------------------- #
# Slow time and windows
# --------------------------------------------------------------------------- #
def record_times(cfg: dict) -> np.ndarray:
    st = cfg["slow_time"]
    return st["record_start_s"] + np.arange(st["record_samples"]) / st["prf_hz"]


def window_starts(cfg: dict) -> np.ndarray:
    st = cfg["slow_time"]
    return np.arange(0, st["record_samples"] - st["window_samples"] + 1, st["hop_samples"])


def window_end_times(cfg: dict) -> np.ndarray:
    """Decision time of each window = time of its last sample (causal)."""
    st = cfg["slow_time"]
    return record_times(cfg)[window_starts(cfg) + st["window_samples"] - 1]


def deadline_mask(onset_s: float, cfg: dict) -> np.ndarray:
    """Windows that end after the onset and no later than onset + deadline."""
    e = window_end_times(cfg)
    return (e > onset_s) & (e <= onset_s + cfg["slow_time"]["deadline_s"])


def windows(x: np.ndarray, cfg: dict) -> np.ndarray:
    """(n_win, L, ...) overlapping windows of a slow-time record (axis 0 = time)."""
    L = cfg["slow_time"]["window_samples"]
    return np.stack([x[s:s + L] for s in window_starts(cfg)])


def template(fc_hz, amplitude_m, rate_hz, phase_rad, L, prf_hz) -> np.ndarray:
    t = np.arange(L) / prf_hz
    return np.exp(1j * (4.0 * np.pi * fc_hz / C) * amplitude_m * np.sin(2 * np.pi * rate_hz * t + phase_rad))


def bank_from_params(fc_hz, params, cfg) -> np.ndarray:
    st = cfg["slow_time"]
    return np.asarray([template(fc_hz, a, f, p, st["window_samples"], st["prf_hz"]) for a, f, p in params])


def baseline_params(cfg) -> np.ndarray:
    b = cfg["banks"]["baseline"]
    return np.asarray([(a, f, p) for a in b["amplitudes_m"] for f in b["rates_hz"] for p in b["phases_rad"]])


# --------------------------------------------------------------------------- #
# Statistic, thresholds, false alarms
# --------------------------------------------------------------------------- #
def window_stat(bank: np.ndarray, X: np.ndarray, s2: float) -> np.ndarray:
    """max_t |<t, x>|^2 / (||t||^2 s2) for each column of X (L, n) -> (n,)."""
    norm = np.sum(np.abs(bank) ** 2, axis=1)
    return (np.abs(bank.conj() @ X) ** 2 / (norm[:, None] * s2)).max(axis=0)


def _cn(rng, shape, s2):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * math.sqrt(s2 / 2.0)


def null_stats(bank, s2, n, ss, chunk=100_000) -> np.ndarray:
    """H0 per-window statistic over n independent noise-only windows."""
    rng = np.random.default_rng(ss)
    L = bank.shape[1]
    out = np.empty(n)
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        out[i:i + m] = window_stat(bank, _cn(rng, (L, m), s2), s2)
    return out


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def calibrate(bank, s2, cfg, ss_cal, ss_val) -> dict:
    d = cfg["detector"]
    pfa = d["pfa_per_window"]
    thr = float(np.quantile(null_stats(bank, s2, d["n_calibration_windows"], ss_cal), 1.0 - pfa))
    val = null_stats(bank, s2, d["n_validation_windows"], ss_val)
    fa = int(np.sum(val > thr))
    lo, hi = wilson(fa, val.size)
    # K-frame averaged reference: change variance s2 (1 + 1/K) at the same threshold
    refavg = {}
    for k in d.get("reference_averaging_frames", []):
        n = int(np.sum(val > thr / (1.0 + 1.0 / k)))
        refavg[str(k)] = {"pfa": n / val.size, "wilson_ci95": list(wilson(n, val.size))}
    return {"threshold": thr, "calibration_N": d["n_calibration_windows"],
            "validation_N": int(val.size), "false_alarms": fa, "pfa": fa / val.size,
            "wilson_ci95": [lo, hi], "accept": bool(lo <= pfa <= hi), "reference_averaging": refavg}


def event_false_alarms(bank, s2, thr, cfg, ss, chunk_windows=50_000) -> dict:
    """Continuous no-motion stream: windows overlap and share samples; an alarm event is
    a maximal run of consecutive crossing windows."""
    st = cfg["slow_time"]
    L, H = st["window_samples"], st["hop_samples"]
    hours = cfg["detector"]["fa_stream_hours"]
    n_win = int(round(hours * 3600.0 * st["prf_hz"] / H))
    rng = np.random.default_rng(ss)
    tail = _cn(rng, (L - H,), s2)
    prev = False
    crossings = events = done = 0
    while done < n_win:
        m = min(chunk_windows, n_win - done)
        x = np.concatenate([tail, _cn(rng, (m * H,), s2)])
        idx = np.arange(m)[:, None] * H + np.arange(L)[None, :]
        c = window_stat(bank, x[idx].T, s2) > thr
        crossings += int(c.sum())
        events += int(c[0] and not prev) + int(np.sum(c[1:] & ~c[:-1]))
        prev = bool(c[-1])
        tail = x[-(L - H):]
        done += m
    return {"hours": hours, "windows": n_win, "window_crossings": crossings,
            "events": events, "crossings_per_hour": crossings / hours,
            "events_per_hour": events / hours,
            "window_pfa": crossings / n_win}


# --------------------------------------------------------------------------- #
# Deadline miss and required EIRP
# --------------------------------------------------------------------------- #
def miss_curve(sig, onset_s, bank, s2, thr, amps, X, cfg) -> np.ndarray:
    """Miss probability vs amplitude scale for one onset.

    ``sig`` (n_samp,) noise-free detector-input record at nominal EIRP; ``X`` (n_samp, T)
    one continuous noise record per trial, shared by all overlapping windows. A trial
    succeeds iff some window ending in (onset, onset + deadline] crosses the threshold
    (equivalently: the first post-onset crossing is within the deadline).
    """
    m = deadline_mask(onset_s, cfg)
    S = windows(sig, cfg)[m]                 # (w, L)
    N = windows(X, cfg)[m]                   # (w, L, T)
    norm = np.sum(np.abs(bank) ** 2, axis=1)
    cs = S @ bank.conj().T                   # (w, M)
    cn = np.einsum("ml,wlt->wmt", bank.conj(), N)
    out = np.empty(len(amps))
    for j, a in enumerate(amps):
        stat = (np.abs(a * cs[:, :, None] + cn) ** 2 / (norm[None, :, None] * s2)).max(axis=1)
        out[j] = 1.0 - np.mean((stat > thr).any(axis=0))
    return out


def noisefree_required_db(sig, onset_s, bank, s2, thr, cfg) -> float:
    """dB re nominal at which the noise-free best in-deadline window reaches the threshold."""
    S = windows(sig, cfg)[deadline_mask(onset_s, cfg)]
    s = float(window_stat(bank, S.T, s2).max())
    return 10 * math.log10(thr / s) if s > 0 else math.inf


def crossing(mean_miss, grid, target=0.10):
    for i in range(len(grid) - 1):
        a, b = mean_miss[i], mean_miss[i + 1]
        if (a - target) * (b - target) <= 0 and a != b:
            return float(grid[i] + (target - a) * (grid[i + 1] - grid[i]) / (b - a))
    return None


# --------------------------------------------------------------------------- #
# Capture and bank selection
# --------------------------------------------------------------------------- #
def capture_matrix(cand_bank, sigs, onsets, cfg) -> np.ndarray:
    """cap[i, o] = max over in-deadline windows w of |<t_i, s_ow>|^2 / (||t_i||^2 max_w ||s_ow||^2).

    Energy captured by template i relative to a perfectly matched filter on the best window.
    """
    norm = np.sum(np.abs(cand_bank) ** 2, axis=1)
    out = np.zeros((cand_bank.shape[0], len(sigs)), dtype=np.float32)
    for o, (s, t0) in enumerate(zip(sigs, onsets, strict=True)):
        S = windows(s, cfg)[deadline_mask(t0, cfg)]
        e = float(np.max(np.sum(np.abs(S) ** 2, axis=1)))
        if e <= 0:
            continue
        out[:, o] = (np.abs(cand_bank.conj() @ S.T) ** 2).max(axis=1) / (norm * e)
    return out


def greedy_facility(cap: np.ndarray, k: int) -> list[int]:
    """Greedy maximisation of sum_o max_{i in S} cap[i, o] (submodular; (1-1/e) guarantee)."""
    cur = np.zeros(cap.shape[1], dtype=cap.dtype)
    chosen: list[int] = []
    for _ in range(k):
        gain = np.maximum(cap - cur[None, :], 0).sum(axis=1)
        gain[chosen] = -1
        i = int(np.argmax(gain))
        chosen.append(i)
        cur = np.maximum(cur, cap[i])
    return chosen


def capture_loss_db(bank, sigs, onsets, cfg) -> np.ndarray:
    cap = capture_matrix(bank, sigs, onsets, cfg).max(axis=0)
    return -10 * np.log10(np.maximum(cap, 1e-30))


def candidate_params(ss, cfg) -> np.ndarray:
    r = cfg["banks"]["redesigned"]
    rng = np.random.default_rng(ss)
    n = r["n_candidates"]
    return np.column_stack([rng.uniform(*r["candidate_amplitude_m"], n),
                            rng.uniform(*r["candidate_rate_hz"], n),
                            rng.uniform(0.0, 2 * np.pi, n)])


def constant_term_share(sigs, onsets, cfg) -> np.ndarray:
    """Energy share of the window mean (constant term left by static subtraction) in the
    best in-deadline window; a chirp template cannot capture it."""
    out = []
    for s, t0 in zip(sigs, onsets, strict=True):
        W = windows(s, cfg)[deadline_mask(t0, cfg)]
        e = np.sum(np.abs(W) ** 2, axis=1)
        w = W[int(np.argmax(e))]
        tot = float(np.sum(np.abs(w) ** 2))
        out.append(abs(w.mean()) ** 2 * w.size / tot if tot > 0 else np.nan)
    return np.asarray(out)


def coverage_search(cap_blocks, tol_db, k_ref, k_max):
    """Greedy facility location on the column-concatenated blocks (one per carrier).

    Returns the first ``k_ref`` chosen indices, the per-block P90 capture loss of the
    pool (best candidate per onset) and of the greedy bank at ``k_ref``, and the smallest
    greedy size whose P90 loss is within ``tol_db`` of the pool at every block (None if
    not reached by ``k_max``)."""
    cap = np.concatenate(cap_blocks, axis=1)
    edges = np.cumsum([0] + [b.shape[1] for b in cap_blocks])

    def p90(cur):
        loss = -10 * np.log10(np.maximum(cur, 1e-30))
        return [float(np.percentile(loss[edges[i]:edges[i + 1]], 90)) for i in range(len(cap_blocks))]

    pool = p90(cap.max(axis=0))
    cur = np.zeros(cap.shape[1], dtype=cap.dtype)
    chosen, at_ref, smallest = [], None, None
    for k in range(1, min(k_max, cap.shape[0]) + 1):
        gain = np.maximum(cap - cur[None, :], 0).sum(axis=1)
        gain[chosen] = -1
        i = int(np.argmax(gain))
        chosen.append(i)
        cur = np.maximum(cur, cap[i])
        if k == k_ref or smallest is None:
            q = p90(cur)
            if k == k_ref:
                at_ref = q
            if smallest is None and all(a <= b + tol_db for a, b in zip(q, pool, strict=True)):
                smallest = k
        if k >= k_ref and smallest is not None:
            break
    return {"chosen": chosen[:k_ref], "pool_p90_loss_db": pool, "bank_p90_loss_db": at_ref,
            "smallest_size_meeting": smallest, "searched_up_to": k}


def placement_sums(P, vis, pid, n_place):
    """Per-placement sum of miss over visible onsets (n_place, n_grid) and visible counts."""
    S = np.zeros((n_place, P.shape[1]))
    C = np.zeros(n_place)
    np.add.at(S, pid[vis], P[vis])
    np.add.at(C, pid[vis], 1.0)
    return S, C


def resample_weights(ss, n_place, n_boot) -> np.ndarray:
    """(n_boot, n_place) multiplicity of each placement in each bootstrap resample."""
    rng = np.random.default_rng(ss)
    idx = rng.integers(0, n_place, (n_boot, n_place))
    return np.stack([np.bincount(r, minlength=n_place) for r in idx]).astype(float)


def boot_crossings(S, C, grid, W, target=0.10) -> np.ndarray:
    """Crossing of the visible-onset mean miss for each weighted resample (nan if none)."""
    mean = (W @ S) / np.maximum(W @ C, 1e-12)[:, None]
    out = np.full(W.shape[0], np.nan)
    for b in range(W.shape[0]):
        c = crossing(mean[b], grid, target)
        if c is not None:
            out[b] = c
    return out


def pct_ci(v, ci_percent=95.0):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [None, None]
    a = (100.0 - ci_percent) / 2.0
    return [float(np.percentile(v, a)), float(np.percentile(v, 100.0 - a))]
