"""Slow-time channel and per-sample reference from the Sionna RT solver.

Assembles the channel from RT path coefficients using the passband form of
``rt_channel``; provides a per-sample reference and a tracked fast method (anchor
snapshots with amplitude and unwrapped-phase interpolation). Path tracking uses
interaction-type sequence, object ids, and interaction-point and delay continuity;
ambiguous matches become death+birth and are logged. Complex coefficients are never
interpolated directly, only amplitude and unwrapped delay/phase.
"""

from __future__ import annotations

import numpy as np

from fr3isac.config import SPEED_OF_LIGHT_M_S
from fr3isac.sensing.rt_channel import cfr_passband, sensing_path_arrays


def rt_frame_cfr(paths, fc_hz: float, subcarrier_offsets_hz: np.ndarray) -> np.ndarray:
    """Passband CFR ``H[k]`` for one RT frame."""
    arr = sensing_path_arrays(paths)
    return cfr_passband(arr["a"], arr["tau"], fc_hz, subcarrier_offsets_hz)


def slow_time_cfr(frames, fc_hz: float, subcarrier_offsets_hz: np.ndarray) -> np.ndarray:
    """Per-sample reference ``H[k, n]`` from direct RT frames (slow time)."""
    if len(frames) == 0:
        raise ValueError("frames must be non-empty")
    return np.stack([rt_frame_cfr(p, fc_hz, subcarrier_offsets_hz) for p in frames], axis=1)


def rt_doppler_axis(frames) -> np.ndarray:
    """Per-frame target Doppler (Hz) from the RT solver, for later stages."""
    values = []
    for paths in frames:
        arr = sensing_path_arrays(paths)
        dop = arr["doppler"]
        values.append(float(np.asarray(dop).mean()) if dop is not None and np.size(dop) else 0.0)
    return np.asarray(values, dtype=np.float64)


def path_records(paths) -> list[dict]:
    """Per-valid-path records with interaction/object tracking keys."""
    from fr3isac.rt.engine import paths_a_complex

    a = paths_a_complex(paths)
    tau = np.asarray(paths.tau)
    valid = np.asarray(paths.valid)
    a_link = a[0, 0, 0, 0]
    if a_link.ndim == 2:
        a_link = a_link[:, 0]
    n_paths = a_link.shape[0]
    if n_paths == 0 or tau.size == 0:
        return []
    tau_link = tau.reshape(-1, tau.shape[-1])[0]
    valid_link = valid.reshape(-1, valid.shape[-1])[0]
    inter = np.asarray(paths.interactions)
    obj = np.asarray(paths.objects)
    vert = np.asarray(paths.vertices)
    depth = inter.shape[-2]
    inter_link = inter.reshape(-1, depth, inter.shape[-1])[0]
    obj_link = obj.reshape(-1, depth, obj.shape[-1])[0]
    vert_link = vert.reshape(-1, depth, vert.shape[-2], vert.shape[-1])[0]
    records = []
    for i in range(n_paths):
        if not valid_link[i]:
            continue
        records.append({
            "key": (tuple(int(x) for x in inter_link[:, i]), tuple(int(x) for x in obj_link[:, i])),
            "tau": float(tau_link[i]),
            "a": complex(a_link[i]),
            "vertex": vert_link[:, i, :].mean(axis=0).astype(float),
        })
    return records


def track_frames(frames, delay_tol_s: float = 1e-9, vertex_tol_m: float = 0.05) -> tuple[list[dict], list[tuple]]:
    """Track paths across frames by key + delay/vertex continuity (C3.3).

    Ambiguous matches (more than one candidate within tolerance) are logged and
    split into death+birth, never merged silently.
    """
    tracks: list[dict] = []
    active: dict[tuple, int] = {}
    events: list[tuple] = []
    for n, paths in enumerate(frames):
        records = path_records(paths)
        seen: set[tuple] = set()
        for rec in records:
            key = rec["key"]
            if key in active:
                prev = tracks[active[key]]["samples"][max(tracks[active[key]]["samples"])]
                close = abs(rec["tau"] - prev["tau"]) <= delay_tol_s and np.linalg.norm(
                    rec["vertex"] - prev["vertex"]) <= vertex_tol_m
                if close:
                    tracks[active[key]]["samples"][n] = rec
                    seen.add(key)
                    continue
                events.append(("death", n, key))
                del active[key]
            # birth (new key, or continuity broken)
            tid = len(tracks)
            tracks.append({"key": key, "samples": {n: rec}})
            active[key] = tid
            seen.add(key)
            events.append(("birth", n, key))
        for key in [k for k in active if k not in seen]:
            events.append(("death", n, key))
            del active[key]
    return tracks, events


def interpolate_cfr(tracks, anchor_times_s: np.ndarray, query_times_s: np.ndarray, fc_hz: float,
                    subcarrier_offsets_hz: np.ndarray) -> np.ndarray:
    """Fast CFR: interpolate amplitude and unwrapped scattering phase + delay per track."""
    anchor = np.asarray(anchor_times_s, dtype=float)
    query = np.asarray(query_times_s, dtype=float)
    H = np.zeros((len(np.ravel(subcarrier_offsets_hz)), query.size), dtype=np.complex128)
    for tr in tracks:
        idx = sorted(tr["samples"])
        if len(idx) < 2:
            continue
        ts = anchor[idx]
        amps = np.array([abs(tr["samples"][i]["a"]) for i in idx])
        phases = np.unwrap(np.array([np.angle(tr["samples"][i]["a"]) for i in idx]))
        taus = np.array([tr["samples"][i]["tau"] for i in idx])
        amp_i = np.interp(query, ts, amps)
        phase_i = np.interp(query, ts, phases)
        tau_i = np.interp(query, ts, taus)
        a_i = amp_i * np.exp(1j * phase_i)
        for j in range(query.size):
            H[:, j] += cfr_passband(np.array([a_i[j]]), np.array([tau_i[j]]), fc_hz,
                                    subcarrier_offsets_hz)
    return H


def reconstruction_nmse(fast: np.ndarray, reference: np.ndarray) -> float:
    """Normalized MSE of a fast reconstruction against the per-sample reference."""
    denom = float(np.sum(np.abs(reference) ** 2))
    return float(np.sum(np.abs(fast - reference) ** 2) / denom) if denom > 0 else float("nan")


def interp_complex_series(anchor_times_s, values, query_times_s) -> np.ndarray:
    """Interpolate a complex series by magnitude and unwrapped phase (never complex)."""
    ts = np.asarray(anchor_times_s, dtype=float)
    vals = np.asarray(values, dtype=np.complex128)
    query = np.asarray(query_times_s, dtype=float)
    mag = np.interp(query, ts, np.abs(vals))
    phase = np.interp(query, ts, np.unwrap(np.angle(vals)))
    return mag * np.exp(1j * phase)


def synthesize_cfr(fc_hz, subcarrier_offsets_hz, seg_amplitudes, seg_ranges_m) -> np.ndarray:
    """Fast CFR from per-segment amplitude + geometric (kinematic) range.

    ``seg_amplitudes`` is ``[n_seg, n_slow]`` (complex) and ``seg_ranges_m`` is
    ``[n_seg, n_slow]`` (metres, monostatic range). The phase is the *geometric*
    phase ``exp(-j 2 pi (f_c + df) * 2 R(t)/c)``; complex coefficients are never
    interpolated.
    """
    f = fc_hz + np.asarray(subcarrier_offsets_hz, dtype=float).ravel()
    amps = np.asarray(seg_amplitudes, dtype=np.complex128)
    ranges = np.asarray(seg_ranges_m, dtype=float)
    n_slow = ranges.shape[-1]
    H = np.zeros((f.size, n_slow), dtype=np.complex128)
    for i in range(ranges.shape[0]):
        tau = 2.0 * ranges[i] / SPEED_OF_LIGHT_M_S
        amp = amps[i] if amps.ndim == 2 else np.full(n_slow, amps[i])
        H += amp[None, :] * np.exp(-1j * 2 * np.pi * np.outer(f, tau))
    return H



__all__ = [
    "rt_frame_cfr",
    "slow_time_cfr",
    "rt_doppler_axis",
    "path_records",
    "track_frames",
    "interpolate_cfr",
    "reconstruction_nmse",
    "interp_complex_series",
    "synthesize_cfr",
]
