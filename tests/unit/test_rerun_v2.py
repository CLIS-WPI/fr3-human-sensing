"""Rerun v2 core: data split, seeds, motions, windowing, deadline detection, FA events."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from fr3isac import geometry as geo
from fr3isac.experiments import rerun_v2_core as core


@pytest.fixture(scope="module")
def cfg():
    return core.load_cfg()


@pytest.fixture(scope="module")
def split(cfg):
    elig = core.eligible_grid(geo.load_descriptor(cfg["scene"]), cfg["grid_step_m"])
    dev, test = core.split_placements(elig, cfg["n_test_placements"])
    return elig, dev, test


def test_dev_test_placements_disjoint_and_complete(split):
    elig, dev, test = split
    assert len(elig) == 90 and len(test) == 50 and len(dev) == 40
    assert not set(dev) & set(test)
    assert sorted(dev + test) == sorted(elig)
    core.assert_disjoint(dev, test)


def test_assert_disjoint_fails_on_overlap(split):
    _, dev, test = split
    with pytest.raises(AssertionError):
        core.assert_disjoint(dev + [test[0]], test)


def test_seed_streams_and_motions_disjoint(cfg, split):
    _, dev, test = split
    ss = core.seed_sequences(cfg["master_seed"])
    keys = [s.spawn_key for s in ss.values()]
    assert len(set(keys)) == len(keys)
    md = core.motions_per_placement(ss["dev_motions"], dev, cfg["k_dev_motions_per_placement"], cfg)
    mt = core.motions_per_placement(ss["test_motions"], test, cfg["k_test_motions_per_placement"], cfg)
    assert mt.shape == (50, cfg["k_test_motions_per_placement"], 4)
    rows_d = {tuple(r) for r in md.reshape(-1, 4)}
    rows_t = {tuple(r) for r in mt.reshape(-1, 4)}
    assert not rows_d & rows_t
    mc, st = cfg["motion_class"], cfg["slow_time"]
    for m in (md, mt):
        a, f, p, t0 = m.reshape(-1, 4).T
        assert a.min() >= mc["amplitude_m"][0] and a.max() <= mc["amplitude_m"][1]
        assert f.min() >= mc["rate_hz"][0] and f.max() <= mc["rate_hz"][1]
        assert p.min() >= 0 and p.max() <= 2 * np.pi
        assert t0.min() >= st["onset_window_s"][0] and t0.max() <= st["onset_window_s"][1]
    # reproducible
    mt2 = core.motions_per_placement(core.seed_sequences(cfg["master_seed"])["test_motions"], test,
                                     cfg["k_test_motions_per_placement"], cfg)
    np.testing.assert_array_equal(mt, mt2)


def test_onset_trajectory_rest_then_continuous():
    root = (4.0, 3.0, geo.TARGET_HEIGHT_M)
    rest = np.asarray(geo.static_sample(root).forearm_position)
    t = np.array([0.05, 0.1199, 0.12, 0.1201, 0.2])
    s = geo.onset_arm_trajectory(root, t, 0.3, 1.0, 1.1, 0.12)
    np.testing.assert_allclose(s[0].forearm_position, rest)
    np.testing.assert_allclose(s[2].forearm_position, rest)
    assert abs(s[3].forearm_position[0] - rest[0]) < 1e-3          # continuous at onset
    assert abs(s[4].forearm_position[0] - rest[0]) > 1e-2


def test_deadline_windows(cfg):
    ends = core.window_end_times(cfg)
    assert np.allclose(np.diff(ends), cfg["slow_time"]["hop_samples"] / cfg["slow_time"]["prf_hz"])
    for t0 in np.linspace(*cfg["slow_time"]["onset_window_s"], 101)[:-1]:
        m = core.deadline_mask(t0, cfg)
        assert m.sum() == 4
        assert np.all(ends[m] > t0) and np.all(ends[m] <= t0 + cfg["slow_time"]["deadline_s"])


def _small(cfg, **det):
    c = copy.deepcopy(cfg)
    c["detector"].update(det)
    return c


def test_miss_curve_causal_and_shared_noise(cfg):
    st = cfg["slow_time"]
    bank = core.bank_from_params(12e9, core.baseline_params(cfg), cfg)
    s2, t0 = 1.0, 0.12
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((st["record_samples"], 400)) + 1j * rng.standard_normal((st["record_samples"], 400))) * np.sqrt(0.5)
    thr = float(np.quantile(core.null_stats(bank, s2, 50_000, np.random.SeedSequence(1)), 1 - 1e-2))
    # strong signal only in samples after the last in-deadline window has ended
    # (windows overlap, so it must lie outside every in-deadline window)
    m = core.deadline_mask(t0, cfg)
    last_end = core.window_starts(cfg)[m][-1] + st["window_samples"]
    assert last_end < st["record_samples"]
    sig = np.zeros(st["record_samples"], complex)
    sig[last_end:] = bank[5][: st["record_samples"] - last_end]
    miss_late = core.miss_curve(sig, t0, bank, s2, thr, [100.0], X, cfg)[0]
    assert miss_late > 0.9                      # a detection after the deadline is not a success
    # same signal inside an in-deadline window -> detected
    sig2 = np.zeros_like(sig)
    s1 = core.window_starts(cfg)[core.deadline_mask(t0, cfg)][0]
    sig2[s1:s1 + st["window_samples"]] = bank[5]
    assert core.miss_curve(sig2, t0, bank, s2, thr, [100.0], X, cfg)[0] == 0.0
    # no signal: miss = P(no crossing in 4 overlapping windows), close to 1 - 4*pfa
    m0 = core.miss_curve(np.zeros_like(sig), t0, bank, s2, thr, [1.0], X, cfg)[0]
    assert 0.9 < m0 <= 1.0


def test_event_false_alarms_grouping(cfg):
    c = _small(cfg, fa_stream_hours=0.01)
    bank = core.bank_from_params(3.5e9, core.baseline_params(cfg), cfg)
    never = core.event_false_alarms(bank, 1.0, np.inf, c, np.random.SeedSequence(0))
    assert never["window_crossings"] == 0 and never["events"] == 0
    always = core.event_false_alarms(bank, 1.0, -1.0, c, np.random.SeedSequence(0), chunk_windows=97)
    assert always["window_crossings"] == always["windows"] and always["events"] == 1


def test_calibration_hits_target(cfg):
    c = _small(cfg, pfa_per_window=1e-2, n_calibration_windows=200_000, n_validation_windows=200_000)
    bank = core.bank_from_params(15e9, core.baseline_params(cfg), cfg)
    r = core.calibrate(bank, 2.5, c, np.random.SeedSequence(10), np.random.SeedSequence(11))
    assert r["accept"] and abs(r["pfa"] - 1e-2) < 2e-3


def test_greedy_facility_picks_cover():
    cap = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.6, 0.6, 0.6]], dtype=np.float32)
    assert core.greedy_facility(cap, 1) == [1]          # gain 2.0 > 1.8 > 1.0
    assert core.greedy_facility(cap, 2) == [1, 0]       # then full cover of item 0


def test_capture_of_exact_template_is_lossless(cfg):
    st = cfg["slow_time"]
    params = np.array([[0.2, 1.0, 0.3]])
    b = core.bank_from_params(28e9, params, cfg)
    sig = np.zeros(st["record_samples"], complex)
    s1 = core.window_starts(cfg)[2]
    sig[s1:s1 + st["window_samples"]] = 3.0 * b[0]
    loss = core.capture_loss_db(b, [sig], [0.12], cfg)
    assert abs(loss[0]) < 1e-9


def test_config_numbers_are_numeric(cfg):
    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif isinstance(x, str):
            try:
                float(x)
            except ValueError:
                return
            raise AssertionError(f"numeric value parsed as string at {path}: {x!r}")
    walk(cfg)
    assert all(isinstance(f, float) for f in cfg["carriers_hz"] + cfg["dry_run"]["carriers_hz"])


def test_coverage_search_smallest_size():
    # 3 items; candidate 0 covers item 0, 1 covers item 1, 2 covers item 2, 3 is mediocre everywhere
    cap = np.array([[1, .01, .01], [.01, 1, .01], [.01, .01, 1], [.5, .5, .5]], dtype=np.float32)
    r = core.coverage_search([cap[:, :2], cap[:, 2:]], tol_db=0.5, k_ref=1, k_max=4)
    assert r["pool_p90_loss_db"] == [pytest.approx(0.0, abs=1e-6)] * 2
    assert r["chosen"] == [3]                                   # best single template
    assert r["bank_p90_loss_db"][0] == pytest.approx(3.0103, abs=1e-3)
    assert r["smallest_size_meeting"] == 4                      # needs all three specialists


def test_bootstrap_is_paired_and_reproducible():
    W1 = core.resample_weights(np.random.SeedSequence(5), 7, 50)
    W2 = core.resample_weights(np.random.SeedSequence(5), 7, 50)
    np.testing.assert_array_equal(W1, W2)
    assert np.all(W1.sum(1) == 7)
    grid = np.arange(0.0, 10.0, 1.0)
    # placement-level miss curves: crossing of each placement differs; two banks offset by 2 dB
    rng = np.random.default_rng(0)
    off = rng.uniform(2, 6, 7)
    P = np.clip(1 - (grid[None, :] - off[:, None]) / 2, 0, 1)
    pid, vis = np.arange(7), np.ones(7, bool)
    S, C = core.placement_sums(P, vis, pid, 7)
    S2, _ = core.placement_sums(np.clip(1 - (grid[None, :] - off[:, None] - 2) / 2, 0, 1), vis, pid, 7)
    d = core.boot_crossings(S2, C, grid, W1) - core.boot_crossings(S, C, grid, W1)
    np.testing.assert_allclose(d[np.isfinite(d)], 2.0, atol=1e-9)   # paired: exact shift survives
    assert core.pct_ci(d) == [pytest.approx(2.0), pytest.approx(2.0)]


def test_placement_sums_ignore_hidden():
    P = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    S, C = core.placement_sums(P, np.array([True, True, False]), np.array([0, 0, 1]), 2)
    np.testing.assert_array_equal(S, [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(C, [2.0, 0.0])


def test_constant_term_share(cfg):
    st = cfg["slow_time"]
    n = st["record_samples"]
    const = np.ones(n, complex)
    assert core.constant_term_share([const], [0.12], cfg)[0] == pytest.approx(1.0)
    t = np.arange(n) / st["prf_hz"]
    chirp = np.exp(1j * 2 * np.pi * 300.0 * t)          # zero-mean over a 100 ms window
    assert core.constant_term_share([chirp], [0.12], cfg)[0] < 1e-3
    assert core.constant_term_share([chirp - 1.0], [0.12], cfg)[0] == pytest.approx(0.5, abs=1e-3)


def test_reference_averaging_raises_pfa(cfg):
    c = _small(cfg, pfa_per_window=1e-2, n_calibration_windows=100_000, n_validation_windows=100_000)
    bank = core.bank_from_params(12e9, core.baseline_params(cfg), cfg)
    r = core.calibrate(bank, 1.0, c, np.random.SeedSequence(1), np.random.SeedSequence(2))
    assert r["reference_averaging"]["16"]["pfa"] > r["reference_averaging"]["64"]["pfa"] >= r["pfa"]
