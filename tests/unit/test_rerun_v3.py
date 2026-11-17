import numpy as np

from fr3isac import geometry as geo
from fr3isac.experiments import rerun_v2_core as core
from fr3isac.experiments import rerun_v3 as r


def test_offset_grid_placements_disjoint_from_v2():
    cfg = core.load_cfg("configs/rerun_v3.yaml")
    elig, test = r.test_placements(cfg, geo.load_descriptor(cfg["scene"]))
    assert len(test) == cfg["n_test_placements"] <= len(elig)
    v2 = core.load_cfg("configs/rerun_v2.yaml")
    old = core.eligible_grid(geo.load_descriptor(v2["scene"]), v2["grid_step_m"])
    assert not set(test) & set(old)
    assert all(abs((x - geo.ROI_X[0]) % 0.5 - 0.25) < 1e-9 for x, _ in test)


def test_v3_streams_disjoint_from_v2():
    a = r.streams(1)["test_motions"].generate_state(4)
    b = core.seed_sequences(1)["test_motions"].generate_state(4)
    assert not np.array_equal(a, b)


def test_rt_small_falls_back_to_summary(tmp_path):
    np.savez_compressed(tmp_path / "rt_cache.npz", test_label=np.arange(3), big=np.zeros(20_000))
    names = r.write_rt_summary(tmp_path)
    assert "test_label" in names and "big" not in names
    (tmp_path / "rt_cache.npz").unlink()
    z = r.rt_small(tmp_path)
    assert np.array_equal(z["test_label"], np.arange(3)) and "big" not in z.files


def test_manifest_release_copy_without_git(tmp_path, monkeypatch):
    from fr3isac.experiments import rerun_v2 as rv

    rel = tmp_path / "RELEASE_VERSION"
    rel.write_text("version: v1.0-commag\nsource_commit: abc1234\n")
    monkeypatch.setattr(rv, "_git", lambda *a: None)          # no git checkout
    monkeypatch.setattr(rv, "RELEASE_FILE", rel)
    monkeypatch.setenv("FR3ISAC_CONTAINER_IMAGE", "img:tag")
    monkeypatch.setenv("FR3ISAC_CONTAINER_DIGEST", "sha256:feed")
    m = rv.manifest(core.load_cfg("configs/rerun_v3.yaml"), "numbers")
    assert m["git_sha"] == "v1.0-commag (source abc1234)"
    assert m["git_dirty_src_configs_tools"] is False and m["provenance_note"] == "release copy"
    assert (m["container_image"], m["container_digest"]) == ("img:tag", "sha256:feed")
    monkeypatch.setattr(rv, "RELEASE_FILE", tmp_path / "absent")  # neither git nor release file
    monkeypatch.delenv("FR3ISAC_GIT_SHA", raising=False)
    m = rv.manifest(core.load_cfg("configs/rerun_v3.yaml"), "numbers")
    assert m["git_sha"] == "unknown" and m["git_dirty_src_configs_tools"] is None and "provenance_note" not in m
