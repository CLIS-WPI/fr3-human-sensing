"""Scene and solver primitives for the v3 production run (review 3, phase 3).

Differences from the v2 primitives in ``localization`` (left unchanged so v2 reproduces):
- the scene descriptor is passed explicitly (cfg["scene"]);
- objects with ``"shape": "plane"`` are single rectangles (zero geometric thickness);
- per-material slab thickness from ``material_thickness_m`` (others: Sionna default);
- target cuboid size is explicit (cfg["rt"]["target_size_m"]), so a target does not shadow
  the other segment the way the default 0.3 m cube does;
- RCSSolver settings (max_depth, refraction) come from cfg["rt"].
"""

from __future__ import annotations

import mitsuba as mi
import numpy as np
from sionna.rt import ITURadioMaterial, PathSolver, Receiver, Scene, SceneObject, Transmitter
from sionna.rt.rcs import ConstantRCSSensingTarget, RCSSolver, TR38901SensingTarget

from fr3isac import geometry as geo
from fr3isac.experiments import localization as loc
from fr3isac.sensing.rt_channel import cfr_passband, sensing_path_arrays


def _p3(v):
    return mi.Point3f(float(v[0]), float(v[1]), float(v[2]))


def _object(o):
    c, s = [float(x) for x in o["center_m"]], [float(x) for x in o["size_m"]]
    if o.get("shape") == "plane":
        if s[0] != 0.0:
            raise ValueError("only x-normal planes are supported")
        # mitsuba rectangle: [-1, 1]^2 in local xy, normal +z; rotate about y so the normal is +x
        tf = (mi.ScalarTransform4f.translate(c) @ mi.ScalarTransform4f.rotate([0, 1, 0], 90)
              @ mi.ScalarTransform4f.scale([s[2] / 2, s[1] / 2, 1.0]))
        return SceneObject(mi_mesh=mi.load_dict({"type": "rectangle", "to_world": tf}), name=o["name"])
    tf = mi.ScalarTransform4f.translate(c) @ mi.ScalarTransform4f.scale([x / 2 for x in s])
    return SceneObject(mi_mesh=mi.load_dict({"type": "cube", "to_world": tf}), name=o["name"])


def build_room(fc_hz, desc, targets=()):
    sc = Scene()
    th = desc.get("material_thickness_m", {})
    mats = {m: ITURadioMaterial(name=m, itu_type=m, thickness=th.get(m)) for m in desc["materials"]}
    for m in mats.values():
        sc.add(m)
    for o in desc["objects"]:
        so = _object(o)
        so.radio_material = mats[o["material"]]
        sc.add(so)
    bs = desc["bs"]["position_m"]
    sc.add(Transmitter("bs", position=_p3(bs)))
    sc.add(Receiver("bs_sense", position=_p3(bs)))
    sc.tx_array = loc._array()
    sc.rx_array = loc._array()
    sc.frequency = fc_hz
    for t in targets:
        sc.add(t)
    return sc


def targets_for(model, s, rtcfg):
    a = float(rtcfg["target_size_m"])
    dims = {"length": a, "width": a, "height": a}
    if model == "R0":
        return [TR38901SensingTarget("human", object_type="human", model_type=2, position=_p3(s.torso_position),
                                     random_sigma_s=False, random_xpr=False, random_phases=False, **dims)]
    if model == "R1-P":
        return [ConstantRCSSensingTarget("point", sigma=1.0, position=_p3(s.torso_position), **dims)]
    return [ConstantRCSSensingTarget("torso", sigma=rtcfg["torso_rcs_m2"], position=_p3(s.torso_position), **dims),
            ConstantRCSSensingTarget("right_forearm", sigma=rtcfg["forearm_rcs_m2"],
                                     position=_p3(s.forearm_position), **dims)]


def rcs_paths(fc_hz, desc, model, s, rtcfg):
    return RCSSolver(deterministic=True)(build_room(fc_hz, desc, targets_for(model, s, rtcfg)),
                                         max_depth=int(rtcfg["max_depth"]), los=True, specular_reflection=False,
                                         refraction=bool(rtcfg["refraction"]), seed=1)


def solve_cfr(fc_hz, desc, model, s, offsets_hz, gain_db, rtcfg):
    arr = sensing_path_arrays(rcs_paths(fc_hz, desc, model, s, rtcfg))
    return cfr_passband(arr["a"], arr["tau"], fc_hz, offsets_hz) * (10.0 ** (gain_db / 20.0))


def bg_cfr(fc_hz, desc, offsets_hz):
    """Static background (no target): LoS + single specular reflections, as in v2."""
    paths = PathSolver(deterministic=True)(build_room(fc_hz, desc), max_depth=1, los=True,
                                           specular_reflection=True, diffuse_reflection=False,
                                           refraction=False, diffraction=False, seed=1)
    arr = sensing_path_arrays(paths)
    return cfr_passband(arr["a"], arr["tau"], fc_hz, offsets_hz)


def blockers_on_leg(bs, point, other_center, blk, half):
    """Names of what blocks the BS<->point leg under the RCSSolver rule: the leg is tested only
    after it leaves the point's own target box; the other target's box absorbs."""
    u = bs - point
    u = u / np.linalg.norm(u)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.min(np.where(u > 0, half / u, np.where(u < 0, -half / u, np.inf)))
    start = point + u * t
    hit = [o["name"] for o in blk if geo.segment_aabb_intersect(bs, start, o["center_m"], o["size_m"])]
    if geo.segment_aabb_intersect(bs, start, other_center, [2 * half] * 3):
        hit.append("other_target")
    return hit
