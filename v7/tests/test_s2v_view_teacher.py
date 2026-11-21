"""v7 S2' tests (21 Nov 2026): the view-keeping teacher (v7/experts/teacher_view_v7.py).

  1. settings: top edge 28 deg below the horizon at 60 deg (13 at 45), margin 4 deg,
     cap 6 m, far from 2.5 m, nose hand-over 4 to 6 m
  2. the height rule: 0 near the pad, fades in over 1 m, then tan(32 deg) x d, capped
  3. NEAR THE PAD it is EXACTLY TeacherV7: a shadow TeacherV7, fed the same states, gives the
     same move and drop at every step with d <= 2.5 m (the drop rule is untouched); every
     phase name is one the demo recorder knows
  4. far: it never asks for LESS climb than TeacherV7; it climbs when below the needed height;
     beyond 6 m it turns the nose toward the pad (yaw command has the sign of the bearing error)
  5. whole flights (fresh sandbox seeds 35000+, safe there): safe, height <= cap + 0.6 m, and
     the pad centre is in the 60 deg picture on more far steps than with TeacherV7
Headless (geometry only, no images). One EGL env for the module.
Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -s v7/tests/test_s2v_view_teacher.py
"""
import numpy as np
import pytest

from v6_drop.experts.teacher import wrap
from v7.envs.gimbal_camera import GimbalCamera
from v7.envs.landing_env_v7 import LandingEnvV7
from v7.envs.platform_v7 import PAD_TOP_Z
from v7.envs.s500 import lowest_z
from v7.experts import teacher_view_v7 as TV
from v7.experts.teacher_v7 import TeacherV7

PATHS = ("straight", "wave", "corners", "random")


@pytest.fixture(scope="module")
def env():
    e = LandingEnvV7(renderer="egl", seed=5, headless=True, pitch_deg=60.0)
    yield e
    e.close()


def fly(env, seed, mode, shadow=True):
    """Fly the view teacher; feed a shadow TeacherV7 the same states. Returns (info, rows)."""
    env.reset(seed=seed, options=dict(mode=mode))
    tv, t7 = TV.TeacherV7View(), TeacherV7()
    tv.reset()
    t7.reset()
    rows = []
    while True:
        s = env.world.true_state()
        rel = s["pad_center"] - s["pos"]
        d = float(np.hypot(rel[0], rel[1]))
        hf = float(lowest_z(s["R"], s["pos"]) - PAD_TOP_Z)
        yaw = float(np.arctan2(s["R"][1, 0], s["R"][0, 0]))
        a7 = t7.act(env)[0] if shadow else None
        a, ti = tv.act(env)
        rows.append(dict(d=d, hf=hf, yaw=yaw, bearing=float(np.arctan2(rel[1], rel[0])),
                         need=ti["h_need"], phase=ti["phase"], a=a, a7=a7, s=s))
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            return info, rows


# ---------------------------------------------------------------- 1. settings
def test_settings():
    assert TV.top_edge_deg(60.0) == pytest.approx(28.0, abs=0.1)
    assert TV.top_edge_deg(45.0) == pytest.approx(13.0, abs=0.1)
    t = TV.TeacherV7View()
    assert t.k_view == pytest.approx(np.tan(np.radians(32.0)), abs=1e-3)
    assert (TV.MARGIN_DEG, TV.H_VIEW_MAX, TV.D_VIEW, TV.D_RAMP) == (4.0, 6.0, 2.5, 1.0)
    assert (t.d_yaw0, t.d_yaw1) == (4.0, 6.0)
    assert t.gate_name == "safe30m" and t.a_tau == 0.2      # TeacherV7 settings kept


# ---------------------------------------------------------------- 2. height rule
def test_height_rule():
    t = TV.TeacherV7View()
    assert t.h_needed(0.0) == 0.0 and t.h_needed(2.5) == 0.0
    assert t.h_needed(3.0) == pytest.approx(0.5 * t.k_view * 3.0)
    assert t.h_needed(5.0) == pytest.approx(t.k_view * 5.0)
    assert t.h_needed(30.0) == 6.0
    assert TV.TeacherV7View(pitch_deg=45.0).h_needed(10.0) == pytest.approx(
        np.tan(np.radians(TV.top_edge_deg(45.0) + 4.0)) * 10.0)


# ---------------------------------------------------------------- 3, 4. per-step rules
@pytest.mark.parametrize("mode,seed", [("corners", 35000), ("wave", 35001), ("random", 35002)])
def test_near_identical_far_rules(env, mode, seed):
    info, rows = fly(env, seed, mode)
    near = [r for r in rows if r["d"] <= TV.D_VIEW]
    assert len(near) > 20
    for r in near:
        np.testing.assert_array_equal(r["a"]["move"], r["a7"]["move"])
        assert r["a"]["drop"] == r["a7"]["drop"]
    from v7.scripts.record_demos_v7 import PHASES            # demos store the phase index
    assert all(r["phase"] in PHASES for r in rows)
    far = [r for r in rows if r["d"] > TV.D_VIEW]
    for r in far:                                            # never less climb than V7
        assert r["a"]["move"][2] >= r["a7"]["move"][2] - 1e-6
    low = [r for r in far if r["need"] > r["hf"] + 0.3]
    for r in low:                                            # below the needed height: climb
        assert r["a"]["move"][2] > 0.2
    turn = [r for r in rows if r["d"] > 6.0 and abs(wrap(r["bearing"] - r["yaw"])) > np.radians(5)]
    for r in turn:                                           # nose toward the pad
        assert np.sign(r["a"]["move"][3]) == np.sign(wrap(r["bearing"] - r["yaw"]))


# ---------------------------------------------------------------- 5. whole flights
def far_view(env, rows):
    cam = GimbalCamera(env.world.cid, pitch_deg=60.0, renderer="egl")
    n = k = 0
    for r in rows:
        if r["d"] > 3.0:
            uv, z = cam.project(r["s"]["pad_center"], r["s"]["R"], r["s"]["pos"])
            n += 1
            k += bool(z[0] > 0 and 0 <= uv[0][0] <= cam.w and 0 <= uv[0][1] <= cam.h)
    return k, n


@pytest.mark.parametrize("mode", PATHS)
def test_flights_safe_and_more_view(env, mode):
    kv = nv = k7 = n7 = 0
    for seed in (35010, 35011, 35012):
        info, rows = fly(env, seed, mode, shadow=False)
        assert info["outcome"] == "success", (mode, seed, info["outcome"])
        assert max(r["hf"] for r in rows) <= TV.H_VIEW_MAX + 0.6
        k, n = far_view(env, rows)
        kv, nv = kv + k, nv + n
        # the same flight with TeacherV7
        env.reset(seed=seed, options=dict(mode=mode))
        t7 = TeacherV7()
        t7.reset()
        rows7 = []
        while True:
            s = env.world.true_state()
            rel = s["pad_center"] - s["pos"]
            rows7.append(dict(d=float(np.hypot(rel[0], rel[1])), s=s))
            _, _, term, trunc, _ = env.step(t7.act(env)[0])
            if term or trunc:
                break
        k, n = far_view(env, rows7)
        k7, n7 = k7 + k, n7 + n
    print(f"\n{mode}: far steps with the pad in view: view teacher {kv}/{nv}, TeacherV7 {k7}/{n7}")
    if nv and n7:
        assert kv / nv >= k7 / n7 - 0.02
