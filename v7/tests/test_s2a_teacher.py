"""v7 S2a tests (20 Nov 2026): the V7 teacher and the referee fix.

  1. the decided settings (D1 to D4): speed-change learning 0.2 s, lead 0.3 s, drop rule
     safe30m, wait below along-road 1 m/s^2, wait height 0.25 m
  2. no V6 standing height (P.REST_HEIGHT 0.152) in the teacher; its height is the S500's
     true lowest point (level drone: = skid height; leaning drone: lower)
  3. the learned pad velocity: for a steady speed change it converges to the TRUE velocity
     (+ the lead), and it starts from zero each flight
  4. speed limits come from the env: chase command length <= 10 m/s, direction kept
  5. referee: the pad point velocity under the drone = centre velocity + turn (checked
     against the pad's own motion), and check 5 uses it
  6. whole flights: 5 tuning seeds per path, all safe; the teacher never drops while the pad
     brakes or speeds up hard
Headless (no images) except where noted. One EGL env for the module.
Run:  python -m pytest -q -s v7/tests/test_s2a_teacher.py
"""
import inspect

import numpy as np
import pytest

from v6_drop.envs import lmf2_params as P
from v6_drop.experts import drop_gates as G
from v7.envs.landing_env_v7 import LandingEnvV7
from v7.envs.platform_v7 import PAD_TOP_Z
from v7.envs.s500 import REST_HEIGHT
from v7.experts import teacher_v7 as T7

PATHS = ("straight", "wave", "corners", "random")


@pytest.fixture(scope="module")
def env():
    e = LandingEnvV7(renderer="egl", seed=5, headless=True)
    yield e
    e.close()


def fly(env, teacher, seed, mode):
    env.reset(seed=seed, options=dict(mode=mode))
    teacher.reset()
    log = []
    while True:
        a, ti = teacher.act(env)
        u = np.array([np.cos(env.world.plat.psi), np.sin(env.world.plat.psi)])
        log.append((ti["phase"], int(a["drop"]), float(teacher.pad_a_est[:2] @ u)))
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            return info, log


# ---------------------------------------------------------------- 1. decided settings
def test_decided_settings():
    t = T7.TeacherV7()
    assert t.a_tau == 0.2 and t.t_lead == 0.3 and t.gate_name == "safe30m"
    assert t.a_calm == 1.0 and t.low_point
    assert T7.H_WAIT == 0.25 and (T7.H_ABORT, T7.V_ABORT) == (0.12, 0.35)
    assert T7.PAD_V_TAU == 0.5                                # velocity learning as V6
    g = G.GATES["safe30m"]
    assert g["tilt"] == pytest.approx(np.radians(8.0)) and g["vxy"] == 0.30 and g["impact"] == 0.8


# ---------------------------------------------------------------- 2. height
def test_no_v6_standing_height():
    src = inspect.getsource(T7)
    assert "P.REST_HEIGHT" not in src
    assert T7.feet_height(PAD_TOP_Z + REST_HEIGHT + 0.05) == pytest.approx(0.05)


def test_height_is_true_lowest_point(env):
    t = T7.TeacherV7()
    for lean_deg in (0.0, 10.0):
        spawn = dict(pos=[0.0, 0.0, PAD_TOP_Z + REST_HEIGHT + 0.10], yaw=0.0)
        env.reset(seed=31000, options=dict(mode="straight", pad_speed=0.0, spawn=spawn))
        if lean_deg:
            q = env.world.quad
            import pybullet as p
            pos, _ = p.getBasePositionAndOrientation(q.body, physicsClientId=env.world.cid)
            p.resetBasePositionAndOrientation(q.body, pos, p.getQuaternionFromEuler(
                [0.0, np.radians(lean_deg), 0.0]), physicsClientId=env.world.cid)
        t.reset()
        _, ti = t.act(env)
        print(f"\n[height] lean {lean_deg:.0f} deg: teacher height {ti['hf'] * 100:.2f} cm "
              f"(level-drone formula 10.00 cm)")
        if lean_deg == 0.0:
            assert ti["hf"] == pytest.approx(0.10, abs=1e-3)
        else:
            assert 0.065 < ti["hf"] < 0.09            # one skid end ~2 cm lower
    env.world.done = True


# ---------------------------------------------------------------- 3. learned pad velocity
def test_learned_velocity_steady_speed_change():
    t = T7.TeacherV7(t_lead=0.0)
    t.reset()
    assert np.all(t.pad_v_est == 0) and np.all(t.pad_a_est == 0)
    a = np.array([-2.0, 0.0, 0.0])                       # braking 2 m/s^2
    v = np.array([9.0, 0.0, 0.0])
    for k in range(60):                                  # 6 s
        pred = t.pad_velocity_pred(dict(pad_v=v.copy(), pad_a=a))
        v = v + a * P.POLICY_DT
    err = np.linalg.norm(pred - (v - a * P.POLICY_DT))
    print(f"\n[learn] braking 2 m/s^2: error after 6 s {err:.4f} m/s (V6 0.5 s filter alone: "
          f"{0.5 * 2:.1f} m/s)")
    assert err < 0.12
    t2 = T7.TeacherV7()                                  # with the 0.3 s lead
    t2.pad_v_est, t2.pad_a_est = v.copy(), a.copy()
    lead = t2.pad_velocity_pred(dict(pad_v=v, pad_a=a)) - t2.pad_v_est
    assert lead[0] == pytest.approx((0.5 + 0.3) * a[0], rel=0.02)


# ---------------------------------------------------------------- 4. limits from the env
def test_chase_command_capped_by_length(env):
    t = T7.TeacherV7()
    spawn = dict(pos=[-8.0, -8.0, PAD_TOP_Z + REST_HEIGHT + 3.0], yaw=0.0)
    env.reset(seed=31001, options=dict(mode="straight", pad_speed=9.0, spawn=spawn))
    t.reset()
    for _ in range(10):                                  # learn the pad speed
        a, _ = t.act(env)
    m = a["move"][:2] * env.act_scale[:2]
    print(f"\n[cap] chase command {np.linalg.norm(m):.2f} m/s")
    assert np.linalg.norm(m) <= 10.0 + 1e-4 and np.linalg.norm(m) > 9.5
    env.world.done = True


# ---------------------------------------------------------------- 5. referee on a turning pad
def test_pad_point_velocity_matches_pad_motion(env):
    env.reset(seed=31002, options=dict(mode="corners", pad_speed=4.0))
    w = env.world
    for _ in range(3000):                                # up to 30 s of pad driving
        if abs(w.plat.w) > 0.3:
            break
        w.plat.integrate()
    assert abs(w.plat.w) > 0.3, "no turn found"
    pt_pad = np.array([[0.2, 0.15]])                     # a point 0.25 m off the centre
    p0 = w.plat.pad_to_world(pt_pad)[0]
    v_formula = w.pad_point_velocity(p0)
    w.plat.integrate()
    p1 = w.plat.pad_to_world(pt_pad)[0]
    v_moved = (p1 - p0) / P.PHYSICS_DT
    print(f"\n[referee] turn {w.plat.w:+.2f} rad/s: point velocity {np.round(v_formula[:2], 3)}, "
          f"from the pad's motion {np.round(v_moved[:2], 3)}, centre {np.round(w.plat.velocity()[:2], 3)}")
    assert np.allclose(v_formula[:2], v_moved[:2], atol=0.02)
    assert np.linalg.norm(v_formula[:2] - w.plat.velocity()[:2]) > 0.07   # the old check's error
    w.done = True


def test_referee_reports_point_velocity(env):
    t = T7.TeacherV7()
    info, _ = fly(env, t, 31000, "corners")
    td = info["touchdown"]
    assert info["outcome"] == "success" and "v_rel_end_centre" in td
    assert td["v_rel_end"] <= 0.1


# ---------------------------------------------------------------- 6. whole flights
@pytest.mark.parametrize("mode", PATHS)
def test_flights_each_path(env, mode):
    t = T7.TeacherV7()
    res = []
    for seed in range(31000, 31005):
        info, log = fly(env, t, seed, mode)
        res.append((info["outcome"], info["t"], sum(ph == "wait" for ph, _, _ in log)))
        for ph, drop, a_along in log:                    # D4: never drop while not settled
            if drop:
                assert abs(a_along) < 1.0
    print(f"\n[teacher {mode}] (outcome, decisions, wait steps): {res}")
    assert all(r[0] == "success" for r in res)
