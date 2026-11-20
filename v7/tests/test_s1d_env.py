"""v7 S1d tests (20 Nov 2026): the V7 world and environment.

  1. gymnasium's own environment checker passes
  2. spaces; the new limits: vx, vy +-10 m/s, speed-change 0.3 (option 0.5) m/s per step,
     flight limit 600 decisions, lost at 40 m, critic pad-speed scale 9
  3. EVERY standing-height user uses the S500's 0.22 m (lowest point, reward potential,
     camera feet height, drop shield, touch detector input), and no V7 file reads the
     V6 value P.REST_HEIGHT (0.152 m)
  4. a simple TRUE-STATE pilot flies one full flight on each path (straight, wave, corners)
     with no errors (checks the code, not the landing skill)
  5. the same seed gives the same flight
  6. the pad has its own random stream: a different pad (speed, path) gives the same start
  7. the ground picture covers the whole flight area (500 m out: ground, not background)
  8. true_state has the pad's acceleration, turn rate, heading, speed and phase

One EGL env for the whole module (V6 B5 rule); extra envs use the TinyRenderer.
Run:  python -m pytest -q -s v7/tests/test_s1d_env.py
"""
import inspect
import time

import numpy as np
import pytest

from v6_drop.envs import lmf2_params as P
from v6_drop.envs.lee_controller import yaw_rotation
from v6_drop.envs.world import OUTCOMES
from v7.envs import landing_env_v7 as E7
from v7.envs import reward_v7 as R7
from v7.envs import world_v7 as W7
from v7.envs.platform_v7 import PAD_TOP_Z
from v7.envs.s500 import REST_HEIGHT

RENDERER = "egl"


# ---------------------------------------------------------------- the true-state test pilot
def pilot_move(env, k_xy=1.5, hold_h=1.2, tau=0.5):
    """A simple pilot that READS THE TRUE STATE (tests and timing only, not the teacher).
    Follows the pad (pad velocity + tau * pad acceleration + k * offset), holds hold_h m above
    the pad top until it is within 0.4 m of the centre, then sinks at 0.5 m/s with the motors
    on (no drop). Points the nose along the pad heading. Sandbox, 15 flights per path (20 Nov):
    straight 15/15, wave 10/15, corners 11/15, random 12/15 safe; the rest mostly tip over at
    the touch (sideways speed mismatch, see S1b). Not a teacher: S2 builds the real one.
    Returns the env action dict (move in [-1, 1])."""
    s = env.world.true_state()
    rel = s["pad_center"] - s["pos"]
    d = float(np.hypot(rel[0], rel[1]))
    v_xy = s["pad_v"][:2] + tau * s["pad_a"][:2] + k_xy * rel[:2]
    n = np.linalg.norm(v_xy)
    if n > E7.VXY_MAX:
        v_xy *= E7.VXY_MAX / n
    h_feet = s["pos"][2] - PAD_TOP_Z - REST_HEIGHT
    vz = -0.5 if d < 0.4 else float(np.clip(1.0 * (hold_h - h_feet), -1.0, 1.0))
    yaw = float(np.arctan2(s["R"][1, 0], s["R"][0, 0]))
    dyaw = (s["pad_psi"] - yaw + np.pi) % (2 * np.pi) - np.pi
    v_head = yaw_rotation(yaw).T @ np.array([v_xy[0], v_xy[1], vz])
    cmd = np.array([*v_head, np.clip(2.0 * dyaw, -E7.YAW_RATE_MAX, E7.YAW_RATE_MAX)])
    return dict(move=(cmd / env.act_scale).astype(np.float32), drop=0)


HOVER = dict(move=np.zeros(4, np.float32), drop=0)


def fly(env, policy, max_steps=None):
    """Run until the end of the episode (or max_steps). Returns (last info, steps, rewards)."""
    n, rewards = 0, []
    while True:
        _, r, term, trunc, info = env.step(policy(env) if callable(policy) else policy)
        n += 1
        rewards.append(r)
        if term or trunc or (max_steps is not None and n >= max_steps):
            return info, n, rewards


def stand_spawn(feet):
    """Level drone over the pad centre, skids `feet` m above the pad top."""
    return dict(pos=[0.0, 0.0, PAD_TOP_Z + REST_HEIGHT + feet], yaw=0.0)


@pytest.fixture(scope="module")
def env():
    # The Stage 1 final setup: drop shield 0.8 m/s and the drone's own touch detector.
    e = E7.LandingEnvV7(renderer=RENDERER, seed=21, touch="sensor", drop_shield=0.8)
    yield e
    e.close()


# ---------------------------------------------------------------- 1. gymnasium checker
def test_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env
    e = E7.LandingEnvV7(renderer="tiny", seed=1)          # tiny: the EGL fixture env is open
    try:
        check_env(e, skip_render_check=True)
    finally:
        e.close()


# ---------------------------------------------------------------- 2. spaces and limits
def test_spaces(env):
    obs, info = env.reset(seed=30000)
    assert obs["image"].shape == (160, 256) and obs["image"].dtype == np.uint8
    assert obs["state"].shape == (11,) and obs["priv"].shape == (8,)
    assert env.observation_space.contains(obs)
    assert env.action_space["move"].shape == (4,) and env.action_space["drop"].n == 2
    assert info["start_marker_ids"] and info["pad_mode"] == "random"


def test_limits_values(env):
    assert np.allclose(env.act_scale, [10.0, 10.0, 1.0, np.radians(60.0)])
    assert np.allclose(env.rate_lim, [0.3, 0.3, 0.3, np.radians(30.0)])
    assert np.allclose(env.vel_scale, [10.0, 10.0, 1.0])
    assert env.priv_scale[6] == 9.0
    assert W7.HORIZON == 600 and W7.LOST_XY == 40.0 and W7.LOST_Z == 12.0
    e = E7.LandingEnvV7(renderer="tiny", seed=1, rate_xy=0.5, v_max=8.0, pitch_deg=45.0)
    try:
        assert np.allclose(e.rate_lim[:2], [0.5, 0.5]) and e.priv_scale[6] == 8.0
        assert e.world.cam.pitch_deg == 45.0 and e.world.v_max == 8.0
    finally:
        e.close()
    with pytest.raises(AssertionError):
        E7.LandingEnvV7(renderer="tiny", rate_xy=0.4)


def test_speed_ramp_reaches_10(env):
    env.reset(seed=30000, options=dict(pad_speed=0.0, mode="straight", spawn=stand_spawn(3.0)))
    applied = []
    for _ in range(36):
        _, _, term, trunc, info = env.step(dict(move=np.array([1, 0, 0, 0], np.float32), drop=0))
        applied.append(info["applied"][0])
        assert not (term or trunc)
    print(f"\n[ramp] vx applied: {np.round(applied[:4], 2).tolist()} ... {applied[-1]:.2f} m/s")
    assert np.allclose(applied[:4], [0.3, 0.6, 0.9, 1.2])
    assert applied[32] == pytest.approx(9.9) and applied[-1] == pytest.approx(10.0)


def test_timeout_at_600(env):
    """The pilot follows the pad but never comes down (hold 3 m): ends by the flight limit."""
    env.reset(seed=30001, options=dict(mode="straight", pad_speed=3.0))
    info, n, _ = fly(env, follow_no_descent)
    print(f"\n[timeout] outcome {info['outcome']} after {n} decisions ({n * P.POLICY_DT:.0f} s)")
    assert info["outcome"] == "timeout" and n == 600 and info["t"] == 600


def follow_no_descent(e):
    """The pilot, but it always holds 3 m above the pad top (never lands)."""
    a = pilot_move(e, hold_h=3.0)
    s = e.world.true_state()
    h_feet = s["pos"][2] - PAD_TOP_Z - REST_HEIGHT
    a["move"][2] = np.clip(1.0 * (3.0 - h_feet), -1.0, 1.0) / e.act_scale[2]
    return a


def test_lost_at_40m(env):
    """A hovering drone and a 9 m/s straight pad: lost when the gap passes 40 m."""
    env.reset(seed=30002, options=dict(mode="straight", pad_speed=9.0))
    info, n, _ = fly(env, HOVER)
    s = env.world.true_state()
    gap = float(np.hypot(*(s["pos"] - s["pad_center"])[:2]))
    print(f"\n[lost] outcome {info['outcome']} after {n} decisions, gap {gap:.1f} m")
    assert info["outcome"] == "lost" and 40.0 < gap < 41.5


def test_priv_pad_speed_scale(env):
    obs, _ = env.reset(seed=30003, options=dict(mode="straight", pad_speed=4.5))
    assert obs["priv"][6] == pytest.approx(4.5 / 9.0, abs=1e-6)


# ---------------------------------------------------------------- 3. standing height 0.22 m
def test_no_v6_standing_height_in_v7_code():
    import v7.envs.world_v7, v7.envs.reward_v7, v7.envs.landing_env_v7
    for mod in (v7.envs.world_v7, v7.envs.reward_v7, v7.envs.landing_env_v7):
        assert "P.REST_HEIGHT" not in inspect.getsource(mod), mod.__name__
    assert REST_HEIGHT == pytest.approx(0.22) and W7.REST_HEIGHT == REST_HEIGHT
    assert R7.REST_HEIGHT == REST_HEIGHT


def test_reward_potential_uses_022():
    assert R7.potential(0.0, REST_HEIGHT + 1.0) == pytest.approx(-1.0)
    assert R7.potential(0.0, REST_HEIGHT) == pytest.approx(0.0)       # V6 value would give -0.068
    assert R7.potential(2.0, REST_HEIGHT + 1.0) == pytest.approx(-2.0)


def test_lowest_point_standing(env):
    env.reset(seed=30004, options=dict(mode="straight", pad_speed=0.0, spawn=stand_spawn(0.0)))
    low = env.world._lowest_z() - PAD_TOP_Z
    print(f"\n[lowest] skids {low * 1000:.2f} mm above the pad top at the standing height")
    assert abs(low) < 1e-3


@pytest.mark.parametrize("feet", [0.05, 0.10])
def test_camera_feet_shield_detector_use_022(env, feet):
    """Level drone over the pad centre, skids `feet` m up: the camera feet height, the drop
    shield and the touch detector's input all read `feet` (V6's 0.152 would read +0.068)."""
    env.reset(seed=30005, options=dict(mode="straight", pad_speed=0.0, spawn=stand_spawn(feet)))
    m = env.last_meas
    assert m.ok and m.confident
    cf = env.cam_feet(m)
    ok, pred, h_feet = env.shield_check()
    det = env.world.detector.cam_feet
    rw = env.reward_fn.phi_last
    print(f"\n[height {feet:.2f}] camera feet {cf:.4f}, shield feet {h_feet:.4f}, "
          f"pred {pred:.3f} m/s, detector {det:.4f}, potential {rw:.4f}")
    for x in (cf, h_feet, det):
        assert x == pytest.approx(feet, abs=0.01)
    assert pred == pytest.approx(np.sqrt(2 * P.G * h_feet), abs=1e-9)
    assert ok == (pred <= 0.8)
    assert rw == pytest.approx(-feet, abs=0.02)        # potential at d ~ 0: -h_feet


# ---------------------------------------------------------------- 4. one flight per path
@pytest.mark.parametrize("mode", ["straight", "wave", "corners"])
def test_pilot_flight_each_path(env, mode):
    obs, info0 = env.reset(seed=30010, options=dict(mode=mode))
    t0 = time.time()
    info, n, rewards = fly(env, pilot_move)
    dt = time.time() - t0
    pad = info["pad"]
    td = info.get("touchdown") or {}
    print(f"\n[pilot {mode}] {info['outcome']} after {n} decisions; pad start "
          f"{pad['v_start']:.1f}, mean {pad['v_mean']:.1f}, max {pad['v_max']:.1f} m/s; "
          f"impact {td.get('impact')}; reward sum {sum(rewards):.2f}; {n / dt:.0f} decisions/s")
    assert info["outcome"] in OUTCOMES
    assert pad["mode"] == mode and info0["pad_mode"] == mode
    assert pad["v_max"] <= 9.0 + 1e-6
    assert all(np.isfinite(rewards))


# ---------------------------------------------------------------- 5. same seed, same flight
def test_same_seed_same_flight(env):
    def run():
        obs, info = env.reset(seed=30020)
        traj = [obs["state"].copy()]
        for _ in range(40):
            obs, _, term, trunc, _ = env.step(pilot_move(env))
            traj.append(obs["state"].copy())
            if term or trunc:
                break
        s = env.world.true_state()
        return info, np.array(traj), s["pos"].copy(), s["pad_center"].copy(), obs["image"].copy()
    a, b = run(), run()
    assert np.allclose(a[0]["spawn"]["pos"], b[0]["spawn"]["pos"])
    assert a[0]["pad_speed"] == b[0]["pad_speed"]
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])
    assert np.array_equal(a[4], b[4])


# ---------------------------------------------------------------- 6. own pad stream
def test_pad_stream_does_not_move_the_start(env):
    starts = []
    for opt in (dict(), dict(mode="corners"), dict(mode="straight", pad_speed=2.0),
                dict(mode="wave")):
        _, info = env.reset(seed=30030, options=opt)
        sp = info["spawn"]
        starts.append((np.asarray(sp["pos"]), sp["yaw"], np.asarray(sp["v"]),
                       np.asarray(sp["w_body"]), info["pad_speed"]))
    for st in starts[1:]:
        assert np.array_equal(st[0], starts[0][0]) and st[1] == starts[0][1]
        assert np.array_equal(st[2], starts[0][2]) and np.array_equal(st[3], starts[0][3])
    assert len({round(st[4], 9) for st in starts}) > 1        # the pads did differ


# ---------------------------------------------------------------- 7. ground picture
def test_ground_picture_far_away(env):
    """A 9 m/s pad drives up to 540 m in 60 s. With the V6 ground box (100 m half side) the
    camera saw the empty background there (grey 181, found 20 Nov). The V7 box is 2 km half
    side. Note: the ground's grey level depends on WHERE the drone is (PyBullet's light sits
    at a fixed point): about 74 to 93 grey in V7's area; V6 had the same (77 at +80 m, 87 at
    -80 m). So this test checks 'ground, not background', not one grey value."""
    w = env.world
    greys = {}
    for xy in ((500.0, 0.0), (-500.0, 0.0), (0.0, 500.0), (0.0, -500.0), (380.0, 380.0)):
        w.plat.reset(np.random.default_rng(0), xy=xy, v=0.0, mode="straight")
        pos = np.array([xy[0] - 2.0, xy[1], PAD_TOP_Z + 3.0])
        w.quad.reset_pose(pos.tolist(), yaw=0.0)
        top = w.cam.render(W7.W6.euler_R(0, 0, 0), pos)[:40]     # ground only, above the pad
        greys[xy] = (float(top.mean()), float(top.std()))
    print("\n[ground] grey (mean, spread) of the ground 500 m out: "
          + ", ".join(f"{k}: {m:.0f}, {sd:.1f}" for k, (m, sd) in greys.items()))
    for m, sd in greys.values():
        assert 65.0 < m < 105.0 and sd < 2.0          # plain ground, not the 181 background
    w.done = True


# ---------------------------------------------------------------- 8. true state for the teacher
def test_true_state_has_pad_motion(env):
    env.reset(seed=30040, options=dict(mode="corners", pad_speed=6.0))
    seen_turn = False
    for _ in range(200):
        _, _, term, trunc, _ = env.step(follow_no_descent(env))
        s = env.world.true_state()
        assert np.allclose(s["pad_a"], env.world.plat.acceleration())
        assert s["pad_speed"] == env.world.plat.v and s["pad_w"] == env.world.plat.w
        assert s["pad_phase"] == env.world.plat.phase
        seen_turn |= abs(s["pad_w"]) > 0.1
        if term or trunc:
            break
    assert seen_turn, "corners mode: the pad should have turned within 20 s"
