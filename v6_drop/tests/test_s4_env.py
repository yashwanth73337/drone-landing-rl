"""v6_drop Step 4: observations and actions (LandingEnv).

Checks:
  1. Gymnasium's own checker accepts the environment
  2. spaces: image (160, 256) uint8, state (11,), priv (8,); action move (4,) + drop {0, 1}
  3. action scaling + rate limit: asking full forward from rest gives 0.3, 0.6, 0.9 ... m/s,
     reaching 6 m/s after 20 decisions; yaw rate moves 30 deg/s per decision
  4. the rate limit keeps the nose-down tilt small when the policy asks for full speed
     at once (and without it the tilt would be large)
  5. the 'state' carries the APPLIED command (not the asked one), velocity in the body
     frame, and a unit quaternion with w >= 0
  6. privilege boundary: 'state' does not change when only the PAD changes; 'priv' does,
     and it matches the simulator truth. ACTOR_KEYS never contains 'priv'.
  7. drop=1 ends the episode with a C1 outcome; drop=0 keeps flying
  8. the image is the shrunk camera frame; exactly one render per step
  9. same seed -> identical observations
 10. timing (printed)

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s4_env.py -v -s
"""
import time

import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs import world as W
from v6_drop.envs.platform_s1 import PAD_TOP_Z


def _egl_ok():
    try:
        w = W.World(renderer="egl")
        w.close()
        return True
    except Exception:
        return False


RENDERER = "egl" if _egl_ok() else "tiny"


@pytest.fixture(scope="module")
def env():
    e = E.LandingEnv(renderer=RENDERER, seed=21)
    yield e
    e.close()


def act(move=(0, 0, 0, 0), drop=0):
    return {"move": np.array(move, dtype=np.float32), "drop": drop}


def hover_spawn(z=3.0, x=-2.0):
    return dict(pos=[x, 0.0, PAD_TOP_Z + z], yaw=0.0)


def tilt_deg(env):
    R = env.world.true_state()["R"]
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))


# ---------------------------------------------------------------- 1. gymnasium checker
def test_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env
    e = E.LandingEnv(renderer="tiny", seed=1)          # tiny: the EGL fixture env is open
    try:
        check_env(e, skip_render_check=True)
    finally:
        e.close()


# ---------------------------------------------------------------- 2. spaces
def test_spaces(env):
    obs, info = env.reset()
    assert obs["image"].shape == (160, 256) and obs["image"].dtype == np.uint8
    assert obs["state"].shape == (11,) and obs["state"].dtype == np.float32
    assert obs["priv"].shape == (8,) and obs["priv"].dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.action_space["move"].shape == (4,) and env.action_space["drop"].n == 2
    assert info["start_marker_ids"], "start rule: at least one marker detected"


# ---------------------------------------------------------------- 3. scaling + rate limit
def test_rate_limited_ramp(env):
    env.reset(options=dict(pad_speed=0.0, spawn=hover_spawn()))
    applied = []
    for _ in range(22):
        _, _, term, trunc, info = env.step(act((1, 0, 0, 1)))
        applied.append(info["applied"].copy())
        assert not (term or trunc)
    applied = np.array(applied)
    print(f"\n[ramp] vx applied: {np.round(applied[:5, 0], 2).tolist()} ... {applied[-1, 0]:.2f} m/s; "
          f"yaw rate: {np.round(np.degrees(applied[:3, 3]), 1).tolist()} deg/s")
    assert np.allclose(applied[:4, 0], [0.3, 0.6, 0.9, 1.2])
    assert applied[19, 0] == pytest.approx(6.0) and applied[-1, 0] == pytest.approx(6.0)
    assert np.allclose(np.degrees(applied[:3, 3]), [30.0, 60.0, 60.0])
    assert np.allclose(info["asked"], E.ACT_SCALE * np.array([1, 0, 0, 1]))


def test_vertical_range(env):
    env.reset(options=dict(pad_speed=0.0, spawn=hover_spawn(z=3.0)))
    for _ in range(10):
        _, _, _, _, info = env.step(act((0, 0, -1, 0)))
    assert info["applied"][2] == pytest.approx(-1.0)        # A1: vz limited to 1 m/s


# ---------------------------------------------------------------- 4. tilt stays small
def test_rate_limit_keeps_tilt_small(env):
    env.reset(options=dict(pad_speed=0.0, spawn=hover_spawn(z=4.0, x=-10.0)))
    peak = 0.0
    for _ in range(30):
        env.step(act((1, 0, 0, 0)))                          # ask for 6 m/s at once
        peak = max(peak, tilt_deg(env))
    # Same request WITHOUT the rate limit, in a separate tiny-renderer env
    e = E.LandingEnv(renderer="tiny", seed=21)
    try:
        e.reset(options=dict(pad_speed=0.0, spawn=hover_spawn(z=4.0, x=-10.0)))
        peak_free = 0.0
        for _ in range(30):
            e.applied = np.zeros(4)                          # bypass: jump straight to target
            e.world.step(E.ACT_SCALE * np.array([1, 0, 0, 0]))
            peak_free = max(peak_free, tilt_deg(e))
    finally:
        e.close()
    print(f"\n[tilt] full forward asked at once: peak tilt {peak:.1f} deg with the rate limit, "
          f"{peak_free:.1f} deg without")
    assert peak < 25.0 and peak_free > 35.0


# ---------------------------------------------------------------- 5. state content
def test_state_content(env):
    env.reset(options=dict(pad_speed=0.0, spawn=hover_spawn()))
    obs, _, _, _, info = env.step(act((0.5, 0, 0, 0)))
    s = env.world.true_state()
    v_body = obs["state"][:3] * E.VEL_SCALE
    q = obs["state"][3:7]
    cmd = obs["state"][7:11] * E.ACT_SCALE
    assert np.allclose(v_body, s["R"].T @ s["v"], atol=1e-5)
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-5) and q[3] >= 0
    assert np.allclose(cmd, info["applied"], atol=1e-5)          # APPLIED, not asked
    assert not np.allclose(cmd, info["asked"])                    # (asked 3.0, applied 0.3)


# ---------------------------------------------------------------- 6. privilege boundary
def test_actor_state_has_no_pad_information(env):
    states, privs = [], []
    for pad_speed in (0.0, 3.5):
        obs, _ = env.reset(options=dict(pad_speed=pad_speed, spawn=hover_spawn()))
        for _ in range(5):
            obs, *_ = env.step(act((0.2, 0, 0, 0)))
        states.append(obs["state"].copy())
        privs.append(obs["priv"].copy())
        s = env.world.true_state()
        expect = np.concatenate([s["rel_pos_body"], s["rel_vel_body"], [env.world.plat.v],
                                 [s["pos"][2] - s["pad_center"][2]]]) / E.PRIV_SCALE
        assert np.allclose(obs["priv"], expect, atol=1e-5)
    print(f"\n[privilege] |state difference| = {np.abs(states[0] - states[1]).max():.2e}; "
          f"|priv difference| = {np.abs(privs[0] - privs[1]).max():.3f}")
    assert np.allclose(states[0], states[1], atol=1e-6)       # pad speed is invisible here
    assert not np.allclose(privs[0], privs[1])
    assert "priv" not in E.ACTOR_KEYS and set(E.ACTOR_KEYS) < set(E.CRITIC_KEYS)


# ---------------------------------------------------------------- 7. drop
def test_drop_ends_episode(env):
    env.reset(options=dict(pad_speed=2.0,
                           spawn=dict(pos=[0.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + 0.03], yaw=0.0,
                                      v=[2.0, 0.0, 0.0])))
    _, _, term, trunc, info = env.step(act((2 / 6, 0, 0, 0), drop=0))
    assert not (term or trunc) and info["outcome"] is None and info["drop"] is False
    _, _, term, trunc, info = env.step(act((2 / 6, 0, 0, 0), drop=1))
    print(f"\n[drop] outcome={info['outcome']} impact={info['touchdown']['impact']:.3f} m/s")
    assert term and not trunc and info["drop"] is True
    assert info["outcome"] in W.OUTCOMES


# ---------------------------------------------------------------- 8. image
def test_image_is_shrunk_frame_one_render_per_step(env):
    env.reset(options=dict(pad_speed=0.0, spawn=hover_spawn()))
    n0 = env.world.cam.n_renders
    obs, *_ = env.step(act())
    assert env.world.cam.n_renders == n0 + 1
    full = env.world.render()
    assert np.array_equal(obs["image"], E.shrink(full))


# ---------------------------------------------------------------- 9. determinism
def test_same_seed_same_observations():
    runs = []
    for _ in range(2):
        e = E.LandingEnv(renderer="tiny", seed=77)
        try:
            obs, _ = e.reset(seed=5)
            seq = [obs["state"].copy()]
            for _ in range(5):
                obs, *_ = e.step(act((0.3, 0.1, -0.2, 0.0)))
                seq.append(obs["state"].copy())
            runs.append((np.array(seq), obs["image"].copy()))
        finally:
            e.close()
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])


# ---------------------------------------------------------------- 10. timing
def test_timing(env):
    env.reset()
    n, t0 = 100, time.time()
    for _ in range(n):
        _, _, term, trunc, _ = env.step(act((0.3, 0, -0.2, 0)))
        if term or trunc:
            env.reset()
    print(f"\n[timing {RENDERER}] {n / (time.time() - t0):.0f} env steps/s "
          f"(physics + render + shrink; no ArUco)")
