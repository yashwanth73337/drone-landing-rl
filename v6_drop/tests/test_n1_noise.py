"""v6_drop N1 (10 Nov): sensor noise on the drone's own estimates (envs/sensor_noise.py,
World(noise), LandingEnv(sensor_noise), eval_student / final_eval / replay_flight --noise).

Checks:
  1. the drifting error has the right size (std) and memory (correlation after 0.5 s ~ 1/e);
     camera grain has the right std; reseed repeats the noise exactly
  2. R_to_quat matches PyBullet's quaternion (same rotation)
  3. default (no noise): the old observation exactly (quaternion = PyBullet's)
  4. noise changes ONLY what the drone senses: with the same commands, the true flight
     (positions) is bit-for-bit the same with and without noise; the observed speed differs
     from the truth by about the noise level
  5. the same seed gives the same noisy observations (repeatable)
  6. the touch detector and the shield read the NOISY values
  7. eval_student.noise_kw: none -> the old call; a level -> argument + file prefix

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_n1_noise.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.envs.sensor_noise import LEVELS, SensorNoise, R_to_quat
from v6_drop.scripts.eval_student import noise_kw


def test_noise_statistics_and_repeat():
    n = SensorNoise("medium", seed=5)
    xs, rs = [], []
    for _ in range(200_000):
        n.advance()
        xs.append(n.v_err.copy())
        rs.append(n.r_err.copy())
    xs, rs = np.array(xs), np.array(rs)
    std_v, std_r = xs.std(axis=0), np.degrees(rs.std(axis=0))
    lag = 50                                              # 0.5 s = TAU
    c = np.corrcoef(xs[:-lag, 0], xs[lag:, 0])[0, 1]
    img = np.full((320, 512), 120, np.uint8)
    g = n.image(img).astype(float)
    print(f"\n[n1] speed std {np.round(std_v, 4)} (want 0.05), tilt std {np.round(std_r, 3)} deg (want 1.0), "
          f"corr after 0.5 s {c:.3f} (want {np.exp(-1):.3f}), pixel std {g.std():.2f} (want 5)")
    assert np.allclose(std_v, 0.05, rtol=0.08) and np.allclose(std_r, 1.0, rtol=0.08)
    assert abs(c - np.exp(-1)) < 0.05 and abs(g.std() - 5.0) < 0.1
    a, b = SensorNoise("high", seed=9), SensorNoise("high", seed=9)
    for _ in range(100):
        a.advance(), b.advance()
    assert np.array_equal(a.v_err, b.v_err) and np.array_equal(a.image(img), b.image(img))
    assert SensorNoise("none").image(img) is img and LEVELS["low"]["v"] == 0.02


def test_R_to_quat():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = p.getQuaternionFromEuler(rng.uniform(-3.1, 3.1, 3))
        R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
        q2 = R_to_quat(R)
        assert min(np.abs(q2 - q).max(), np.abs(q2 + q).max()) < 1e-9


def spawn(env, seed=1, feet=1.0, v=1.0):
    pos = [0.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + feet]
    return env.reset(seed=seed, options=dict(pad_speed=v, spawn=dict(pos=pos, yaw=0.0, v=[v, 0.0, 0.0])))


ACT = {"move": np.array([1.0 / E.ACT_SCALE[0], 0.0, 0.0, 0.0], np.float32), "drop": 0}   # hold 1 m up


def test_default_is_old_observation():
    env = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal")
    try:
        assert env.noise is None and env.world.noise is None
        obs, _ = spawn(env)
        q = np.array(env.world.true_state()["quat"])
        q = -q if q[3] < 0 else q
        assert np.allclose(obs["state"][3:7], q, atol=1e-7)
    finally:
        env.close()


def fly(env, n=20):
    obs, _ = spawn(env)
    out = []
    for _ in range(n):
        obs, _, term, trunc, info = env.step(ACT)
        s = env.world.true_state()
        out.append((s["pos"].copy(), obs["state"].copy(), (s["R"].T @ s["v"]) / E.VEL_SCALE))
        assert not (term or trunc)
    return out


def test_noise_only_changes_what_the_drone_senses():
    clean = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal")
    noisy = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal", sensor_noise="high")
    try:
        a, b = fly(clean), fly(noisy)
    finally:
        clean.close()
        noisy.close()
    pos_same = all(np.array_equal(x[0], y[0]) for x, y in zip(a, b))
    err = np.array([(y[1][:3] - y[2]) * E.VEL_SCALE for y in b])           # observed - true body speed
    print(f"\n[n1] true flight identical: {pos_same}; observed speed error std {np.round(err.std(axis=0), 3)} "
          f"m/s (level high = 0.10)")
    assert pos_same
    assert np.allclose(np.array([x[1] for x in a])[:, :3], np.array([x[2] for x in a])[:, :3], atol=1e-6)
    assert 0.02 < float(np.abs(err).mean()) < 0.3


def test_repeatable():
    outs = []
    for _ in range(2):
        env = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal", sensor_noise="medium")
        try:
            obs0, _ = spawn(env, seed=4)
            obs1 = env.step(ACT)[0]
            outs.append((obs0["image"].copy(), obs1["state"].copy(), obs1["image"].copy()))
        finally:
            env.close()
    assert all(np.array_equal(x, y) for x, y in zip(*outs))


def test_detector_and_shield_read_noisy_values():
    env = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal", touch="sensor", sensor_noise="high",
                       drop_shield=0.8)
    try:
        spawn(env, feet=0.05, v=1.0)
        env.step(ACT)
        w = env.world
        true_vz = float(w.quad.state()["v"][2])
        assert w.own_vz() == pytest.approx(true_vz + float(w.noise.v_err[2]))
        assert abs(w.own_vz() - true_vz) > 1e-6
        R_own, v_own = env.own()
        s = w.true_state()
        assert not np.allclose(R_own, s["R"]) and not np.allclose(v_own, s["v"])
        ok, pred, h = env.shield_check()
        if pred is not None:                               # uses own sink: sqrt(sink^2 + 2 g h)
            assert pred == pytest.approx(np.sqrt(max(-v_own[2], 0) ** 2 + 2 * P.G * h))
    finally:
        env.close()


def test_noise_kw():
    assert noise_kw("none") == ({}, "") and noise_kw(None) == ({}, "")
    assert noise_kw("medium") == (dict(sensor_noise="medium"), "noisemedium_")
