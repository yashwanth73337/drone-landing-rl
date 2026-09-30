"""Block 8: gymnasium env, observation and action interfaces, privilege boundary.

Run:  python -m pytest v5_shin/tests/test_block8_env.py -v -s
"""
import gymnasium as gym
import numpy as np
import pytest

from v5_shin.envs import dr as DR
from v5_shin.envs.dr import DRConfig
from v5_shin.envs.platform import PAD_TOP_Z
from v5_shin.envs.shin_env import ACTION_SCALE, ACTOR_KEYS, ShinLandingEnv, actor_view


def make(mode="vision", dr=None, seed=0, **kw):
    return ShinLandingEnv(mode=mode, renderer="tiny", egl=False, seed=seed,
                          dr=dr if dr is not None else DRConfig(), **kw)


def test_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env
    env = make()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_observation_contents():
    env = make(dr=DRConfig(forces=True, initial_state=True, visual=True, sensor_noise=False))
    try:
        obs, _ = env.reset(seed=3)
        for _ in range(5):
            obs, *_ = env.step(env.action_space.sample())
        s = env.sim.quad.state()
        Rt = s["R"].T
        dx = Rt @ (env.sim.plat.pad_center() - s["pos"])
        dv = Rt @ (env.sim.plat.velocity() - s["v"])
        assert np.allclose(obs["target"], np.r_[dx, dv], atol=1e-5)
        assert np.array_equal(obs["target"], obs["s_rel"])
        assert np.allclose(obs["critic"][7:], obs["target"])
        assert np.allclose(obs["critic"][:3], Rt @ s["v"], atol=1e-5)          # clean v_body
        assert np.allclose(obs["u"], obs["critic"][:7], atol=1e-6)            # noise off here
        assert obs["critic"][6] >= 0 and obs["u"][6] >= 0                     # q_w >= 0
        assert obs["image"].shape == (160, 256) and obs["image"].dtype == np.uint8
    finally:
        env.close()


def test_u_is_noisy_and_critic_is_clean():
    env = make(mode="privileged")         # no rendering: fast
    try:
        env.reset(seed=4)
        d = []
        for _ in range(300):
            obs, _, te, tr, _ = env.step(np.zeros(4))
            d.append(obs["u"][:3] - obs["critic"][:3])
            if te or tr:
                env.reset()
        d = np.array(d)
        print(f"\n[u noise] v_body noise std {np.round(d.std(0), 4)} (spec {DR.VEL_NOISE_STD})")
        assert np.allclose(d.std(0), DR.VEL_NOISE_STD, rtol=0.15)
    finally:
        env.close()


def test_actor_cannot_see_platform_velocity_directly():
    """Privilege boundary: at reset, two envs differ ONLY in platform velocity. What a
    vision actor may read must be identical; the label (and critic input) must differ."""
    cfg = DRConfig(forces=False, initial_state=False, visual=True, sensor_noise=True)
    a, b = make(dr=cfg, seed=11), make(dr=cfg, seed=11)
    try:
        oa, _ = a.reset(seed=5)
        b.sim._seed_streams(5)
        info = b.sim.reset(c=1.0)
        b.sim.plat.v = a.sim.plat.v + 3.0          # hidden difference
        ob, _ = b._obs()
        # re-draw a's obs with the same noise stream position as b
        a.sim._seed_streams(5)
        a.sim.reset(c=1.0)
        oa, _ = a._obs()
        va, vb = actor_view(oa, "vision"), actor_view(ob, "vision")
        assert set(va) == {"image", "u"}
        assert all(np.array_equal(va[k], vb[k]) for k in va)
        assert not np.allclose(oa["target"][3:], ob["target"][3:])
        assert not np.allclose(oa["critic"], ob["critic"])
    finally:
        a.close()
        b.close()


def test_actor_keys():
    assert ACTOR_KEYS["vision"] == ("image", "u")
    assert ACTOR_KEYS["privileged"] == ("u", "s_rel")
    assert "target" not in ACTOR_KEYS["vision"] and "critic" not in ACTOR_KEYS["vision"]


def test_action_scaling_and_clipping():
    """Speed and yaw rate tested separately. (Combined, a full-rate turn at 10 m/s needs
    ~10.5 m/s^2 centripetal acceleration, i.e. ~47 deg bank, and the BODY-z rate is then
    only cos(tilt) of the world yaw rate. That is physics, not a scaling error.)"""
    env = make(mode="privileged", dr=DRConfig.off(), gains_mode="nominal")
    try:
        sp = dict(pos=np.array([0, 0, PAD_TOP_Z + 8.0]), yaw=0.4, psi_plat=0.0)
        env.reset(seed=0, options={"c": 0.0, "spawn": sp})
        for _ in range(15):     # 1.5 s: stays inside the 15 m drift limit at 10 m/s
            _, _, te, tr, info = env.step(np.array([2.0, 0, 0, 0]))      # clipped to [1, 0, 0, 0]
            assert not (te or tr), info["outcome"]
        assert np.allclose(info["action_cmd"], [10, 0, 0, 0])
        s = env.sim.quad.state()
        yaw = np.arctan2(s["R"][1, 0], s["R"][0, 0])
        v_head = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ s["v"][:2]

        env.reset(seed=0, options={"c": 0.0, "spawn": sp})
        yaws = []
        for _ in range(20):
            _, _, _, _, info = env.step(np.array([0, 0, 0, -3.0]))       # clipped to -1
            R = env.sim.quad.state()["R"]
            yaws.append(np.arctan2(R[1, 0], R[0, 0]))
        assert np.allclose(info["action_cmd"], [0, 0, 0, -np.pi / 3])
        rate = (np.unwrap(yaws)[-1] - np.unwrap(yaws)[-11]) / 1.0     # world yaw rate, last 1 s
        print(f"\n[action] heading-frame v {np.round(v_head, 2)} m/s; world yaw rate {rate:.3f} rad/s")
        assert v_head[0] == pytest.approx(10.0, abs=0.6) and abs(v_head[1]) < 0.3
        assert rate == pytest.approx(-np.pi / 3, rel=0.03)
    finally:
        env.close()


def test_termination_and_truncation_flags():
    env = make(mode="privileged", dr=DRConfig.off())
    try:
        env.reset(seed=0, options={"c": 0.0, "spawn": dict(pos=np.array([0, 0, PAD_TOP_Z + 5]),
                                                          yaw=0.0, psi_plat=0.0)})
        n = 0
        while True:
            _, r, te, tr, info = env.step(np.zeros(4))
            n += 1
            if te or tr:
                break
        assert tr and not te and n == 300 and info["outcome"] == "timeout" and r == 0.0
        env.reset(seed=0, options={"c": 0.0, "spawn": dict(pos=np.array([0, 0, PAD_TOP_Z + 1]),
                                                          yaw=0.0, psi_plat=0.0)})
        while True:
            _, _, te, tr, info = env.step(np.array([0, 0, -1 / 3, 0]))
            if te or tr:
                break
        assert te and not tr and info["outcome"] == "success"
    finally:
        env.close()


def test_render_counts():
    v, pv = make(), make(mode="privileged")
    try:
        for env in (v, pv):
            env.reset(seed=1)
            for _ in range(10):
                env.step(np.zeros(4))
        assert v.sim.cam.n_renders == 11           # 1 reset + 10 steps
        assert pv.sim.cam.n_renders == 0
    finally:
        v.close()
        pv.close()


def test_determinism_with_seed():
    def roll(seed):
        env = make(seed=seed)
        try:
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(0)
            out = [obs["image"].copy(), obs["u"].copy()]
            for _ in range(15):
                obs, *_ = env.step(rng.uniform(-1, 1, 4))
                out += [obs["image"].copy(), obs["u"].copy()]
            return out
        finally:
            env.close()
    a, b, c = roll(7), roll(7), roll(8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_async_vector_env():
    fns = [lambda i=i: make(seed=100 + i) for i in range(2)]
    venv = gym.vector.AsyncVectorEnv(fns, context="spawn")
    try:
        obs, _ = venv.reset(seed=[1, 2])
        assert obs["image"].shape == (2, 160, 256) and obs["u"].shape == (2, 7)
        for _ in range(3):
            obs, r, te, tr, info = venv.step(np.zeros((2, 4), np.float32))
        assert obs["critic"].shape == (2, 13)
    finally:
        venv.close()
