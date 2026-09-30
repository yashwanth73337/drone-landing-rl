"""Block 10: curriculum logic on synthetic success streams, plus env integration.

Run:  python -m pytest v5_shin/tests/test_block10_curriculum.py -v -s
"""
import gymnasium as gym
import numpy as np
import pytest

from v5_shin.envs import curriculum as CU
from v5_shin.envs.curriculum import LEVELS, Curriculum, false_promotion_probability, level_to_c
from v5_shin.envs.dr import DRConfig
from v5_shin.envs.platform import V0_MAX
from v5_shin.envs.shin_env import ShinLandingEnv


def feed(cur, p, n, rng):
    for _ in range(n):
        cur.record(rng.random() < p, cur.c)


def test_levels_and_c():
    assert LEVELS == (10, 20, 30, 40, 50, 60, 70, 80)
    assert level_to_c(10) == 0.125 and level_to_c(80) == 1.0
    assert Curriculum().c == 0.125


def test_high_success_promotes_every_window_and_caps_at_80():
    rng = np.random.default_rng(10000)
    cur = Curriculum()
    levels = []
    for _ in range(9):
        feed(cur, 0.95, CU.WINDOW, rng)
        levels.append(cur.level)
    assert levels == [20, 30, 40, 50, 60, 70, 80, 80, 80]
    assert [h["promoted"] for h in cur.history] == [True] * 7 + [False] * 2


def test_low_success_never_promotes_or_demotes():
    rng = np.random.default_rng(10001)
    cur = Curriculum(start_level=40)
    feed(cur, 0.70, 20 * CU.WINDOW, rng)
    assert cur.level == 40 and len(cur.history) == 20


def test_threshold_is_inclusive():
    for k, expect in ((410, 50), (409, 40)):          # 410/512 = 0.8008, 409/512 = 0.7988
        cur = Curriculum(start_level=40)
        for i in range(CU.WINDOW):
            cur.record(i < k, cur.c)
        assert cur.level == expect, (k, cur.level)


def test_stale_episodes_are_not_counted():
    cur = Curriculum(start_level=10)
    for _ in range(CU.WINDOW):
        cur.record(True, 0.125)
    assert cur.level == 20
    for _ in range(100):                               # in-flight episodes from level 10
        cur.record(False, 0.125)
    assert cur.stale == 100 and cur.n == 0
    for _ in range(CU.WINDOW):
        cur.record(True, 0.25)
    assert cur.level == 30


def test_state_dict_roundtrip():
    rng = np.random.default_rng(3)
    a = Curriculum()
    feed(a, 0.9, 1300, rng)
    b = Curriculum.from_state_dict(a.state_dict())
    r1, r2 = np.random.default_rng(4), np.random.default_rng(4)
    feed(a, 0.85, 3000, r1)
    feed(b, 0.85, 3000, r2)
    assert a.state_dict() == b.state_dict()


def test_promotion_noise_is_small():
    """Contrast with notes §7: a single 20-episode eval at 90% threshold promoted a true-80%
    policy ~1 in 5. With 512-episode windows at 80%, report the exact binomial tails."""
    rows = {p: false_promotion_probability(p) for p in (0.70, 0.75, 0.78, 0.80, 0.82, 0.85)}
    print("\n[promotion] P(promote per window | true p): "
          + ", ".join(f"{p:.2f}: {v:.4f}" for p, v in rows.items()))
    assert rows[0.75] < 0.01 and rows[0.85] > 0.99
    assert 0.4 < rows[0.80] < 0.6


def test_env_set_c_applies_at_next_reset_and_tags_episodes():
    cfg = DRConfig.off()
    env = ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=0, dr=cfg, c=0.125)
    try:
        v0 = []
        for i in range(200):
            env.reset(seed=i)
            v0.append(env.sim.plat.v)
        assert max(v0) <= V0_MAX * 0.125 + 1e-9
        env.reset(seed=1)
        env.set_c(1.0)                                  # mid-episode: must not affect it
        _, _, _, _, info = env.step(np.zeros(4))
        assert info["c_episode"] == 0.125
        v0 = []
        for i in range(200):
            env.reset(seed=1000 + i)
            v0.append(env.sim.plat.v)
        _, _, _, _, info = env.step(np.zeros(4))
        assert info["c_episode"] == 1.0 and max(v0) > 4.0
    finally:
        env.close()


def test_vector_env_call_set_c():
    fns = [lambda i=i: ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=i,
                                      dr=DRConfig.off(), c=0.125) for i in range(2)]
    venv = gym.vector.AsyncVectorEnv(fns, context="spawn")
    try:
        venv.reset(seed=[0, 1])
        venv.call("set_c", 0.5)
        assert list(venv.get_attr("c")) == [0.5, 0.5]
    finally:
        venv.close()
