"""Block 9: reward (Table III + terminals, D3 = 'literal' (default), 'prose' ablation, D4 sign).

Run:  python -m pytest v5_shin/tests/test_block9_reward.py -v -s
"""
import numpy as np
import pytest

from v5_shin.envs.dr import DRConfig
from v5_shin.envs.platform import PAD_TOP_Z
from v5_shin.envs.reward import ShinReward, geometry, shaping_terms
from v5_shin.envs.shin_env import ACTION_SCALE, ShinLandingEnv


# ---------------------------------------------------------------- hand-computed terms
@pytest.mark.parametrize("args,expect", [
    # (d_prev, d, adz_prev, adz, dz, vz, yaw_cmd) -> weighted terms
    ((3.0, 2.5, 5.0, 4.8, -4.8, -0.2, 0.0),
     dict(lateral=0.5, vertical=0.2 / 2.5, vz=0.0, undershoot=0.0, yaw=0.0)),
    ((0.5, 0.4, 3.0, 2.9, -2.9, -0.5, 0.0),        # d < 1 -> divide by 1
     dict(lateral=0.1, vertical=0.1, vz=0.0, undershoot=0.0, yaw=0.0)),
    ((4.0, 1.5, 6.0, 4.0, -4.0, -1.3, 0.3),        # clips at +1; vz penalty 0.5*0.8
     dict(lateral=1.0, vertical=1.0 / 1.5, vz=-0.4, undershoot=0.0, yaw=-0.6)),
    ((1.0, 3.0, 2.0, 2.5, -2.5, 0.4, -0.5),        # moving away: clip at -1; climbing: no penalty
     dict(lateral=-1.0, vertical=-0.5 / 3.0, vz=0.0, undershoot=0.0, yaw=-1.0)),
    ((2.0, 2.0, 0.1, 0.3, 0.3, -0.2, 0.0),         # below the pad top by 0.3 m: undershoot
     dict(lateral=0.0, vertical=-0.2 / 2.0, vz=0.0, undershoot=-0.3, yaw=0.0)),
])
def test_shaping_terms_hand_computed(args, expect):
    got = shaping_terms(*args, vz_penalty="prose")
    for k, v in expect.items():
        assert got[k] == pytest.approx(v, abs=1e-12), (k, got[k], v)


@pytest.mark.parametrize("vz,prose,literal", [
    (0.0, 0.0, -0.25), (-0.5, 0.0, 0.0), (-1.0, -0.25, 0.0), (-2.5, -1.0, 0.0), (1.0, 0.0, -0.75)])
def test_vz_penalty_variants(vz, prose, literal):
    """D3: 'prose' penalises descent faster than 0.5 m/s; 'literal' (as printed)
    penalises anything slower than a 0.5 m/s descent, including hover."""
    a = dict(d_prev=1, d=1, adz_prev=1, adz=1, dz=-1, vz=vz, yaw_cmd=0)
    assert shaping_terms(**a, vz_penalty="prose")["vz"] == pytest.approx(prose)
    assert shaping_terms(**a, vz_penalty="literal")["vz"] == pytest.approx(literal)


# ---------------------------------------------------------------- env integration
def make(**kw):
    return ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=0,
                          dr=kw.pop("dr", DRConfig()), gains_mode="nominal", **kw)


def test_env_reward_matches_independent_recomputation():
    env = make()
    try:
        env.reset(seed=21, options={"c": 1.0})
        rng = np.random.default_rng(1)
        d0, dz0, _ = geometry(env.sim)
        prev = (d0, abs(dz0))
        for _ in range(40):
            a = rng.uniform(-0.3, 0.3, 4)
            _, r, te, tr, info = env.step(a)
            d, dz, vz = geometry(env.sim)
            exp = sum(shaping_terms(prev[0], d, prev[1], abs(dz), dz, vz,
                                    np.clip(a, -1, 1)[3] * ACTION_SCALE[3],
                                    env.reward_fn.vz_penalty).values())
            prev = (d, abs(dz))
            if te:
                break
            assert r == pytest.approx(exp, abs=1e-9)
            assert r == pytest.approx(sum(info["reward_terms"].values()), abs=1e-12)
    finally:
        env.close()


@pytest.mark.parametrize("spawn,cmd,outcome,reward", [
    (dict(pos=[0, 0, PAD_TOP_Z + 1.0]), [0, 0, -1.0, 0], "success", 10.0),
    (dict(pos=[3.0, 0, PAD_TOP_Z + 1.0]), [0, 0, -1.0, 0], "crash_ground", -10.0),
    (dict(pos=[-2.0, 0, 0.5]), [2.0, 0, 0, 0], "crash_platform", -10.0),
    (dict(pos=[0, 0, PAD_TOP_Z + 4.0]), [10.0, 0, 0, 0], "drift", -10.0),
    (dict(pos=[0, 0, PAD_TOP_Z + 5.0], roll=np.radians(85)), [0, 0, 0, 0], "tilt", -10.0),
])
def test_terminal_rewards(spawn, cmd, outcome, reward):
    env = make(dr=DRConfig.off())
    try:
        sp = dict(yaw=0.0, psi_plat=0.0, **{k: np.asarray(v, float) if k == "pos" else v
                                             for k, v in spawn.items()})
        env.reset(seed=0, options={"c": 0.0, "spawn": sp})
        while True:
            _, r, te, tr, info = env.step(np.array(cmd) / ACTION_SCALE)
            if te or tr:
                break
        assert info["outcome"] == outcome and r == reward
    finally:
        env.close()


def test_default_is_literal():
    env = make()
    try:
        assert env.reward_fn.vz_penalty == "literal"
    finally:
        env.close()


def test_timeout_step_gets_shaping_not_terminal():
    """Default reward (D3 literal): a hover costs 0.5 * 0.5 = 0.25 per step, and the timeout
    step gets that shaping, not a terminal value."""
    env = make(dr=DRConfig.off())
    try:
        env.reset(seed=0, options={"c": 0.0, "spawn": dict(pos=np.array([0, 0, PAD_TOP_Z + 5]),
                                                          yaw=0.0, psi_plat=0.0)})
        rs = []
        while True:
            _, r, te, tr, info = env.step(np.zeros(4))
            rs.append(r)
            if te or tr:
                break
        assert tr and info["outcome"] == "timeout"
        assert rs[-1] == pytest.approx(-0.25, abs=2e-3)
        assert sum(rs) == pytest.approx(-75.0, abs=0.5)
    finally:
        env.close()


# ---------------------------------------------------------------- landscape facts
def test_reward_landscape_facts():
    """Records the landscape the SPEC reports (scripts/reward_profile.py). These are facts
    about the decided reward, not goals: they must change only when the reward changes."""
    from v5_shin.scripts.reward_profile import fly
    hover = fly(0.0, "prose")
    slow = fly(0.5, "prose")
    fast = fly(1.5, "prose")
    lit_hover = fly(0.0, "literal")
    print(f"\n[landscape prose] disc. return hover {hover['disc']:.2f}, 0.5 m/s {slow['disc']:.2f}, "
          f"1.5 m/s {fast['disc']:.2f}; literal hover {lit_hover['disc']:.2f}")
    assert hover["outcome"] == "timeout" and abs(hover["ret"]) < 0.05
    assert slow["outcome"] == "success" and slow["disc"] > hover["disc"] + 5
    assert fast["disc"] < hover["disc"]           # prose: a 1.5 m/s landing is worth less than hovering
    assert lit_hover["disc"] < -20
    lit_fast = fly(1.5, "literal")
    assert lit_fast["disc"] > fly(0.5, "literal")["disc"] > lit_hover["disc"]   # literal: faster is better
