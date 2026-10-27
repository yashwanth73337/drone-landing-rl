"""Block 23: reward v6 = v5 + flare (SPEC D30). No training runs.

Run:  python -m pytest v5_shin/tests/test_block23_reward_v6.py -v
"""
import numpy as np
import pytest

from v5_shin.envs.dr import DRConfig
from v5_shin.envs.landing_sim import HOLD_STEPS
from v5_shin.envs.reward import (TouchdownReward, flare_term, geometry, hard_landing_reward,
                                 hold_step_reward)
from v5_shin.envs.shin_env import ShinLandingEnv
from v5_shin.tests.test_block15_hold import SPAWN, place


@pytest.mark.parametrize("vz", [-1.2, -2.0, -3.0, -4.5])
def test_v6_terminal_equals_v5(vz):
    assert hard_landing_reward("v6", vz) == hard_landing_reward("v5", vz)


def test_v6_hold_equals_v5():
    for vrel in (0.0, 0.4, 2.0):
        assert hold_step_reward("v6", vrel) == hold_step_reward("v5", vrel)


@pytest.mark.parametrize("variant,has_flare", [("v5", False), ("v6", True)])
def test_flare_only_in_v6(variant, has_flare):
    env = ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=0, dr=DRConfig.off(),
                         gains_mode="nominal", reward_fn=TouchdownReward(variant), hold_steps=HOLD_STEPS)
    try:
        env.reset(seed=0, options={"c": 0.0, "spawn": dict(SPAWN)})
        place(env.sim, 1.0, -3.5)                       # 1 m above the pad, diving at 3.5 m/s
        env.reward_fn.reset(env)
        info = dict(outcome=None, action_cmd=np.zeros(4))
        r = env.reward_fn(env, info)
        assert ("flare" in info["reward_terms"]) == has_flare
        if has_flare:
            d, dz, vz = geometry(env.sim)
            assert info["reward_terms"]["flare"] == pytest.approx(flare_term(d, dz, vz)) and r < 0
        assert r == pytest.approx(sum(info["reward_terms"].values()))
    finally:
        env.close()
