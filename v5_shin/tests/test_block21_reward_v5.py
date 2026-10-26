"""Block 21: reward v5 (SPEC D27). No training runs.

Run:  python -m pytest v5_shin/tests/test_block21_reward_v5.py -v
"""
import pytest

from v5_shin.envs.reward import (R_HOLD_FAIL_V4, TouchdownReward, centre_bonus, hard_landing_reward,
                                 hold_step_reward)


@pytest.mark.parametrize("vz,expect", [(-1.01, -5.025), (-1.2, -5.5), (-2.0, -7.5), (-2.5, -8.75),
                                       (-3.0, -10.0), (-4.5, -10.0)])
def test_v5_hard_landing(vz, expect):
    assert hard_landing_reward("v5", vz) == pytest.approx(expect)
    assert hard_landing_reward("v5", vz) <= R_HOLD_FAIL_V4      # still worse than a failed hold


def test_v5_keeps_a_gradient_where_v4_is_flat():
    assert hard_landing_reward("v4", -2.2) == hard_landing_reward("v4", -2.8) == -10.0
    assert hard_landing_reward("v5", -2.2) > hard_landing_reward("v5", -2.8)


def test_v5_hold_terms_equal_v4():
    for vrel in (0.0, 0.3, 1.0, 2.0):
        assert hold_step_reward("v5", vrel) == hold_step_reward("v4", vrel)
    assert TouchdownReward("v5").variant == "v5"
    assert centre_bonus((0.0, 0.0)) == pytest.approx(2.0)
