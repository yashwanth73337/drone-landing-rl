"""Block 19: vertical command low-pass (SPEC D25). No training runs.

Run:  python -m pytest v5_shin/tests/test_block19_lowpass.py -v
"""
import numpy as np
import pytest

from v5_shin.envs.dr import DRConfig
from v5_shin.envs.platform import PAD_TOP_Z
from v5_shin.envs.shin_env import ShinLandingEnv

HIGH = dict(pos=np.array([0.0, 0.0, PAD_TOP_Z + 8.0]), yaw=0.0, psi_plat=0.0, v=(0.0, 0.0, 0.0))


def env(alpha):
    e = ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=0, dr=DRConfig.off(),
                       gains_mode="nominal", reward_fn=None, action_lowpass=alpha)
    e.reset(seed=0, options={"c": 0.0, "spawn": dict(HIGH)})
    return e


def cmds(e, actions):
    out = []
    for a in actions:
        _, _, te, tr, info = e.step(np.asarray(a, float))
        assert not (te or tr)
        out.append(info["action_cmd"].copy())
    return np.array(out)


def test_lowpass_filters_vz_hand_computed():
    e = env(0.5)
    try:
        c = cmds(e, [[0.2, 0, -1.0, 0], [0.2, 0, 1.0, 0], [0.2, 0, -1.0, 0], [0.2, 0, -1.0, 0]])
        assert np.allclose(c[:, 2], [-1.0, 0.0, -0.5, -0.75])          # vz_max = 1 m/s
        assert np.allclose(c[:, 0], 0.2 * 10.0)                         # lateral unchanged
    finally:
        e.close()


def test_lowpass_off_by_default_and_alpha_one():
    for alpha in (None, 1.0):
        e = env(alpha)
        try:
            c = cmds(e, [[0, 0, -1.0, 0], [0, 0, 1.0, 0]])
            assert np.allclose(c[:, 2], [-1.0, 1.0])
        finally:
            e.close()


def test_lowpass_resets_each_episode():
    e = env(0.5)
    try:
        cmds(e, [[0, 0, 1.0, 0], [0, 0, 1.0, 0]])
        e.reset(seed=1, options={"c": 0.0, "spawn": dict(HIGH)})
        c = cmds(e, [[0, 0, -1.0, 0]])
        assert c[0, 2] == pytest.approx(-1.0)                           # no carry-over
    finally:
        e.close()


def test_lowpass_bad_alpha():
    with pytest.raises(AssertionError):
        env(0.0)
