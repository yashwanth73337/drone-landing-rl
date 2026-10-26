"""Block 18: smooth pad speed changes and the hold sit-down range (SPEC D23). No training runs.

Run:  python -m pytest v5_shin/tests/test_block18_smooth_sitdown.py -v
"""
import numpy as np
import pytest

from v5_shin.envs import lmf2_params as P
from v5_shin.envs.dr import DRConfig
from v5_shin.envs.landing_sim import HOLD_STEPS, LandingSim
from v5_shin.envs.platform import DV_MAX, DW_MAX, PAD_TOP_Z
from v5_shin.envs.reward import TouchdownReward
from v5_shin.envs.shin_env import ShinLandingEnv

HIGH = dict(pos=np.array([0.0, 0.0, PAD_TOP_Z + 8.0]), yaw=0.0, psi_plat=0.0, v=(0.0, 0.0, 0.0))


def fly_high(pad_smooth, n=12, c=1.0, record=None):
    """Drone hovering 8 m above the pad (never touches it); returns the pad random-walk targets."""
    sim = LandingSim(renderer="tiny", marker=False, dr=DRConfig.off(), seed=11, pad_smooth=pad_smooth)
    try:
        sim.reset(c=c, spawn=dict(HIGH))
        if record is not None:
            orig = sim.plat.integrate
            def integrate(dt=P.PHYSICS_DT):
                record.append((sim.plat.v, sim.plat.w))
                orig(dt)
            sim.plat.integrate = integrate
        out = []
        for _ in range(n):
            te, tr, info = sim.step(np.zeros(4))
            assert not (te or tr), info["outcome"]
            out.append(sim.pad_target())
        return out
    finally:
        sim.close()


def test_smooth_keeps_the_random_walk():
    """Same draws, same clipping: the v, w at every 0.1 s mark are identical."""
    a, b = fly_high(False), fly_high(True)
    assert np.allclose(a, b, atol=0, rtol=0)


def test_smooth_spreads_each_change_over_the_step():
    rec_on, rec_off = [], []
    fly_high(True, record=rec_on)
    fly_high(False, record=rec_off)
    on, off = np.array(rec_on), np.array(rec_off)
    n = P.SUBSTEPS
    d_on = np.abs(np.diff(on, axis=0))
    assert d_on[:, 0].max() <= DV_MAX / n + 1e-9           # at most 0.05 m/s per substep at c = 1
    assert d_on[:, 1].max() <= DW_MAX / n + 1e-9
    assert np.abs(np.diff(off, axis=0))[:, 0].max() > DV_MAX / n   # the paper model jumps
    # the last substep of each step reaches the step's random-walk value
    assert np.allclose(on[n - 1::n], off[n - 1::n])


def test_smooth_off_by_default():
    sim = LandingSim(renderer="tiny", marker=False, dr=DRConfig.off(), seed=0)
    try:
        assert sim.pad_smooth is False
    finally:
        sim.close()


# ---------------------------------------------------------------- hold sit-down range
def env(hold_vz_down):
    return ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=0, dr=DRConfig.off(),
                          gains_mode="nominal", reward_fn=TouchdownReward("v4"),
                          hold_steps=HOLD_STEPS, hold_vz_down=hold_vz_down)


def test_sitdown_scale_only_in_hold_and_only_downward():
    e = env(5.0)
    try:
        e.reset(seed=0, options={"c": 0.0, "spawn": dict(HIGH)})               # flight
        _, _, _, _, info = e.step(np.array([0, 0, -1.0, 0]))
        assert info["action_cmd"][2] == pytest.approx(-1.0)
        e.reset(seed=0, options={"c": 0.0, "on_pad": True})                    # hold
        _, _, _, _, info = e.step(np.array([0, 0, -1.0, 0]))
        assert info["action_cmd"][2] == pytest.approx(-5.0)
        e.reset(seed=0, options={"c": 0.0, "on_pad": True})
        _, _, _, _, info = e.step(np.array([0, 0, 0.5, 0]))                    # upward: unchanged
        assert info["action_cmd"][2] == pytest.approx(0.5)
    finally:
        e.close()


def test_sitdown_off_by_default():
    e = env(None)
    try:
        e.reset(seed=0, options={"c": 0.0, "on_pad": True})
        _, _, _, _, info = e.step(np.array([0, 0, -1.0, 0]))
        assert info["action_cmd"][2] == pytest.approx(-1.0)
    finally:
        e.close()


def test_sitdown_must_not_be_below_vz_max():
    with pytest.raises(AssertionError):
        env(0.5)


# ---------------------------------------------------------------- CLI
def test_train_script_d23_options_end_to_end():
    import csv, json, os, shutil, subprocess, sys
    name = "_pytest_d23"
    run = os.path.join(os.path.dirname(__file__), "..", "runs", name)
    shutil.rmtree(run, ignore_errors=True)
    try:
        cmd = [sys.executable, "-m", "v5_shin.scripts.train_ppo", "--mode", "privileged", "--name", name,
               "--reward", "v4", "--hold-practice", "0.5", "--hold-vz-down", "5", "--pad-smooth",
               "--total-steps", "64", "--n-envs", "2", "--n-steps", "16", "--seq-len", "4",
               "--minibatches", "2", "--renderer", "tiny", "--device", "cpu", "--ckpt-every", "1"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                             cwd=os.path.join(os.path.dirname(__file__), "..", ".."))
        assert res.returncode == 0, res.stderr[-3000:]
        cfg = json.load(open(os.path.join(run, "config.json")))
        assert cfg["hold_vz_down"] == 5.0 and cfg["pad_smooth"] is True
        assert len(list(csv.DictReader(open(os.path.join(run, "updates.csv"))))) == 2
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_train_script_rejects_sitdown_without_hold():
    import os, shutil, subprocess, sys
    name = "_pytest_d23_bad"
    run = os.path.join(os.path.dirname(__file__), "..", "runs", name)
    try:
        cmd = [sys.executable, "-m", "v5_shin.scripts.train_ppo", "--mode", "privileged", "--name", name,
               "--hold-vz-down", "5", "--total-steps", "64", "--device", "cpu", "--renderer", "tiny"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                             cwd=os.path.join(os.path.dirname(__file__), "..", ".."))
        assert res.returncode != 0 and "hold-vz-down" in res.stderr
    finally:
        shutil.rmtree(run, ignore_errors=True)


# ---------------------------------------------------------------- D24 friction
def test_effective_drone_pad_friction_is_half():
    import pybullet as pb
    from v5_shin.envs.landing_sim import DRONE_FRICTION, PAD_FRICTION
    sim = LandingSim(renderer="tiny", marker=False, dr=DRConfig.off(), seed=0)
    try:
        mu_pad = pb.getDynamicsInfo(sim.plat.body, -1, physicsClientId=sim.cid)[1]
        mu_drone = pb.getDynamicsInfo(sim.quad.body, -1, physicsClientId=sim.cid)[1]
        assert (mu_pad, mu_drone) == (PAD_FRICTION, DRONE_FRICTION)
        assert mu_pad * mu_drone == pytest.approx(0.5)
        sim.reseed(123)                                   # survives per-episode reseeding
        sim.reset(c=0.0, spawn=dict(HIGH))
        assert pb.getDynamicsInfo(sim.quad.body, -1, physicsClientId=sim.cid)[1] == DRONE_FRICTION
    finally:
        sim.close()
