"""Block 17: hold practice (on-pad starts) and reward v4 (SPEC D22). No training runs.

Run:  python -m pytest v5_shin/tests/test_block17_hold_practice.py -v
"""
import numpy as np
import pybullet as p
import pytest

from v5_shin.envs import landing_sim as LS
from v5_shin.envs.dr import DRConfig
from v5_shin.envs.landing_sim import HOLD_STEPS, PRACTICE_VZ, PRACTICE_XY, LandingSim
from v5_shin.envs.platform import PAD_SIZE, PAD_TOP_Z
from v5_shin.envs.reward import (R_SUCCESS, R_HOLD_FAIL_V4, TouchdownReward, centre_bonus,
                                 hard_landing_reward, hold_step_reward)
from v5_shin.envs.shin_env import ACTION_SCALE, ShinLandingEnv
from v5_shin.tests.test_block15_hold import SPAWN, place


# ---------------------------------------------------------------- RNG streams
def test_first_five_rng_streams_unchanged():
    """Adding the practice stream must not change any existing stream of a pinned seed."""
    sim = LandingSim(renderer="tiny", marker=False, dr=DRConfig.off(), seed=0)
    try:
        for seed in (0, 7, 9001):
            sim._seed_streams(seed)
            got = [g.random() for g in (sim.rng, sim.rng_gains, sim.rng_dr, sim.rng_vis, sim.rng_noise)]
            old = [np.random.default_rng(c).random() for c in np.random.SeedSequence(seed).spawn(5)]
            assert got == old
    finally:
        sim.close()


# ---------------------------------------------------------------- on-pad start
def make_sim(c=0.0):
    sim = LandingSim(renderer="tiny", marker=False, dr=DRConfig.off(), seed=3, hold_steps=HOLD_STEPS)
    sim.reset(c=c, on_pad=True)
    return sim


def test_on_pad_needs_hold():
    sim = LandingSim(renderer="tiny", marker=False, dr=DRConfig.off(), seed=0, hold_steps=0)
    try:
        with pytest.raises(AssertionError):
            sim.reset(c=0.0, on_pad=True)
    finally:
        sim.close()


@pytest.mark.parametrize("c", [0.0, 1.0])
def test_on_pad_start_state(c):
    sim = make_sim(c)
    try:
        assert sim.phase == "hold" and sim.practice and sim.hold_t == 0
        s = sim.quad.state()
        rel = s["pos"] - sim.plat.pad_center()
        cs, sn = np.cos(sim.plat.psi), np.sin(sim.plat.psi)
        rel_pad = np.array([cs * rel[0] + sn * rel[1], -sn * rel[0] + cs * rel[1]])
        assert np.all(np.abs(rel_pad) <= PRACTICE_XY + 1e-9)
        assert 0 < rel[2] < 0.5                                   # just above the pad top
        assert sim.hold_vrel() == pytest.approx(PRACTICE_VZ, abs=1e-6)   # pad velocity + soft descent
        assert np.allclose(sim._td["rel_vel"], [0.0, 0.0, -PRACTICE_VZ])
        # lowest point at the pad top (exact distance, not the bounding box)
        gap = min(c[8] for c in p.getClosestPoints(sim.quad.body, sim.plat.body, distance=1.0,
                                                   physicsClientId=sim.cid))
        assert gap == pytest.approx(LS.PRACTICE_GAP, abs=1e-3)
    finally:
        sim.close()


def test_on_pad_static_hold_succeeds():
    sim = make_sim(0.0)
    try:
        for _ in range(HOLD_STEPS + 2):
            te, tr, info = sim.step(np.array([0, 0, -0.3, 0], float))
            if te or tr:
                break
        assert info["outcome"] == "success" and info["practice"], info
        assert info["hold_steps_done"] == HOLD_STEPS and info["t"] == HOLD_STEPS
    finally:
        sim.close()


def test_normal_reset_is_not_practice():
    sim = LandingSim(renderer="tiny", marker=False, dr=DRConfig.off(), seed=0, hold_steps=HOLD_STEPS)
    try:
        sim.reset(c=0.0, spawn=dict(SPAWN))
        assert sim.phase == "flight" and not sim.practice
        te, tr, info = sim.step(np.zeros(4))
        assert info["practice"] is False
    finally:
        sim.close()


# ---------------------------------------------------------------- env: practice share
def env(hold_practice, variant="v4"):
    return ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=0, dr=DRConfig.off(),
                          gains_mode="nominal", reward_fn=TouchdownReward(variant),
                          hold_steps=HOLD_STEPS, hold_practice=hold_practice)


def test_practice_share():
    e = env(0.25)
    try:
        n = sum(e.reset(seed=s, options={"c": 0.5})[1]["practice"] for s in range(400))
        assert 60 <= n <= 140                                     # binomial(400, 0.25): mean 100
    finally:
        e.close()


def test_no_practice_by_default():
    e = env(0.0)
    try:
        assert not any(e.reset(seed=s, options={"c": 0.5})[1]["practice"] for s in range(100))
    finally:
        e.close()


def test_practice_needs_hold():
    with pytest.raises(AssertionError):
        ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=0, hold_practice=0.25)


# ---------------------------------------------------------------- reward v4
@pytest.mark.parametrize("vz,expect", [(-1.2, -6.0), (-1.5, -7.5), (-2.0, -10.0), (-4.0, -10.0)])
def test_v4_hard_landing(vz, expect):
    assert hard_landing_reward("v4", vz) == pytest.approx(expect)
    assert hard_landing_reward("v4", vz) <= R_HOLD_FAIL_V4      # worse than any failed hold


@pytest.mark.parametrize("xy,expect", [((0, 0), 2.0), ((0.375, 0.1), 1.0), ((0.1, -0.375), 1.0),
                                       ((0.75, 0), 0.0), ((1.0, 0.2), 0.0)])
def test_centre_bonus(xy, expect):
    assert centre_bonus(xy) == pytest.approx(expect)


@pytest.mark.parametrize("vrel,expect", [(0.0, 0.5), (0.5, 0.25), (1.0, 0.0), (3.0, 0.0)])
def test_v4_hold_step(vrel, expect):
    assert hold_step_reward("v4", vrel) == pytest.approx(expect)


def step(e, cmd):
    _, r, te, tr, info = e.step(np.asarray(cmd, float) / ACTION_SCALE)
    return r, te, tr, info


def test_v4_soft_landing_rewards():
    e = env(0.0)
    try:
        e.reset(seed=0, options={"c": 0.0, "spawn": dict(SPAWN)})
        place(e.sim, 0.02, -0.3)
        e.reward_fn.reset(e)
        while True:
            r, te, tr, info = step(e, [0, 0, -0.3, 0])
            assert not (te or tr), info["outcome"]
            if e.sim.phase == "hold":
                assert r == pytest.approx(centre_bonus(e.sim._td["rel_pos_pad"]))
                assert r > 1.9                                    # landed at the centre
                break
        while True:
            r, te, tr, info = step(e, [0, 0, -0.3, 0])
            if te or tr:
                break
            assert r == pytest.approx(hold_step_reward("v4", e.sim._vrel_hist[-1]))
            assert 0.0 <= r <= 0.5
        assert info["outcome"] == "success" and r == R_SUCCESS
    finally:
        e.close()


def test_v4_failed_hold_costs_five():
    e = env(0.0)
    try:
        e.reset(seed=0, options={"c": 0.0, "on_pad": True})
        for _ in range(5):
            r, te, tr, info = step(e, [0, 0, 1.0, 0])             # climb away: bounce
            if te or tr:
                break
        assert info["outcome"] == "bounce" and info["practice"] and r == R_HOLD_FAIL_V4
    finally:
        e.close()


def test_practice_first_step_is_a_hold_step():
    """A practice episode gets no touchdown bonus: its first step is already a hold step."""
    e = env(0.0)
    try:
        e.reset(seed=0, options={"c": 0.0, "on_pad": True})
        r, te, tr, info = step(e, [0, 0, -0.3, 0])
        assert not te and info["hold_t"] == 1
        assert r == pytest.approx(hold_step_reward("v4", e.sim._vrel_hist[-1]))
    finally:
        e.close()


# ---------------------------------------------------------------- CLI
def _run_cli(name, extra):
    import os, subprocess, sys
    cmd = [sys.executable, "-m", "v5_shin.scripts.train_ppo", "--mode", "privileged", "--name", name,
           "--total-steps", "64", "--n-envs", "2", "--n-steps", "16", "--seq-len", "4",
           "--minibatches", "2", "--renderer", "tiny", "--device", "cpu", "--ckpt-every", "1"] + extra
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=os.path.join(os.path.dirname(__file__), "..", ".."))


def test_cli_rejects_practice_without_hold():
    import os, shutil
    name = "_pytest_practice_bad"
    run = os.path.join(os.path.dirname(__file__), "..", "runs", name)
    try:
        res = _run_cli(name, ["--hold-practice", "0.25"])
        assert res.returncode != 0 and "hold-practice" in res.stderr
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_train_script_v4_practice_end_to_end():
    import csv, json, os, shutil
    name = "_pytest_v4_practice"
    run = os.path.join(os.path.dirname(__file__), "..", "runs", name)
    shutil.rmtree(run, ignore_errors=True)
    try:
        res = _run_cli(name, ["--reward", "v4", "--hold-practice", "0.5"])
        assert res.returncode == 0, res.stderr[-3000:]
        rows = list(csv.DictReader(open(os.path.join(run, "updates.csv"))))
        assert len(rows) == 2 and "n_practice" in rows[0] and "n_practice_success" in rows[0]
        eps = list(csv.DictReader(open(os.path.join(run, "episodes.csv"))))
        assert "practice" in eps[0]
        cfg = json.load(open(os.path.join(run, "config.json")))
        assert cfg["reward"] == "v4" and cfg["hold_practice"] == 0.5
        assert cfg["hold_steps"] == HOLD_STEPS
    finally:
        shutil.rmtree(run, ignore_errors=True)
