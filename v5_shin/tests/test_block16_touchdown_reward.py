"""Block 16: touchdown-aware reward (SPEC D21) on the D20 hold task. No training runs.

Run:  python -m pytest v5_shin/tests/test_block16_touchdown_reward.py -v
"""
import numpy as np
import pytest

from v5_shin.envs.dr import DRConfig
from v5_shin.envs.landing_sim import HOLD_STEPS, V_IMP
from v5_shin.envs.reward import (R_CRASH, R_SUCCESS, TouchdownReward, flare_term, geometry,
                                 hard_landing_reward)
from v5_shin.envs.shin_env import ACTION_SCALE, ShinLandingEnv
from v5_shin.tests.test_block15_hold import SPAWN, place


# ---------------------------------------------------------------- pure functions
@pytest.mark.parametrize("d,dz,vz,expect", [
    (0.5, -1.0, -1.0, 0.0),            # v_ref = 0.8 + 1.0 = 1.8: slower than allowed
    (0.5, -1.0, -2.3, -0.5),           # 0.5 m/s too fast
    (0.5, -0.1, -3.0, -1.0),           # v_ref 0.9, 2.1 too fast: clipped at -1
    (0.5, 0.2, -1.0, -0.2),            # below the pad top: v_ref = 0.8
    (2.0, -0.1, -3.0, 0.0),            # outside 1.5 m: no flare
    (0.5, -0.5, 0.5, 0.0),             # climbing
])
def test_flare_term_hand_computed(d, dz, vz, expect):
    assert flare_term(d, dz, vz) == pytest.approx(expect, abs=1e-12)


@pytest.mark.parametrize("variant,vz,expect", [
    ("v1", -3.0, R_CRASH), ("v3", -1.2, R_CRASH),
    ("v2", -1.2, -1.0), ("v2", -2.0, -5.0), ("v2", -3.0, -10.0), ("v2", -4.5, -10.0),
])
def test_hard_landing_reward(variant, vz, expect):
    assert hard_landing_reward(variant, vz) == pytest.approx(expect)


# ---------------------------------------------------------------- env integration
def make(variant):
    env = ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=0, dr=DRConfig.off(),
                         gains_mode="nominal", reward_fn=TouchdownReward(variant),
                         hold_steps=HOLD_STEPS)
    env.reset(seed=0, options={"c": 0.0, "spawn": dict(SPAWN)})
    return env


def put(env, gap, vz):
    place(env.sim, gap, vz)
    env.reward_fn.reset(env)


def step(env, cmd):
    _, r, te, tr, info = env.step(np.asarray(cmd, float) / ACTION_SCALE)
    return r, te, tr, info


@pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
def test_soft_landing_rewards(variant):
    """touchdown step 0; each hold step -min(1, |v_rel|); hold success +10."""
    env = make(variant)
    try:
        put(env, 0.02, -0.3)
        while True:                                   # flight until the touchdown step
            r, te, tr, info = step(env, [0, 0, -0.3, 0])
            assert not (te or tr), info["outcome"]
            if env.sim.phase == "hold":
                assert r == 0.0 and info["hold_t"] == 0
                break
        while True:
            r, te, tr, info = step(env, [0, 0, -0.3, 0])
            if te or tr:
                break
            assert r == pytest.approx(-min(1.0, env.sim._vrel_hist[-1]), abs=1e-12)
            assert info["reward_terms"] == dict(hold=r)
        assert info["outcome"] == "success" and r == R_SUCCESS
    finally:
        env.close()


@pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
def test_hard_landing_rewards(variant):
    env = make(variant)
    try:
        put(env, 0.3, -4.0)
        while True:
            r, te, tr, info = step(env, [0, 0, -1.0, 0])
            if te or tr:
                break
        assert info["outcome"] == "hard_landing" and abs(info["rel_vel"][2]) > V_IMP
        assert r == pytest.approx(hard_landing_reward(variant, info["rel_vel"][2]))
    finally:
        env.close()


def test_bounce_reward():
    env = make("v1")
    try:
        put(env, 0.02, -0.3)
        while env.sim.phase != "hold":
            step(env, [0, 0, -0.3, 0])
        for _ in range(5):                            # climb away until the hold fails
            r, te, tr, info = step(env, [0, 0, 1.0, 0])
            if te or tr:
                break
        assert te and info["outcome"] == "bounce" and r == R_CRASH
    finally:
        env.close()


def test_v3_flight_step_includes_flare():
    env = make("v3")
    try:
        put(env, 1.0, -3.5)                         # 1 m above the pad, diving at 3.5 m/s
        info = dict(outcome=None, action_cmd=np.zeros(4))
        r = env.reward_fn(env, info)                # reward of this state, no physics step
        d, dz, vz = geometry(env.sim)
        assert info["reward_terms"]["flare"] == pytest.approx(flare_term(d, dz, vz), abs=1e-12)
        assert info["reward_terms"]["flare"] < 0
        assert r == pytest.approx(sum(info["reward_terms"].values()), abs=1e-12)
    finally:
        env.close()


def test_v1_flight_step_has_no_flare():
    env = make("v1")
    try:
        put(env, 1.5, -2.5)
        r, te, tr, info = step(env, [0, 0, -1.0, 0])
        assert "flare" not in info["reward_terms"]
        assert r == pytest.approx(sum(info["reward_terms"].values()), abs=1e-12)
    finally:
        env.close()


def test_train_script_reward_v3_end_to_end():
    """The CLI with --reward v3: 2 updates, privileged, CPU. Checks the hold columns and config."""
    import csv, json, os, shutil, subprocess, sys
    name = "_pytest_reward_v3"
    run = os.path.join(os.path.dirname(__file__), "..", "runs", name)
    shutil.rmtree(run, ignore_errors=True)
    try:
        cmd = [sys.executable, "-m", "v5_shin.scripts.train_ppo", "--mode", "privileged",
               "--name", name, "--reward", "v3", "--total-steps", "64", "--n-envs", "2",
               "--n-steps", "16", "--seq-len", "4", "--minibatches", "2", "--renderer", "tiny",
               "--device", "cpu", "--ckpt-every", "1"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                             cwd=os.path.join(os.path.dirname(__file__), "..", ".."))
        assert res.returncode == 0, res.stderr[-3000:]
        rows = list(csv.DictReader(open(os.path.join(run, "updates.csv"))))
        assert len(rows) == 2
        for k in ("n_hard_landing", "n_bounce", "n_slide_off", "n_tip_over", "n_not_still",
                  "n_touchdown", "td_impact_vz_mean"):
            assert k in rows[0], k
        cfg = json.load(open(os.path.join(run, "config.json")))
        assert cfg["reward"] == "v3" and cfg["hold_steps"] == HOLD_STEPS
    finally:
        shutil.rmtree(run, ignore_errors=True)
