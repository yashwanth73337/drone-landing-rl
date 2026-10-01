"""Motor time-constant diagnostic (evaluation only; SPEC §15, 1 Oct 2026).

Checks that (1) the default is unchanged, (2) each mode sets the time constants it claims,
(3) the RNG stream (hence every pinned episode) is identical across modes, and (4) the
evaluation flag works end-to-end without touching the default output folder name.

Run:  python -m pytest v5_shin/tests/test_diag_motor.py -v -s
"""
import os
import shutil

import numpy as np
import pybullet as p
import pytest

from v5_shin.envs import lmf2_params as P
from v5_shin.envs.quad import MOTOR_MODES, LMF2Quad, MotorModel, make_client
from v5_shin.envs.shin_env import ShinLandingEnv
from v5_shin.scripts.evaluate import evaluate
from v5_shin.tests.test_block14_evaluate import make_run


def test_mode_time_constants_and_rng_stream():
    after = []
    for mode in MOTOR_MODES:
        rng = np.random.default_rng(7)
        m = MotorModel(rng, mode=mode)
        after.append(rng.random())                 # next draw: must not depend on the mode
        if mode == "asym":
            ref = m.tau_inc.copy()
            assert np.all((m.tau_inc >= 0.05) & (m.tau_inc <= 0.08)) and np.all(m.tau_dec == P.TAU_DEC)
        elif mode == "sym_slow":
            assert np.array_equal(m.tau_inc, ref) and np.array_equal(m.tau_dec, ref)
        else:
            assert np.all(m.tau_inc == P.TAU_DEC) and np.all(m.tau_dec == P.TAU_DEC)
    assert after[0] == after[1] == after[2]
    with pytest.raises(ValueError):
        MotorModel(np.random.default_rng(0), mode="bogus")


def test_default_is_asym_everywhere():
    env = ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=0)
    try:
        assert env.sim.motor_mode == "asym" and env.sim.quad.motors.mode == "asym"
        env.sim.reseed(9000)                       # evaluation path re-creates the motors
        assert env.sim.quad.motors.mode == "asym"
    finally:
        env.close()
    env = ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=0, motor_mode="sym_fast")
    try:
        env.sim.reseed(9000)
        assert env.sim.quad.motors.mode == "sym_fast" and np.all(env.sim.quad.motors.tau_inc == P.TAU_DEC)
    finally:
        env.close()


def _jitter_sink(mode, seed=0):
    rng = np.random.default_rng(seed)
    cid = make_client()
    try:
        quad = LMF2Quad(cid, rng, P.sample_gains(rng, "nominal"), motor_mode=mode)
        quad.reset_pose([0, 0, 200.0])
        z0, v = quad.state()["pos"][2], np.zeros(2)
        for _ in range(150):
            v = 0.7 * v + 0.3 * rng.normal(0, 1.0, 2) * 3
            for _ in range(P.SUBSTEPS):
                quad.apply_control([v[0], v[1], 0.0, 0.0])
                p.stepSimulation(physicsClientId=cid)
        return (quad.state()["pos"][2] - z0) / 15.0
    finally:
        p.disconnect(cid)


def test_jitter_sink_by_mode():
    """Same protocol as test_block1 jitter test. Scratch study (5 seeds): asym -0.73,
    sym_slow -0.20, sym_fast -0.06 m/s."""
    r = {m: _jitter_sink(m) for m in MOTOR_MODES}
    print(f"\n[motor diag] jitter sink m/s: " + ", ".join(f"{m} {v:+.3f}" for m, v in r.items()))
    assert r["asym"] < -0.4 and r["sym_slow"] > -0.35 and r["sym_fast"] > -0.15


def test_evaluate_motor_flag_same_episodes():
    run = make_run("_pytest_eval_motor", "privileged")
    try:
        sa, ra = evaluate("_pytest_eval_motor", "c.pt", episodes=4, c=1.0, workers=1,
                          out=os.path.join(run, "a"))
        sd, rd = evaluate("_pytest_eval_motor", "c.pt", episodes=4, c=1.0, workers=1,
                          out=os.path.join(run, "d"), motor="asym")
        sf, rf = evaluate("_pytest_eval_motor", "c.pt", episodes=4, c=1.0, workers=1,
                          out=os.path.join(run, "f"), motor="sym_fast")
        assert sa["motor"] == "asym" and sf["motor"] == "sym_fast"
        assert [(x["outcome"], x["steps"], x["ret"]) for x in ra] == \
               [(x["outcome"], x["steps"], x["ret"]) for x in rd]          # default == explicit asym
        assert [x["seed"] for x in rf] == [x["seed"] for x in ra]
        assert [x["d0"] for x in rf] == [x["d0"] for x in ra]              # same spawns
        assert any(not np.isclose(x["ret"], y["ret"]) for x, y in zip(ra, rf))   # dynamics differ
        for s in (sa, sf):
            assert set(s["safe_success"]) == {"<=0.5", "<=1.0", "<=1.5", "<=2.0"}
        assert "cmd_vz_last5" in ra[0] and abs(ra[0]["cmd_vz_last"]) <= 1.0 + 1e-9
        # default folder name has no motor suffix; the CLI tag adds one for non-default modes
        s2, _ = evaluate("_pytest_eval_motor", "c.pt", episodes=1, c=1.0, workers=1, motor="sym_slow")
        assert os.path.isdir(os.path.join(run, "eval_c_c1.0_s9000_det_sym_slow"))
    finally:
        shutil.rmtree(run, ignore_errors=True)
