"""Block 14: pinned evaluation. Determinism across worker counts, file outputs, vision metrics.

Run:  python -m pytest v5_shin/tests/test_block14_evaluate.py -v -s
"""
import json
import os
import shutil

import numpy as np
import pytest
import torch

from v5_shin.policies.shin_policy import ShinPolicy
from v5_shin.scripts.evaluate import RUNS, evaluate, wilson


def make_run(name, mode, vz_max=1.0):
    run = os.path.join(RUNS, name)
    shutil.rmtree(run, ignore_errors=True)
    os.makedirs(run)
    torch.manual_seed(0)
    pol = ShinPolicy(mode)
    torch.save(dict(policy=pol.state_dict(), update=3, global_step=12288), os.path.join(run, "c.pt"))
    json.dump(dict(mode=mode, vz_max=vz_max, vz_penalty="literal"), open(os.path.join(run, "config.json"), "w"))
    return run


def test_wilson():
    lo, hi = wilson(90, 100)
    assert 0.82 < lo < 0.83 and 0.94 < hi < 0.95


def test_deterministic_and_worker_independent():
    run = make_run("_pytest_eval_p", "privileged")
    try:
        s1, r1 = evaluate("_pytest_eval_p", "c.pt", episodes=6, c=1.0, workers=1, out=os.path.join(run, "a"))
        s3, r3 = evaluate("_pytest_eval_p", "c.pt", episodes=6, c=1.0, workers=3, out=os.path.join(run, "b"))
        assert [r["seed"] for r in r1] == list(range(9000, 9006))
        for a, b in zip(r1, r3):
            assert a["outcome"] == b["outcome"] and a["steps"] == b["steps"]
            assert np.isclose(a["ret"], b["ret"])
        assert s1["timestep"] == 12288 and s1["vz_max"] == 1.0
        assert os.path.exists(os.path.join(run, "a", "episodes.csv"))
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_vision_metrics_present():
    run = make_run("_pytest_eval_v", "vision")
    try:
        s, rows = evaluate("_pytest_eval_v", "c.pt", episodes=2, c=0.125, workers=1,
                           out=os.path.join(run, "a"))
        assert "est_pos_rmse_m" in s and np.isfinite(s["est_pos_rmse_m"])
        bins = s["est_pos_err_by_blind_age"]
        assert sum(v["n"] for v in bins.values()) == sum(r["steps"] for r in rows)
    finally:
        shutil.rmtree(run, ignore_errors=True)
