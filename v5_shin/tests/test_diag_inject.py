"""True-state injection diagnostic (evaluation only; SPEC §15, 3 Oct 2026).

Run:  python -m pytest v5_shin/tests/test_diag_inject.py -v -s
"""
import os
import shutil

import pytest
import torch

from v5_shin.policies.shin_policy import ShinPolicy
from v5_shin.scripts.evaluate import evaluate
from v5_shin.tests.test_block14_evaluate import make_run


def _obs(T=3, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"image": torch.randint(0, 256, (T, B, 160, 256), dtype=torch.uint8, generator=g),
            "u": torch.randn(T, B, 7, generator=g), "target": torch.randn(T, B, 6, generator=g)}


def test_injection_semantics():
    torch.manual_seed(0)
    pol = ShinPolicy("vision").eval()
    obs, st, starts = _obs(), pol.initial_state(2), torch.zeros(3, 2)
    with torch.no_grad():
        mu0, s0, _ = pol.actor_seq(obs, st, starts)
        mu_d, s_d, _ = pol.actor_seq(obs, st, starts, inject_true_state=False)
        assert torch.equal(mu0, mu_d) and torch.equal(s0, s_d)            # default unchanged
        obs_same = dict(obs, target=s0.clone())                            # target := own estimate
        mu_s, _, _ = pol.actor_seq(obs_same, st, starts, inject_true_state=True)
        assert torch.allclose(mu_s, mu0, atol=1e-6)                        # only y[0:6] is replaced
        mu_i, s_i, _ = pol.actor_seq(obs, st, starts, inject_true_state=True)
        assert not torch.allclose(mu_i, mu0)                               # true state used
        assert torch.equal(s_i, s0)                                        # reported estimate unchanged
    assert not any("inject" in k for k in pol.state_dict())               # no new parameters


def test_evaluate_flag():
    run = make_run("_pytest_eval_inj", "vision")
    try:
        s0, r0 = evaluate("_pytest_eval_inj", "c.pt", episodes=2, c=0.125, workers=1, out=os.path.join(run, "a"))
        s1, r1 = evaluate("_pytest_eval_inj", "c.pt", episodes=2, c=0.125, workers=1, inject_true_state=True)
        assert s0["inject_true_state"] is False and s1["inject_true_state"] is True
        assert os.path.isdir(os.path.join(run, "eval_c_c0.125_s9000_det_injtrue"))
        assert [x["d0"] for x in r0] == [x["d0"] for x in r1]                # same episodes
    finally:
        shutil.rmtree(run, ignore_errors=True)
    run = make_run("_pytest_eval_inj_p", "privileged")
    try:
        with pytest.raises(ValueError):
            evaluate("_pytest_eval_inj_p", "c.pt", episodes=1, c=1.0, workers=1, inject_true_state=True)
    finally:
        shutil.rmtree(run, ignore_errors=True)
