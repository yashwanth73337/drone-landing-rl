"""Block 13: active-perception reward r_active (SPEC §11.3), hand-computed on recorded rollouts.

Run:  python -m pytest v5_shin/tests/test_block13_active.py -v -s
"""
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

import v5_shin.envs.landing_sim as LS
from v5_shin.policies.active_perception import (ALPHA, BETA, TAU, ActivePerceptionReward,
                                                active_perception_reward, est_loss_per_step)
from v5_shin.policies.ppo import PPOConfig, PPOTrainer, compute_gae
from v5_shin.scripts.train_ppo import make_venv


def test_paper_constants():
    assert (ALPHA, BETA, TAU) == (0.1, 1.0, 0.01)


def test_formula_hand_cases():
    l_next = torch.tensor([[0.0, 0.005, 0.01, 0.06, 0.51, 1.01, 5.0, 0.3]])
    done = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1.0]])
    r = active_perception_reward(l_next, done)
    # -0.1 * clip(L - 0.01, 0, 1); zero below tau, saturates at -0.1; zero on a done step
    exp = torch.tensor([[0.0, 0.0, 0.0, -0.005, -0.05, -0.1, -0.1, 0.0]])
    assert torch.allclose(r, exp, atol=1e-7)
    s = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 3.0])
    assert est_loss_per_step(s, torch.zeros(6)).item() == pytest.approx((1 + 4 + 9) / 6)


def trainer(n_steps=8, seq_len=4, n_envs=2, fn=True):
    venv = make_venv("vision", n_envs, seed=3, renderer="tiny", vz_penalty="literal", c=0.125, sync=True)
    cfg = PPOConfig(mode="vision", n_steps=n_steps, seq_len=seq_len, epochs=1, minibatches=1)
    tr = PPOTrainer(venv, cfg, device="cpu", total_updates=10,
                    r_active_fn=ActivePerceptionReward() if fn else None)
    tr.reset(seeds=range(n_envs))
    return tr


def test_recorded_rollout_hand_check_and_gae(monkeypatch):
    """Recompute r_active independently from the recorded buffer, across an episode boundary
    (horizon forced to 5), and check GAE used rewards + r_active."""
    monkeypatch.setattr(LS, "HORIZON", 5)
    torch.manual_seed(0)
    tr = trainer(n_steps=8, seq_len=4)
    buf, _ = tr.collect()
    T, B = buf["rewards"].shape
    done = torch.clamp(buf["terminated"] + buf["truncated"], max=1.0)
    assert done.sum() >= B                              # every env finished at least once
    for t in range(T):
        for b in range(B):
            if t + 1 < T:
                s, y = buf["s_est"][t + 1, b], buf["target"][t + 1, b]
            else:
                s, y = buf["s_est_next_last"][b], buf["target_next_last"][b]
            L = float(((s - y) ** 2).sum() / 6)
            exp = 0.0 if done[t, b] else -0.1 * min(max(L - 0.01, 0.0), 1.0)
            assert float(buf["r_active"][t, b]) == pytest.approx(exp, abs=1e-6), (t, b)
    assert (buf["r_active"][done > 0] == 0).all()
    adv, ret = compute_gae(buf["rewards"] + buf["r_active"], buf["values"], buf["next_values"],
                           buf["terminated"], buf["truncated"], tr.cfg.gamma, tr.cfg.lam)
    assert torch.allclose(adv, buf["advantages"]) and torch.allclose(ret, buf["returns"])
    print(f"\n[r_active] mean {buf['r_active'].mean():+.4f}, active fraction {buf['r_active_frac']:.2f}, "
          f"L_est(t+1) median {buf['l_est_next'].median():.3f}")
    tr.venv.close()


def test_last_step_estimate_matches_next_rollout():
    """s_est_next_last (computed without advancing the LSTM) equals the estimate the policy
    actually produces for that observation at the start of the next collect()."""
    torch.manual_seed(1)
    tr = trainer(n_steps=4, seq_len=4)
    b1, _ = tr.collect()
    b2, _ = tr.collect()
    assert torch.allclose(b1["s_est_next_last"], b2["s_est"][0], atol=1e-6)
    assert torch.allclose(b1["target_next_last"], b2["target"][0])
    tr.venv.close()


def test_off_means_zero_and_policy_P_unaffected():
    torch.manual_seed(2)
    tr = trainer(n_steps=4, seq_len=4, fn=False)
    b, _ = tr.collect()
    assert (b["r_active"] == 0).all()
    tr.venv.close()


@pytest.mark.parametrize("mode,flag,expect", [("vision", "auto", True), ("vision", "off", False),
                                              ("privileged", "auto", False)])
def test_train_cli_r_active(mode, flag, expect):
    name = f"_pytest_ra_{mode}_{flag}"
    run = os.path.join(os.path.dirname(__file__), "..", "runs", name)
    shutil.rmtree(run, ignore_errors=True)
    try:
        cmd = [sys.executable, "-m", "v5_shin.scripts.train_ppo", "--mode", mode, "--name", name,
               "--total-steps", "32", "--n-envs", "2", "--n-steps", "16", "--seq-len", "4",
               "--minibatches", "2", "--renderer", "tiny", "--device", "cpu", "--r-active", flag]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                             cwd=os.path.join(os.path.dirname(__file__), "..", ".."))
        assert res.returncode == 0, res.stderr[-3000:]
        assert json.load(open(os.path.join(run, "config.json")))["r_active_used"] is expect
        import csv
        row = list(csv.DictReader(open(os.path.join(run, "updates.csv"))))[-1]
        if expect:
            assert row["r_active_frac"] != "" and float(row["r_active_mean"]) <= 0
            assert " ra " in res.stdout
        else:
            assert row["r_active_frac"] == "" and float(row["r_active_mean"]) == 0.0
    finally:
        shutil.rmtree(run, ignore_errors=True)
