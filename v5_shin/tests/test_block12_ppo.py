"""Block 12: recurrent PPO + L_est. CPU unit tests only (no training runs).

Run:  python -m pytest v5_shin/tests/test_block12_ppo.py -v -s
"""
import copy

import numpy as np
import pytest
import torch

import v5_shin.envs.landing_sim as LS
from v5_shin.policies.ppo import PPOConfig, PPOTrainer, compute_gae
from v5_shin.scripts.train_ppo import make_venv


# ---------------------------------------------------------------- GAE
def test_gae_hand_computed():
    """B = 1, T = 4, gamma = 0.9, lam = 0.5. Step 1 is TRUNCATED (bootstrap with V_final = 10),
    step 3 is TERMINATED. Hand computation below."""
    g, l = 0.9, 0.5
    r = torch.tensor([[1.0], [2.0], [0.0], [5.0]])
    v = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    nv = torch.tensor([[2.0], [10.0], [4.0], [999.0]])   # t=0: V(s1); t=1: V(final); t=2: V(s3); t=3 masked
    te = torch.tensor([[0.0], [0.0], [0.0], [1.0]])
    tr = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    adv, ret = compute_gae(r, v, nv, te, tr, g, l)
    d3 = 5.0 - 4.0                                       # terminal: no bootstrap
    d2 = 0.0 + g * 4.0 - 3.0
    a3, a2 = d3, d2 + g * l * d3
    d1 = 2.0 + g * 10.0 - 2.0                            # truncated: bootstrap, chain cut
    a1 = d1
    d0 = 1.0 + g * 2.0 - 1.0
    a0 = d0 + g * l * a1
    assert torch.allclose(adv[:, 0], torch.tensor([a0, a1, a2, a3]))
    assert torch.allclose(ret, adv + v)


# ---------------------------------------------------------------- trainer on real envs
def small_trainer(mode="vision", n_envs=2, n_steps=8, seq_len=4, **cfg_kw):
    venv = make_venv(mode, n_envs, seed=3, renderer="tiny", vz_penalty="literal", c=0.125, sync=True)
    cfg = PPOConfig(mode=mode, n_steps=n_steps, seq_len=seq_len, epochs=1, minibatches=1, **cfg_kw)
    tr = PPOTrainer(venv, cfg, device="cpu", total_updates=10)
    tr.reset(seeds=range(n_envs))
    return tr


def test_rollout_logprobs_reproduced_by_chunked_evaluate():
    """The key recurrent-PPO check: with stored chunk-start LSTM states and episode starts,
    re-evaluating the rollout in seq_len chunks reproduces the rollout-time log-probs,
    values and estimates (ratio == 1 before any update)."""
    torch.manual_seed(0)
    tr = small_trainer()
    buf, _ = tr.collect()
    chunks = [(k, b) for k in range(2) for b in range(2)]
    obs, rest, state = tr._gather(buf, chunks)
    with torch.no_grad():
        logp, _, v, s_est = tr.policy.evaluate(obs, rest["actions"], state, rest["starts"])
    assert torch.allclose(logp, rest["logp"], atol=1e-4)
    assert torch.allclose(v, rest["values"], atol=1e-5)
    ks = torch.tensor([k for k, _ in chunks]); bs = torch.tensor([b for _, b in chunks])
    tidx = ks[None] * 4 + torch.arange(4)[:, None]
    assert torch.allclose(s_est, buf["s_est"][tidx, bs[None]], atol=1e-5)
    tr.venv.close()


def test_truncation_bootstrap_uses_final_obs(monkeypatch):
    monkeypatch.setattr(LS, "HORIZON", 5)            # force a timeout at t = 4 (same process)
    torch.manual_seed(0)
    tr = small_trainer(mode="privileged", n_steps=8, seq_len=4)
    tr.policy.critic[-1].bias.data.fill_(0.3)        # make values non-trivial
    buf, eps = tr.collect()
    t_idx, b_idx = torch.nonzero(buf["truncated"], as_tuple=True)
    assert len(t_idx) > 0 and all(e["outcome"] in ("timeout", "crash_ground", "tilt", "drift",
                                                   "crash_platform", "success") for e in eps)
    for t, b in zip(t_idx.tolist(), b_idx.tolist()):
        nv = buf["next_values"][t, b]
        assert abs(float(nv)) > 0                      # bootstrapped, not zero
        if t + 1 < buf["values"].shape[0]:
            assert not torch.isclose(nv, buf["values"][t + 1, b])   # not the reset obs' value
            assert buf["starts"][t + 1, b] == 1
    te = buf["terminated"] > 0
    assert torch.all(buf["next_values"][te] == 0)
    tr.venv.close()


def test_update_runs_and_first_ratio_is_one():
    torch.manual_seed(0)
    tr = small_trainer()
    buf, _ = tr.collect()
    before = copy.deepcopy(tr.policy.state_dict())
    stats = tr.learn(buf)
    assert abs(stats["kl_first"]) < 1e-6 and np.isfinite(stats["est"]) and np.isfinite(stats["v"])
    changed = [k for k in before if not torch.equal(before[k], tr.policy.state_dict()[k])]
    assert changed and tr.update == 1
    tr.venv.close()


def test_gradient_accumulation_is_exact():
    """One optimizer step with the minibatch split into micro-batches equals the unsplit step."""
    torch.manual_seed(0)
    tr = small_trainer()
    buf, _ = tr.collect()
    chunks = [(k, b) for k in range(2) for b in range(2)]
    a = copy.deepcopy(tr)
    a.venv = None
    b = copy.deepcopy(a)
    a.cfg.micro_chunks, b.cfg.micro_chunks = None, 1
    a.update_step(buf, chunks)
    b.update_step(buf, chunks)
    for (k, pa), (_, pb) in zip(a.policy.named_parameters(), b.policy.named_parameters()):
        assert torch.allclose(pa, pb, atol=1e-6), k
    tr.venv.close()


def test_estimation_loss_alone_learns():
    torch.manual_seed(0)
    tr = small_trainer(n_steps=16, seq_len=4)
    buf, _ = tr.collect()
    chunks = [(k, b) for k in range(4) for b in range(2)]
    obs, rest, state = tr._gather(buf, chunks)
    opt = torch.optim.Adam(tr.policy.parameters(), lr=1e-3)
    losses = []
    for _ in range(40):
        _, _, _, s_est = tr.policy.evaluate(obs, rest["actions"], state, rest["starts"])
        loss = ((s_est - obs["target"]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"\n[L_est] {losses[0]:.3f} -> {losses[-1]:.3f} in 40 steps")
    assert losses[-1] < 0.2 * losses[0]
    tr.venv.close()


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    tr = small_trainer()
    buf, _ = tr.collect()
    tr.learn(buf)
    path = tmp_path / "c.pt"
    torch.save(tr.state_dict(dict(curriculum={"level": 20})), path)
    tr2 = small_trainer()
    extra = tr2.load_state_dict(torch.load(path, weights_only=False))
    for (k, p1), (_, p2) in zip(tr.policy.named_parameters(), tr2.policy.named_parameters()):
        assert torch.equal(p1, p2), k
    assert tr2.update == 1 and extra["curriculum"]["level"] == 20
    tr.venv.close(); tr2.venv.close()


def test_blind_age_bins_reported():
    torch.manual_seed(0)
    tr = small_trainer(n_steps=16, seq_len=4)
    buf, _ = tr.collect()
    d = tr.estimate_error_by_blind_age(buf)
    assert sum(v for k, v in d.items() if k.startswith("n_blind_")) == 16 * 2
    tr.venv.close()


@pytest.mark.parametrize("mode", ["privileged", "vision"])
def test_train_script_end_to_end(mode):
    """The CLI itself: 2 updates, 2 async envs, tiny renderer, CPU. Checks files and columns."""
    import csv, json, os, shutil, subprocess, sys
    name = f"_pytest_{mode}"
    run = os.path.join(os.path.dirname(__file__), "..", "runs", name)
    shutil.rmtree(run, ignore_errors=True)
    try:
        cmd = [sys.executable, "-m", "v5_shin.scripts.train_ppo", "--mode", mode, "--name", name,
               "--total-steps", "64", "--n-envs", "2", "--n-steps", "16", "--seq-len", "4",
               "--minibatches", "2", "--renderer", "tiny", "--device", "cpu", "--ckpt-every", "1"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                             cwd=os.path.join(os.path.dirname(__file__), "..", ".."))
        assert res.returncode == 0, res.stderr[-3000:]
        rows = list(csv.DictReader(open(os.path.join(run, "updates.csv"))))
        assert len(rows) == 2 and rows[-1]["global_step"] == "64"
        for k in ("n_success", "kl", "explained_var", "level", "impact_vz_mean"):
            assert k in rows[0]
        if mode == "vision":
            assert "est_err_blind_0_0" in rows[0] and float(rows[0]["est"]) > 0
        cfg = json.load(open(os.path.join(run, "config.json")))
        assert cfg["mode"] == mode and cfg["ppo"]["lambda_est"] == (1.0 if mode == "vision" else 0.0)
        assert os.path.exists(os.path.join(run, "latest.pt")) and os.path.exists(os.path.join(run, "u00002.pt"))
    finally:
        shutil.rmtree(run, ignore_errors=True)
