"""v6_drop Step 7d: PPO pieces (learn/ppo.py) and the training loop (scripts/train_ppo.py).

Uses UNTRAINED networks (fixed torch seed) and TinyRenderer, so it needs no results/.

Checks:
  1. THE CRITIC CANNOT CHANGE THE LSTM (decided 28 Oct): the value loss gives no gradient
     to the eyes, the LSTM or the heads, and an optimizer step on it leaves them unchanged;
     the critic itself does get gradients; the actor's loss DOES reach the LSTM
  2. GAE = the 7.3 hand example (early drop: -9.76 ... -13.23)
  3. GAE edge cases: timeout -> bootstrap V(final), carry stops; landing -> future 0;
     rollout cut -> last_value; nothing leaks across a flight boundary
  4. log-chance and entropy = torch.distributions (Normal x 4 + Bernoulli)
  5. unroll with an episode-start mask = acting step by step (memory reset at the start)
     = Student.forward on a sequence without starts (same network as Step 6)
  6. the learner recomputes the workers' chances: ratio = 1 at the start of a round
  7. the clip: ratio 1.5 with a positive advantage -> no actor gradient; negative -> gradient
  8. train_ppo end to end (1 worker): critic warm-up rounds leave the actor EXACTLY as
     loaded; after --resume the PPO round changes it; the exam and the guard run;
     checkpoints load with eval_student.load

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s7d_ppo.py -v -s
"""
import csv
import math
import os

import numpy as np
import pytest
import torch

from v6_drop.learn import ppo as PPO
from v6_drop.learn.student import Student
from v6_drop.scripts import eval_student as ES
from v6_drop.scripts import train_ppo as TP

H, Wd = 160, 256


def model(seed=0):
    torch.manual_seed(seed)
    return PPO.ActorCritic(Student()).eval()


def fake_batch(m, L=4, B=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    b = dict(image=torch.randint(0, 255, (L, B, H, Wd), generator=g, dtype=torch.uint8),
             state=torch.randn(L, B, 11, generator=g), priv=torch.randn(L, B, 8, generator=g),
             move_raw=torch.randn(L, B, 4, generator=g) * 0.1, drop=torch.zeros(L, B),
             adv=torch.randn(L, B, generator=g), target=torch.randn(L, B, generator=g) * 5,
             starts=torch.zeros(L, B))
    b["starts"][0] = 1.0
    b["h0"], b["c0"] = m.zero_hc(B)
    with torch.no_grad():
        feats, _ = m.unroll(b["image"], b["state"], (b["h0"], b["c0"]), b["starts"])
        mean, logit = m.heads(feats)
        b["logp_old"] = PPO.log_prob(mean, m.clamped_log_std(), logit, b["move_raw"], b["drop"])
    return b


# ---------------------------------------------------------------- 1. stop-gradient
def test_critic_cannot_change_lstm():
    m = model()
    b = fake_batch(m)
    actor = {k: v.detach().clone() for k, v in m.student.state_dict().items()}
    feats, _ = m.unroll(b["image"], b["state"], (b["h0"], b["c0"]), b["starts"])
    v_loss = 0.5 * (m.value(b["priv"], feats) - b["target"]).pow(2).mean()
    v_loss.backward()
    lstm_grads = [p.grad for p in m.student.parameters()]
    assert all(g is None or float(g.abs().max()) == 0.0 for g in lstm_grads)
    assert m.log_std.grad is None
    assert all(p.grad is not None and float(p.grad.abs().max()) > 0 for p in m.critic.parameters())
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)      # one optimizer over EVERYTHING
    opt.step()
    for k, v in m.student.state_dict().items():
        assert torch.equal(v, actor[k]), k
    # ... while the actor's own loss does reach the LSTM
    m.zero_grad(set_to_none=True)
    loss, st = PPO.ppo_losses(m, b, train_actor=True)
    (loss - 0.0).backward()
    g = m.student.memory.weight_hh_l0.grad
    print(f"\n[stop-gradient] value loss: LSTM grad 0, LSTM unchanged after a step; "
          f"actor loss: LSTM grad max {float(g.abs().max()):.2e}")
    assert g is not None and float(g.abs().max()) > 0
    assert float(m.student.eyes.fc[1].weight.grad.abs().max()) > 0


# ---------------------------------------------------------------- 2-3. GAE
def test_gae_hand_example():
    r = [0.02, 0.02, 0.01, 0.02, 0.01, -0.73]
    v = [12.0, 12.1, 12.2, 12.3, 12.4, 12.5]
    term = [0, 0, 0, 0, 0, 1]
    adv, tgt = PPO.gae(r, v, term, [0] * 6, [0.0] * 6, 99.0)
    print(f"\n[gae] {np.round(adv, 2).tolist()}")
    assert adv == pytest.approx([-9.76, -10.38, -11.04, -11.72, -12.46, -13.23], abs=0.01)
    assert tgt == pytest.approx(adv + np.array(v))


def test_gae_edges():
    g, lam = 0.99, 0.95
    # step 1 is a timeout (bootstrap 5.0), step 3 a landing, then a flight cut by the rollout
    r = [1.0, 2.0, 0.5, 3.0, 0.1]
    v = [4.0, 4.5, 3.0, 3.5, 2.0]
    term = [0, 0, 0, 1, 0]
    trunc = [0, 1, 0, 0, 0]
    boot = [0.0, 5.0, 0.0, 0.0, 0.0]
    last = 7.0
    adv, _ = PPO.gae(r, v, term, trunc, boot, last, g, lam)
    d4 = 0.1 + g * last - 2.0
    d3 = 3.0 - 3.5
    d2 = 0.5 + g * 3.5 - 3.0
    d1 = 2.0 + g * 5.0 - 4.5
    d0 = 1.0 + g * 4.5 - 4.0
    want = [d0 + g * lam * d1, d1, d2 + g * lam * d3, d3, d4]
    assert adv == pytest.approx(want)


# ---------------------------------------------------------------- 4. log-chance
def test_log_prob_and_entropy():
    torch.manual_seed(0)
    mean, log_std = torch.randn(5, 4) * 0.3, torch.log(torch.tensor([0.03, 0.03, 0.05, 0.05]))
    logit, a = torch.randn(5) * 3, mean + torch.randn(5, 4) * 0.05
    d = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    want = (torch.distributions.Normal(mean, log_std.exp()).log_prob(a).sum(-1)
            + torch.distributions.Bernoulli(logits=logit).log_prob(d))
    assert PPO.log_prob(mean, log_std, logit, a, d) == pytest.approx(want.tolist(), abs=1e-5)
    ent = (torch.distributions.Normal(mean, log_std.exp()).entropy().sum(-1)
           + torch.distributions.Bernoulli(logits=logit).entropy())
    assert PPO.entropy(log_std, logit) == pytest.approx(ent.tolist(), abs=1e-5)


# ---------------------------------------------------------------- 5. unroll
def test_unroll_masks_and_matches_student():
    m = model()
    T, B = 8, 2
    g = torch.Generator().manual_seed(3)
    img = torch.randint(0, 255, (T, B, H, Wd), generator=g, dtype=torch.uint8)
    st = torch.randn(T, B, 11, generator=g)
    starts = torch.zeros(T, B)
    starts[0] = 1
    starts[5, 0] = 1                                      # flight 2 starts at step 5 in column 0
    with torch.no_grad():
        feats, _ = m.unroll(img, st, m.zero_hc(B), starts)
        for b in range(B):
            hc = m.zero_hc(1)
            for t in range(T):
                if starts[t, b]:
                    hc = m.zero_hc(1)
                f, hc = m.unroll(img[t:t + 1, b:b + 1], st[t:t + 1, b:b + 1], hc, torch.zeros(1, 1))
                assert torch.allclose(f[0, 0], feats[t, b], atol=1e-5)
        # column 1 has no new start: same as the Step 6 Student on the whole sequence
        _, _, f_st, _ = m.student(img[:, 1:2], st[:, 1:2])
        assert torch.allclose(f_st[:, 0], feats[:, 1], atol=1e-5)
        # column 0 after the reset = a fresh Student run on steps 5..7
        _, _, f_new, _ = m.student(img[5:, 0:1], st[5:, 0:1])
        assert torch.allclose(f_new[:, 0], feats[5:, 0], atol=1e-5)


# ---------------------------------------------------------------- 6. ratio = 1
def test_learner_recomputes_worker_chances():
    m = model()
    rng = np.random.default_rng(0)
    g = torch.Generator().manual_seed(5)
    T, L = 8, 4
    keys = ("image", "state", "priv", "move_raw", "drop", "logp", "value", "start", "h", "c", "reward")
    buf = {k: [] for k in keys}
    hc = None
    for t in range(T):
        start = int(t in (0, 5))
        if start:
            hc = m.zero_hc()
        obs = dict(image=torch.randint(0, 255, (H, Wd), generator=g, dtype=torch.uint8).numpy(),
                   state=torch.randn(11, generator=g).numpy(), priv=torch.randn(8, generator=g).numpy())
        buf["h"].append(hc[0][0, 0].numpy().copy())
        buf["c"].append(hc[1][0, 0].numpy().copy())
        _, rec, hc = PPO.act(m, obs, hc, rng)
        for k in ("move_raw", "drop", "logp", "value"):
            buf[k].append(rec[k])
        for k in ("image", "state", "priv"):
            buf[k].append(obs[k])
        buf["start"].append(start)
        buf["reward"].append(0.0)
    roll = {k: np.asarray(v)[None] for k, v in buf.items()}
    roll["adv"] = np.ones((1, T), np.float32)
    roll["target"] = np.zeros((1, T), np.float32)
    ch = PPO.make_chunks(roll, L)
    assert ch["image"].shape == (2, L, H, Wd) and ch["h0"].shape == (2, 256)
    b = PPO.to_batch(ch, np.arange(2), "cpu")
    with torch.no_grad():
        _, st = PPO.ppo_losses(m, b)
        feats, _ = m.unroll(b["image"], b["state"], (b["h0"], b["c0"]), b["starts"])
        v = m.value(b["priv"], feats)
    print(f"\n[ratio] max |ratio - 1| = {st['ratio_dev']:.2e}; KL {st['kl']:.1e}")
    assert st["ratio_dev"] < 1e-4 and abs(st["kl"]) < 1e-8
    assert v.transpose(0, 1).reshape(-1).numpy() == pytest.approx(roll["value"][0], abs=1e-5)


# ---------------------------------------------------------------- 7. the clip
def test_clip_stops_the_push():
    m = model()
    b = fake_batch(m)
    for adv, expect_grad in ((1.0, False), (-1.0, True)):
        bb = dict(b, logp_old=b["logp_old"] - math.log(1.5), adv=torch.full_like(b["adv"], adv))
        m.zero_grad(set_to_none=True)
        loss, st = PPO.ppo_losses(m, bb)
        loss.backward()
        g = m.student.move_head[0].weight.grad
        has = g is not None and float(g.abs().max()) > 0
        print(f"\n[clip] ratio 1.5, advantage {adv:+.0f}: clip fraction {st['clipfrac']:.2f}, actor gradient {has}")
        assert st["clipfrac"] == pytest.approx(1.0) and has == expect_grad


# ---------------------------------------------------------------- 8. end to end
def test_train_ppo_end_to_end(tmp_path, capsys):
    torch.manual_seed(0)
    s = Student()
    init = tmp_path / "init.pt"
    torch.save(dict(model=s.state_dict(), size=256, epoch=0), init)
    common = ["--name", "t", "--out-root", str(tmp_path), "--workers", "1", "--steps", "16", "--chunk", "8",
              "--mb-chunks", "2", "--critic-min", "2", "--critic-max", "2", "--exam-every", "1",
              "--exam-n", "1", "--renderer", "tiny", "--device", "cpu", "--actor-warmup", "1",
              "--lr-actor", "1e-3"]
    TP.main(["--init", str(init), "--rounds", "2"] + common)
    run = tmp_path / "t"
    ck = torch.load(run / "last.pt", weights_only=False)
    assert ck["ppo_start"] is None and ck["round"] == 1 and ck["steps"] == 32
    for k, v in s.state_dict().items():
        assert torch.equal(ck["model"][k], v), k            # frozen during the critic warm-up
    TP.main(["--resume", str(run / "last.pt"), "--rounds", "3"] + common)
    out = capsys.readouterr().out
    ck2 = torch.load(run / "last.pt", weights_only=False)
    assert ck2["ppo_start"] == 2 and ck2["round"] == 2 and ck2["steps"] == 48
    changed = [k for k, v in s.state_dict().items() if not torch.equal(ck2["model"][k], v)]
    assert "memory.weight_hh_l0" in changed and "move_head.0.weight" in changed
    rows = list(csv.DictReader(open(run / "metrics.csv")))
    print(f"\n[e2e] phases {[r['phase'] for r in rows]}; changed {len(changed)} actor tensors")
    assert [r["phase"] for r in rows] == ["critic", "critic", "ppo"]
    assert (run / "r0002.pt").exists() and (run / "exam_r0002.csv").exists()
    assert "[exam] round 2" in out and "[guard]" in out     # untrained student: 0/1 < 90 %
    m, _ = ES.load(str(run / "last.pt"), "cpu")
    assert m.size == 256
