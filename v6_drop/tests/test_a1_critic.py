"""v6_drop A1 (12 Nov): ABLATION 1, the camera-only critic (learn/ppo.py critic_input,
scripts/train_ppo.py --critic-input).

Main method (default "priv_h"): the critic (coach, training only) reads the 8 true-state
numbers (priv) + the actor's memory h (256, detached).
Ablation 1 ("h"): the critic reads ONLY the memory h (256, detached). Nothing else changes.

Uses UNTRAINED networks (fixed torch seed) and TinyRenderer, so it needs no results/.

Checks:
  1. the default is the main method EXACTLY: same layers, same weight names and shapes
     (first layer 264 -> 256), V = net([priv, h]); an old critic state_dict loads as it is
  2. the camera-only critic never reads priv: first layer 256 -> 256; V is identical when
     priv is changed to random numbers or even NaN
  3. the camera-only critic still cannot change the memory: its value loss gives no gradient
     to the eyes, the LSTM or the heads, while its own layers do get gradients
  4. train_ppo --critic-input h end to end (1 worker): config and checkpoint say "h", the
     checkpoint critic has the 256-input layer, the start line names ablation 1, the PPO
     round changes the actor; --resume with the wrong --critic-input stops with a clear
     message; an old checkpoint with no 'critic_input' key resumes as the main method;
     eval_student.load reads the ablation checkpoint (the actor is the same Student)

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_a1_critic.py -v -s
"""
import json

import pytest
import torch

from v6_drop.learn import ppo as PPO
from v6_drop.learn.student import Student
from v6_drop.scripts import eval_student as ES
from v6_drop.scripts import train_ppo as TP


def model(critic_input, seed=0):
    torch.manual_seed(seed)
    return PPO.ActorCritic(Student(), critic_input=critic_input).eval()


def inputs(B=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 8, generator=g), torch.randn(B, 256, generator=g)


# ---------------------------------------------------------------- 1. default = main method
def test_default_is_main_method():
    m_def = model("priv_h")
    torch.manual_seed(0)
    m_old = PPO.ActorCritic(Student()).eval()                  # the call used before A1
    assert m_old.critic_input == "priv_h" and m_old.critic.use_priv
    sd_def, sd_old = m_def.critic.state_dict(), m_old.critic.state_dict()
    assert list(sd_def) == list(sd_old)
    for k in sd_def:
        assert sd_def[k].shape == sd_old[k].shape and torch.equal(sd_def[k], sd_old[k]), k
    assert tuple(m_def.critic.net[0].weight.shape) == (256, 264)
    priv, h = inputs()
    with torch.no_grad():
        v = m_def.value(priv, h)
        v_hand = m_def.critic.net(torch.cat([priv, h], -1)).squeeze(-1)
    assert torch.equal(v, v_hand)
    m_def.critic.load_state_dict(sd_old)                      # old checkpoints load as they are
    print(f"\n[a1] default critic: first layer {tuple(m_def.critic.net[0].weight.shape)}, "
          f"{sum(p.numel() for p in m_def.critic.parameters()):,} numbers")


# ---------------------------------------------------------------- 2. h only never reads priv
def test_camera_only_never_reads_priv():
    m = model("h")
    assert m.critic_input == "h" and not m.critic.use_priv
    assert tuple(m.critic.net[0].weight.shape) == (256, 256)
    priv, h = inputs()
    with torch.no_grad():
        v1 = m.value(priv, h)
        v2 = m.value(torch.randn_like(priv) * 100, h)
        v3 = m.value(torch.full_like(priv, float("nan")), h)
        v_hand = m.critic.net(h).squeeze(-1)
    assert torch.equal(v1, v2) and torch.equal(v1, v3) and torch.equal(v1, v_hand)
    assert torch.isfinite(v1).all()
    with pytest.raises(AssertionError):
        PPO.ActorCritic(Student(), critic_input="priv")         # only the two named inputs
    print(f"\n[a1] camera-only critic: first layer {tuple(m.critic.net[0].weight.shape)}; "
          f"V with priv / random priv / NaN priv: {float(v1[0]):.6f} {float(v2[0]):.6f} {float(v3[0]):.6f}")


# ---------------------------------------------------------------- 3. still cannot change h
def test_camera_only_critic_cannot_change_memory():
    m = model("h")
    m.train()
    g = torch.Generator().manual_seed(2)
    L, B = 4, 3
    image = torch.randint(0, 255, (L, B, 160, 256), generator=g, dtype=torch.uint8)
    state = torch.randn(L, B, 11, generator=g)
    priv = torch.randn(L, B, 8, generator=g)
    starts = torch.zeros(L, B)
    starts[0] = 1.0
    feats, _ = m.unroll(image, state, m.zero_hc(B), starts)
    v = m.value(priv, feats)
    v_loss = 0.5 * (v - torch.randn(L, B, generator=g) * 5).pow(2).mean()
    v_loss.backward()
    actor_grads = [n for n, p in m.student.named_parameters()
                   if p.grad is not None and float(p.grad.abs().max()) > 0]
    critic_grads = [n for n, p in m.critic.named_parameters()
                    if p.grad is not None and float(p.grad.abs().max()) > 0]
    print(f"\n[a1] value loss {float(v_loss):.3f}: actor tensors with gradient {len(actor_grads)}, "
          f"critic tensors with gradient {len(critic_grads)}/{len(list(m.critic.parameters()))}")
    assert actor_grads == []
    assert len(critic_grads) == len(list(m.critic.parameters()))


# ---------------------------------------------------------------- 4. train_ppo end to end
def test_train_ppo_critic_h_end_to_end(tmp_path, capsys):
    torch.manual_seed(0)
    s = Student()
    init = tmp_path / "init.pt"
    torch.save(dict(model=s.state_dict(), size=256, epoch=0), init)
    common = ["--name", "a", "--out-root", str(tmp_path), "--workers", "1", "--steps", "16", "--chunk", "8",
              "--mb-chunks", "2", "--critic-min", "1", "--critic-max", "1", "--exam-every", "1",
              "--exam-n", "1", "--renderer", "tiny", "--device", "cpu", "--actor-warmup", "1",
              "--lr-actor", "1e-3", "--guard", "0"]
    TP.main(["--init", str(init), "--rounds", "2", "--critic-input", "h"] + common)
    out = capsys.readouterr().out
    run = tmp_path / "a"
    cfg = json.load(open(run / "config.json"))
    ck = torch.load(run / "last.pt", weights_only=False)
    print(f"\n[a1] config critic_input {cfg['critic_input']}; checkpoint critic_input {ck['critic_input']}; "
          f"critic first layer {tuple(ck['critic']['net.0.weight'].shape)}; ppo from round {ck['ppo_start']}")
    assert "[ppo] critic input: h (memory h ONLY, camera-only critic = ablation 1)" in out
    assert cfg["critic_input"] == "h" and ck["critic_input"] == "h"
    assert tuple(ck["critic"]["net.0.weight"].shape) == (256, 256)
    assert ck["ppo_start"] == 1 and ck["round"] == 1
    changed = [k for k, v in s.state_dict().items() if not torch.equal(ck["model"][k], v)]
    assert "memory.weight_hh_l0" in changed and "move_head.0.weight" in changed
    assert (run / "r0001.pt").exists() and "[exam] round 1" in out

    # resume with the wrong critic input: clear stop before any worker starts
    with pytest.raises(AssertionError, match="give --critic-input h"):
        TP.main(["--resume", str(run / "last.pt"), "--rounds", "3"] + common)

    # an old checkpoint (no 'critic_input' key) resumes as the main method
    torch.manual_seed(0)
    old = PPO.ActorCritic(Student())
    ck_old = dict(ck, model=old.student.state_dict(), critic=old.critic.state_dict())
    del ck_old["critic_input"]
    torch.save(ck_old, tmp_path / "old.pt")
    TP.main(["--resume", str(tmp_path / "old.pt"), "--rounds", "2", "--name", "o"] + common[2:])
    out = capsys.readouterr().out
    assert "resumed" in out and "[ppo] critic input: priv_h (true state + memory h, main method)" in out

    # the deployed part (the Student) loads as usual
    m, _ = ES.load(str(run / "r0001.pt"), "cpu")
    assert m.size == 256
