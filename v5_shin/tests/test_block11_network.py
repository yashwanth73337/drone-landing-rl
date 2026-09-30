"""Block 11: network (Figs. 3-4): dimensions, privilege boundary, recurrence, gradient routing.

Run:  python -m pytest v5_shin/tests/test_block11_network.py -v -s
"""
import numpy as np
import pytest
import torch
from torch.distributions import Normal

from v5_shin.policies.encoder import OBS_H, OBS_W
from v5_shin.policies.shin_policy import H, S_DIM, Y_DIM, ShinPolicy


def rand_obs(T, B, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "image": torch.randint(0, 256, (T, B, OBS_H, OBS_W), dtype=torch.uint8, generator=g),
        "u": torch.randn(T, B, 7, generator=g),
        "critic": torch.randn(T, B, 13, generator=g),
        "target": torch.randn(T, B, 6, generator=g),
        "s_rel": torch.randn(T, B, 6, generator=g),
    }


def n(module):
    return sum(p.numel() for p in module.parameters())


def test_dimensions_match_paper():
    torch.manual_seed(0)
    pol = ShinPolicy("vision")
    assert pol.lstm.input_size == 512 + 7 and pol.lstm.hidden_size == H == 512
    assert pol.est[0].in_features == 512 + 512 + 7 and pol.est[-1].out_features == Y_DIM == 256
    assert pol.dec[0].in_features == 256 + 7 and pol.dec[-1].out_features == 4
    assert pol.critic[0].in_features == 13
    obs = rand_obs(3, 2)
    mu, s_est, (h, c) = pol.actor_seq(obs, pol.initial_state(2), torch.zeros(3, 2))
    assert mu.shape == (3, 2, 4) and s_est.shape == (3, 2, S_DIM) and h.shape == (1, 2, H)
    print(f"\n[params] encoder {n(pol.encoder):,}, lstm {n(pol.lstm):,}, estimation {n(pol.est):,}, "
          f"decision {n(pol.dec):,}, critic {n(pol.critic):,}, total {n(pol):,}")


def test_estimate_is_first_six_of_y():
    torch.manual_seed(0)
    pol = ShinPolicy("vision").eval()
    obs = rand_obs(2, 3)
    with torch.no_grad():
        _, s_est, _ = pol.actor_seq(obs, pol.initial_state(3), torch.zeros(2, 3))
        # recompute y by hand
        l = pol.encoder(obs["image"].reshape(6, OBS_H, OBS_W)).reshape(2, 3, -1)
        x = torch.cat([l, obs["u"]], -1)
        out, _ = pol.lstm(x, pol.initial_state(3))
        y = pol.est(torch.cat([l, out, obs["u"]], -1))
    assert torch.allclose(s_est, y[..., :6], atol=1e-5)


@pytest.mark.parametrize("mode,hidden_keys,visible_key", [
    ("vision", ("critic", "target", "s_rel"), "image"),
    ("privileged", ("critic", "target", "image"), "s_rel"),
])
def test_actor_privilege_boundary(mode, hidden_keys, visible_key):
    """The actor's outputs must not change when keys outside ACTOR_KEYS[mode] change,
    and must change when an actor key changes."""
    torch.manual_seed(0)
    pol = ShinPolicy(mode).eval()
    a, b = rand_obs(4, 2, seed=1), rand_obs(4, 2, seed=1)
    for k in hidden_keys:
        b[k] = rand_obs(4, 2, seed=99)[k]
    st = pol.initial_state(2)
    starts = torch.zeros(4, 2)
    with torch.no_grad():
        ma, sa, _ = pol.actor_seq(a, st, starts)
        mb, sb, _ = pol.actor_seq(b, st, starts)
        assert torch.equal(ma, mb)
        if sa is not None:
            assert torch.equal(sa, sb)
        b[visible_key] = rand_obs(4, 2, seed=7)[visible_key]
        mc, _, _ = pol.actor_seq(b, st, starts)
        assert not torch.allclose(ma, mc)


def test_critic_reads_only_critic_obs():
    torch.manual_seed(0)
    pol = ShinPolicy("vision").eval()
    a = rand_obs(1, 5)
    v1 = pol.value(a["critic"][0])
    a["image"] = torch.zeros_like(a["image"])
    a["u"] = torch.zeros_like(a["u"])
    v2 = pol.value(a["critic"][0])
    assert torch.equal(v1, v2) and v1.shape == (5,)


def test_step_by_step_equals_sequence_with_resets():
    """act() one step at a time (as in rollouts) must give the same log-probs, values and
    estimates as evaluate() on the whole sequence (as in the PPO update), including an
    episode boundary in the middle where the LSTM state must be zeroed."""
    torch.manual_seed(0)
    pol = ShinPolicy("vision").eval()
    T, B = 6, 3
    obs = rand_obs(T, B, seed=2)
    starts = torch.zeros(T, B)
    starts[0] = 1
    starts[3, 1] = 1                                  # env 1 starts a new episode at t = 3
    st = pol.initial_state(B)
    acts, logps, vals, ests = [], [], [], []
    g = torch.Generator().manual_seed(5)
    for t in range(T):
        a, lp, v, e, st = pol.act({k: x[t] for k, x in obs.items()}, st, starts[t])
        acts.append(a), logps.append(lp), vals.append(v), ests.append(e)
    A = torch.stack(acts)
    with torch.no_grad():
        lp2, ent, v2, e2 = pol.evaluate(obs, A, pol.initial_state(B), starts)
    assert torch.allclose(torch.stack(logps), lp2, atol=1e-4)
    assert torch.allclose(torch.stack(vals), v2, atol=1e-5)
    assert torch.allclose(torch.stack(ests), e2, atol=1e-5)
    # the reset really isolates env 1's second episode from its first
    sub = {k: x[3:, 1:2] for k, x in obs.items()}
    with torch.no_grad():
        _, _, _, e_fresh = pol.evaluate(sub, A[3:, 1:2], pol.initial_state(1), torch.zeros(3, 1))
    assert torch.allclose(e2[3:, 1:2], e_fresh, atol=1e-5)


def test_log_prob_and_entropy():
    torch.manual_seed(0)
    pol = ShinPolicy("privileged")
    mu = torch.randn(5, 4)
    a = torch.randn(5, 4)
    d = pol.dist(mu)
    ref = Normal(mu, torch.full((4,), np.exp(-0.5))).log_prob(a).sum(-1)
    assert torch.allclose(d.log_prob(a).sum(-1), ref, atol=1e-6)
    assert torch.allclose(pol.log_std, torch.full((4,), -0.5))


def _grad_groups(pol, loss):
    pol.zero_grad()
    loss.backward()
    return {k: any(p.grad is not None and p.grad.abs().sum() > 0 for p in ps)
            for k, ps in pol.groups().items()}


def test_gradient_routing():
    """L_est reaches encoder, LSTM and estimation layer, but not decision, critic or log_std.
    The policy loss reaches the whole actor but not the critic. The value loss reaches only
    the critic."""
    torch.manual_seed(0)
    pol = ShinPolicy("vision")
    obs = rand_obs(3, 2, seed=3)
    st, starts = pol.initial_state(2), torch.zeros(3, 2)
    acts = torch.randn(3, 2, 4)
    logp, ent, v, s_est = pol.evaluate(obs, acts, st, starts)
    g = _grad_groups(pol, ((s_est - obs["target"]) ** 2).mean())
    assert g == dict(critic=False, decision=False, log_std=False, encoder=True, lstm=True,
                     estimation=True), g
    logp, ent, v, s_est = pol.evaluate(obs, acts, st, starts)
    g = _grad_groups(pol, -logp.mean())
    assert g == dict(critic=False, decision=True, log_std=True, encoder=True, lstm=True,
                     estimation=True), g
    logp, ent, v, s_est = pol.evaluate(obs, acts, st, starts)
    g = _grad_groups(pol, v.pow(2).mean())
    assert g == dict(critic=True, decision=False, log_std=False, encoder=False, lstm=False,
                     estimation=False), g


def test_no_batch_dependence():
    torch.manual_seed(0)
    pol = ShinPolicy("vision").eval()
    obs = rand_obs(2, 4, seed=4)
    with torch.no_grad():
        m_all, e_all, _ = pol.actor_seq(obs, pol.initial_state(4), torch.zeros(2, 4))
        m_one, e_one, _ = pol.actor_seq({k: v[:, 2:3] for k, v in obs.items()},
                                        pol.initial_state(1), torch.zeros(2, 1))
    assert torch.allclose(m_all[:, 2:3], m_one, atol=1e-5)
    assert torch.allclose(e_all[:, 2:3], e_one, atol=1e-5)


def test_env_observation_roundtrip():
    """One real env step through the policy: numpy obs -> tensors -> action in range."""
    from v5_shin.envs.dr import DRConfig
    from v5_shin.envs.shin_env import ShinLandingEnv
    env = ShinLandingEnv(mode="vision", renderer="tiny", egl=False, seed=0, dr=DRConfig())
    pol = ShinPolicy("vision").eval()
    try:
        obs, _ = env.reset(seed=0)
        st = pol.initial_state(1)
        for t in range(3):
            tob = {k: torch.as_tensor(np.asarray(v))[None] for k, v in obs.items()}
            a, lp, v, e, st = pol.act(tob, st, torch.tensor([1.0 if t == 0 else 0.0]))
            obs, r, te, tr, info = env.step(a[0].numpy())
        assert a.shape == (1, 4) and e.shape == (1, 6) and np.isfinite(r)
    finally:
        env.close()
