"""Actor, estimator and asymmetric critic (V5 Block 11; Shin et al. Figs. 3-4, Sec. III-B/D; SPEC §10).

Vision actor (mode='vision'):
  l_t = CNN(image_t)                                   R^512              [paper dim]
  h_t = LSTM([l_t, u_t], h_{t-1})                      R^512, 1 layer     [paper dim; layers unspec.]
  y_t = MLP_est([l_t, h_t, u_t])  1031 -> 512 -> 256   R^256 (N = 256)    [paper dims; hidden unspec.]
  s~_t = y_t[0:6]                 estimate of s_rel (body-frame dx, dv)    [paper]
  mu_t = MLP_dec([y_t, u_t])      263 -> 256 -> 128 -> 4                   [inputs paper; sizes unspec.]
  a_t ~ N(mu_t, exp(log_std)),    state-independent log_std, init -0.5     [unspecified]
Privileged actor (mode='privileged', variant P): mu_t = MLP_dec([s_rel, u_t]), no CNN/LSTM/estimate.
Critic (both modes): V_t = MLP([u_clean, s_rel]) 13 -> 256 -> 256 -> 1, non-recurrent, discarded at
  deployment [paper: o_priv]. Critic inputs are scaled by a fixed vector [unspecified].

The actor reads ONLY the keys in shin_env.ACTOR_KEYS[mode]; tests check this numerically.
Hidden-state handling: before step t, h <- h * (1 - episode_start_t) (the cleanRL convention).
"""
import torch
import torch.nn as nn
from torch.distributions import Normal

from .encoder import EMBED_DIM, ImageEncoder

H = 512
Y_DIM = 256
U_DIM = 7
S_DIM = 6
# Fixed critic-input scale [unspecified]: v_b (m/s), quaternion, relative position (m),
# relative velocity (m/s), so that typical magnitudes are O(1).
CRITIC_SCALE = torch.tensor([5, 5, 5, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5], dtype=torch.float32)


def mlp(sizes, act=nn.ELU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class ShinPolicy(nn.Module):
    def __init__(self, mode="vision", log_std_init=-0.5, act_dim=4):
        super().__init__()
        assert mode in ("vision", "privileged")
        self.mode = mode
        if mode == "vision":
            self.encoder = ImageEncoder()
            self.lstm = nn.LSTM(EMBED_DIM + U_DIM, H, num_layers=1)
            self.est = mlp([EMBED_DIM + H + U_DIM, 512, Y_DIM])
            self.dec = mlp([Y_DIM + U_DIM, 256, 128, act_dim])
        else:
            self.dec = mlp([S_DIM + U_DIM, 256, 128, act_dim])
        self.critic = mlp([U_DIM + S_DIM, 256, 256, 1])
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.register_buffer("critic_scale", CRITIC_SCALE.clone())

    # ---- recurrent state ---------------------------------------------------------
    def initial_state(self, batch, device=None):
        if self.mode != "vision":
            return None
        z = torch.zeros(1, batch, H, device=device)
        return (z, z.clone())

    # ---- critic -------------------------------------------------------------------
    def value(self, critic_obs):
        return self.critic(critic_obs / self.critic_scale).squeeze(-1)

    # ---- actor over a sequence --------------------------------------------------------
    def actor_seq(self, obs, state, episode_starts, inject_true_state=False):
        """obs: dict of (T, B, ...) tensors; episode_starts: (T, B) float (1 = new episode at t).
        Returns mu (T, B, A), s_est (T, B, 6) or None, new state.

        inject_true_state (EVALUATION DIAGNOSTIC ONLY, vision): the decision layer receives
        y with y[0:6] replaced by the TRUE s_rel (obs["target"]); the other 250 latent dims are
        unchanged. The returned s_est is still the estimator's own output. Never used in training."""
        u = obs["u"]
        T, B = u.shape[:2]
        if self.mode == "privileged":
            return self.dec(torch.cat([obs["s_rel"], u], -1)), None, None
        img = obs["image"]
        l = self.encoder(img.reshape(T * B, *img.shape[2:])).reshape(T, B, EMBED_DIM)
        x = torch.cat([l, u], -1)
        h, c = state
        hs = []
        for t in range(T):
            keep = (1.0 - episode_starts[t]).view(1, B, 1)
            h, c = h * keep, c * keep
            out, (h, c) = self.lstm(x[t:t + 1], (h, c))
            hs.append(out[0])
        hseq = torch.stack(hs)
        y = self.est(torch.cat([l, hseq, u], -1))
        y_dec = torch.cat([obs["target"], y[..., S_DIM:]], -1) if inject_true_state else y
        mu = self.dec(torch.cat([y_dec, u], -1))
        return mu, y[..., :S_DIM], (h, c)

    def dist(self, mu):
        return Normal(mu, self.log_std.exp().expand_as(mu))

    # ---- rollout step ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs, state, episode_starts, deterministic=False):
        """obs: dict of (B, ...) tensors for ONE step. Returns action, logp, value, s_est, state."""
        seq = {k: v.unsqueeze(0) for k, v in obs.items()}
        mu, s_est, state = self.actor_seq(seq, state, episode_starts.unsqueeze(0))
        d = self.dist(mu[0])
        a = mu[0] if deterministic else d.sample()
        return (a, d.log_prob(a).sum(-1), self.value(obs["critic"]),
                None if s_est is None else s_est[0], state)

    # ---- PPO update ----------------------------------------------------------------------
    def evaluate(self, obs, actions, state, episode_starts):
        """Sequences (T, B, ...). Returns logp (T, B), entropy (T, B), value (T, B), s_est."""
        mu, s_est, _ = self.actor_seq(obs, state, episode_starts)
        d = self.dist(mu)
        return (d.log_prob(actions).sum(-1), d.entropy().sum(-1), self.value(obs["critic"]), s_est)

    # ---- parameter groups (for tests / logging) ----------------------------------------------
    def groups(self):
        g = {"critic": list(self.critic.parameters()), "decision": list(self.dec.parameters()),
             "log_std": [self.log_std]}
        if self.mode == "vision":
            g.update(encoder=list(self.encoder.parameters()), lstm=list(self.lstm.parameters()),
                     estimation=list(self.est.parameters()))
        return g
