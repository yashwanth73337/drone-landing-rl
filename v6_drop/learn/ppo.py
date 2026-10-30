"""v6_drop Step 7d: PPO pieces (decided 30 Oct, P2-P6; taught in 7.1-7.5).

ActorCritic
  ACTOR  = the Step 6 Student (eyes CNN -> LSTM memory h -> move head, drop head), started
           from dagger_r4, + a learnable noise width log_std (4 numbers, P4). Only this part
           flies after training. It reads only image + state (never priv).
  CRITIC = a new small network V([priv (8), h DETACHED (256)]) -> 256 -> 256 -> 1 (P5, the
           asymmetric critic V(h, s) decided 28 Oct). h.detach(): the critic READS the
           actor's memory but its loss cannot change the LSTM or the eyes.
  The dice (P4): move = mean + Normal(0, std) (the env clips to [-1, 1]; the log-chance
  is for the unclipped sample), drop = Bernoulli(sigmoid(drop_logit)). One joint action:
  log pi(a | o) = log N(move) + log Bernoulli(drop).

unroll(): the LSTM run step by step over a (T, B) sequence with an episode-start mask, so
a chunk may hold the end of one flight and the start of the next (memory reset to zero
at a start). Acting in the workers uses the same function with T = 1, so the learner
recomputes exactly the chances the workers used (ratio = 1 at the start of a round).

gae(): 7.3. terminated -> the future is 0; truncated (timeout, P3) -> the future is the
bootstrap value V(final observation); the rollout cut -> V(next observation).

ppo_losses(): 7.4. clipped surrogate (eps 0.2), value loss 0.5 (V - target)^2, entropy,
approx KL = mean((ratio - 1) - log ratio), clip fraction. Entropy bonus coefficient 0 at
the start (P4); the loss is pg_loss + v_loss - ent_coef x entropy.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v6_drop.learn.student import Student

PRIV_DIM = 8
LOG_2PI = math.log(2 * math.pi)
LOG_STD_MIN, LOG_STD_MAX = math.log(0.005), math.log(0.5)     # policy units (action / ACT_SCALE)


class Critic(nn.Module):
    def __init__(self, feat=256, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(PRIV_DIM + feat, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))

    def forward(self, priv, feats):
        return self.net(torch.cat([priv, feats], dim=-1)).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, student=None, sigma_n=(0.2 / 6, 0.2 / 6, 0.05, 0.05), size=256):
        super().__init__()
        self.student = student if student is not None else Student(size)
        self.log_std = nn.Parameter(torch.log(torch.as_tensor(sigma_n, dtype=torch.float32)))
        self.critic = Critic(self.student.size)

    def actor_parameters(self):
        return list(self.student.parameters()) + [self.log_std]

    def unroll(self, image, state, hc, starts):
        """image (T, B, H, W) uint8, state (T, B, 11), hc = (h, c) each (1, B, size),
        starts (T, B) float: 1 where a new flight begins (memory reset to 0 before that
        step). Returns feats (T, B, size) and the final (h, c)."""
        T, B = image.shape[:2]
        st = self.student
        z = st.eyes(image.reshape(T * B, *image.shape[2:])).reshape(T, B, -1)
        x = torch.cat([z, state], dim=-1)
        h, c = hc
        outs = []
        for t in range(T):
            keep = (1.0 - starts[t]).view(1, B, 1)
            h, c = h * keep, c * keep
            o, (h, c) = st.memory(x[t:t + 1], (h, c))
            outs.append(o)
        return torch.cat(outs, 0), (h, c)

    def heads(self, feats):
        move_mean = self.student.move_head(feats)
        drop_logit = self.student.drop_head(feats).squeeze(-1)
        return move_mean, drop_logit

    def value(self, priv, feats):
        return self.critic(priv, feats.detach())            # the stop-gradient (28 Oct)

    def clamped_log_std(self):
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def zero_hc(self, B=1, device="cpu"):
        z = torch.zeros(1, B, self.student.size, device=device)
        return z, z.clone()


def log_prob(move_mean, log_std, drop_logit, move_raw, drop):
    """log pi(a | o) for the joint action (Normal move, Bernoulli drop)."""
    std = log_std.exp()
    lp_move = (-0.5 * ((move_raw - move_mean) / std) ** 2 - log_std - 0.5 * LOG_2PI).sum(-1)
    lp_drop = -F.binary_cross_entropy_with_logits(drop_logit, drop.float(), reduction="none")
    return lp_move + lp_drop


def entropy(log_std, drop_logit):
    ent_move = (0.5 + 0.5 * LOG_2PI + log_std).sum()
    p = torch.sigmoid(drop_logit)
    ent_drop = F.binary_cross_entropy_with_logits(drop_logit, p, reduction="none")
    return ent_move + ent_drop


@torch.no_grad()
def act(model, obs, hc, rng, deterministic=False):
    """One decision in a worker (CPU). Returns (action dict for the env, record, new hc).
    record: move_raw (unclipped sample), drop, logp, value, p_drop."""
    img = torch.as_tensor(obs["image"]).reshape(1, 1, *obs["image"].shape)
    st = torch.as_tensor(obs["state"], dtype=torch.float32).reshape(1, 1, -1)
    pv = torch.as_tensor(obs["priv"], dtype=torch.float32).reshape(1, 1, -1)
    feats, hc2 = model.unroll(img, st, hc, torch.zeros(1, 1))
    mean, logit = model.heads(feats)
    log_std = model.clamped_log_std()
    mean, logit = mean[0, 0].numpy().astype(np.float64), float(logit[0, 0])
    p = 1.0 / (1.0 + math.exp(-logit))
    if deterministic:
        move_raw, drop = mean.copy(), int(p > 0.5)
    else:
        move_raw = mean + rng.normal(0.0, 1.0, 4) * np.exp(log_std.numpy())
        drop = int(rng.random() < p)
    lp = log_prob(torch.as_tensor(mean), log_std.double(), torch.tensor(logit, dtype=torch.float64),
                  torch.as_tensor(move_raw), torch.tensor(drop))
    v = float(model.value(pv, feats)[0, 0])
    action = {"move": np.clip(move_raw, -1.0, 1.0).astype(np.float32), "drop": drop}
    rec = dict(move_raw=move_raw.astype(np.float32), drop=drop, logp=float(lp), value=v, p_drop=p)
    return action, rec, hc2


@torch.no_grad()
def value_of(model, obs, hc):
    """V of an observation the actor has not acted on yet (rollout cut / timeout bootstrap)."""
    img = torch.as_tensor(obs["image"]).reshape(1, 1, *obs["image"].shape)
    st = torch.as_tensor(obs["state"], dtype=torch.float32).reshape(1, 1, -1)
    pv = torch.as_tensor(obs["priv"], dtype=torch.float32).reshape(1, 1, -1)
    feats, _ = model.unroll(img, st, hc, torch.zeros(1, 1))
    return float(model.value(pv, feats)[0, 0])


def gae(rewards, values, terminated, truncated, boot, last_value, gamma=0.99, lam=0.95):
    """One worker's rollout, arrays of length T (7.3). boot[t] = V(final observation) where
    truncated[t]; last_value = V(observation after the last step) if that flight goes on.
    Returns (advantages, targets = advantages + values)."""
    T = len(rewards)
    adv = np.zeros(T, np.float64)
    run = 0.0
    for t in reversed(range(T)):
        if terminated[t]:
            nv, carry = 0.0, 0.0
        elif truncated[t]:
            nv, carry = boot[t], 0.0
        else:
            nv = last_value if t == T - 1 else values[t + 1]
            carry = run
        delta = rewards[t] + gamma * nv - values[t]
        run = delta + gamma * lam * carry
        adv[t] = run
    return adv, adv + np.asarray(values, np.float64)


def explained_variance(values, targets):
    var = np.var(targets)
    return float("nan") if var < 1e-12 else float(1.0 - np.var(targets - values) / var)


def ppo_losses(model, batch, clip=0.2, train_actor=True, ent_coef=0.0):
    """batch (torch, on the learner device): image (L, B, H, W), state, priv, move_raw
    (L, B, 4), drop, logp_old, adv (normalised), target, starts (L, B), h0/c0 (1, B, size).
    Returns (loss, stats dict). With train_actor=False (critic warm-up) the eyes and the
    LSTM run without gradients and only the value loss is returned."""
    hc = (batch["h0"], batch["c0"])
    if train_actor:
        feats, _ = model.unroll(batch["image"], batch["state"], hc, batch["starts"])
    else:
        with torch.no_grad():
            feats, _ = model.unroll(batch["image"], batch["state"], hc, batch["starts"])
    v = model.value(batch["priv"], feats)
    v_loss = 0.5 * (v - batch["target"]).pow(2).mean()
    stats = dict(v_loss=float(v_loss.detach()))
    if not train_actor:
        return v_loss, stats
    mean, logit = model.heads(feats)
    log_std = model.clamped_log_std()
    lp = log_prob(mean, log_std, logit, batch["move_raw"], batch["drop"])
    log_ratio = lp - batch["logp_old"]
    ratio = log_ratio.exp()
    adv = batch["adv"]
    surr = torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    pg_loss = -surr.mean()
    ent = entropy(log_std, logit).mean()
    with torch.no_grad():
        kl = float(((ratio - 1) - log_ratio).mean())
        clipfrac = float(((ratio - 1).abs() > clip).float().mean())
        ratio_dev = float((ratio - 1).abs().max())
    stats.update(pg_loss=float(pg_loss.detach()), entropy=float(ent.detach()), kl=kl, clipfrac=clipfrac,
                 ratio_dev=ratio_dev)
    return pg_loss + v_loss - ent_coef * ent, stats


def make_chunks(roll, L):
    """roll: dict of arrays shaped (W, T, ...) from the workers (T a multiple of L) plus h/c
    (W, T, size) = the memory BEFORE each step. Returns a dict of arrays (N, L, ...) with
    h0/c0 (N, size) = the memory at each chunk start."""
    W, T = roll["reward"].shape
    assert T % L == 0, (T, L)
    n = W * (T // L)
    out = {}
    for k, a in roll.items():
        if k in ("h", "c"):
            continue
        out[k] = a.reshape(W, T // L, L, *a.shape[2:]).reshape(n, L, *a.shape[2:])
    out["h0"] = roll["h"][:, ::L].reshape(n, -1)
    out["c0"] = roll["c"][:, ::L].reshape(n, -1)
    return out


def to_batch(chunks, idx, device):
    """Select chunks idx -> torch tensors shaped (L, B, ...) for ppo_losses."""
    def tt(a, dtype=None):
        t = torch.as_tensor(a[idx])
        if dtype is not None:
            t = t.to(dtype)
        return t.transpose(0, 1).contiguous().to(device, non_blocking=True)
    b = dict(image=tt(chunks["image"]), state=tt(chunks["state"], torch.float32),
             priv=tt(chunks["priv"], torch.float32), move_raw=tt(chunks["move_raw"], torch.float32),
             drop=tt(chunks["drop"], torch.float32), logp_old=tt(chunks["logp"], torch.float32),
             adv=tt(chunks["adv"], torch.float32), target=tt(chunks["target"], torch.float32),
             starts=tt(chunks["start"], torch.float32))
    b["h0"] = torch.as_tensor(chunks["h0"][idx], dtype=torch.float32).unsqueeze(0).to(device)
    b["c0"] = torch.as_tensor(chunks["c0"][idx], dtype=torch.float32).unsqueeze(0).to(device)
    return b
