"""Active-perception reward (V5 Block 13; Shin et al. Sec. III-C, p.5545; SPEC §11.3).

    r_active_t = -alpha * [ beta * (L_est_{t+1} - tau) ]_0^1,   alpha 0.1, beta 1.0, tau 0.01  [paper]
    L_est_t    = (1/6) sum_i (s_rel_t,i - s~_rel_t,i)^2                                     [paper Eq. 1]

Implementation choices [unspecified in the paper]:
  - L_est_{t+1} uses the estimate the policy produced DURING the rollout for the observation at
    t+1 (current parameters, detached). For the last rollout step, the trainer computes the
    estimate for the next observation without advancing the LSTM (buf["s_est_next_last"]).
  - If step t ends the episode (terminated OR truncated), there is no t+1 in that episode, so
    r_active_t = 0. (The obs after a done step belongs to a new episode.)
  - Added to the env reward before GAE (the trainer does this); never part of L_est's gradient.

Only meaningful when the actor has a learned estimator: vision mode with lambda_est > 0.
"""
import torch

ALPHA, BETA, TAU = 0.1, 1.0, 0.01


def est_loss_per_step(s_est, target):
    """(..., 6) -> (...): L_est per step, raw units (m, m/s), as Eq. 1."""
    return ((s_est - target) ** 2).mean(-1)


def active_perception_reward(l_next, done, alpha=ALPHA, beta=BETA, tau=TAU):
    """l_next: L_est at t+1, shape (T, B). done: 1 where step t ended its episode."""
    r = -alpha * torch.clamp(beta * (l_next - tau), 0.0, 1.0)
    return r * (1.0 - done)


class ActivePerceptionReward:
    """r_active_fn(buf) -> (T, B) for PPOTrainer. Also leaves diagnostics in buf:
    buf['l_est_next'] (T, B) and buf['r_active_frac'] (fraction of non-done steps with
    L_est_{t+1} > tau, i.e. where the term is active; SPEC §11.3 watch item)."""

    def __init__(self, alpha=ALPHA, beta=BETA, tau=TAU):
        self.alpha, self.beta, self.tau = alpha, beta, tau

    def __call__(self, buf):
        l_now = est_loss_per_step(buf["s_est"], buf["target"])                     # (T, B)
        l_last = est_loss_per_step(buf["s_est_next_last"], buf["target_next_last"])  # (B,)
        l_next = torch.cat([l_now[1:], l_last[None]], 0)
        done = torch.clamp(buf["terminated"] + buf["truncated"], max=1.0)
        r = active_perception_reward(l_next, done, self.alpha, self.beta, self.tau)
        live = done < 0.5
        buf["l_est_next"] = l_next
        buf["r_active_frac"] = float(((l_next > self.tau) & live).float().sum() / live.float().sum().clamp(min=1))
        return r
