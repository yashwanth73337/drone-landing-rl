"""
train_shin_v4.py
================

Stage V4. The first attempt at the FULL method of Shin et al. (RA-L 2026)
rather than one component of it.

    V3a   ArUco + constant velocity,        TD3, no estimator
    V3b   ArUco + LSTM trained offline,     TD3, estimator frozen
    V4    ArUco + LSTM trained JOINTLY,     recurrent PPO, privileged
          critic, active-perception reward                  <- this file

Four variables move at once and they are not separable -- see the header of
envs/ShinLanderAviary.py for why. V3a and V3b are the ablation points, which
is the structure the paper itself uses (Table IV).


THE TWO PIECES THAT ARE NOT IN STOCK RecurrentPPO
------------------------------------------------------------------------------
1. AUXILIARY STATE-ESTIMATION LOSS (paper Eq. 1)

   L_est = mean over 6 components of (s_rel - s_rel_hat)^2

   added to the PPO objective in the SAME optimisation step, not in an
   alternating pass. train() below is your installed RecurrentPPO.train()
   with two lines changed: evaluate_actions -> evaluate_actions_with_estimate,
   and the aux term added to `loss`. Everything else, including the padded-
   sequence masking, is byte-identical to the version it was copied from.

   The masking matters more than it looks. Every loss in RecurrentPPO is a
   masked mean over real (non-padded) timesteps. An aux loss averaged over
   padding would train the estimator on zeros and still converge to
   something.

2. ACTIVE-PERCEPTION REWARD (paper Section III-C)

       r_active_t = -alpha * clip(beta * (L_est_{t+1} - tau), 0, 1)

   The reward at t is conditioned on the estimation loss at t+1. That is the
   whole point: it penalises actions whose CONSEQUENCE is a worse estimate,
   which is what couples control to perception. Using L_est_t instead would
   be a different, non-causal quantity.

   This cannot be computed in the environment, because L_est is produced by
   the estimation head inside the policy. It is therefore injected into the
   rollout rewards after collection and BEFORE advantages are computed, in
   collect_rollouts() below.

   alpha=0.1, beta=1.0, tau=0.01 are the paper's values. NOTE that the paper's
   reward scale is not this project's: here the success bonus is +20 and the
   per-step time cost -0.02 (inherited from the Lander.AI rebuild). At
   convergence r_active is around -0.002 per step, roughly a tenth of the time
   cost; early in training, when L_est saturates the clip, it is -0.1, five
   times the time cost. Both ends look reasonable, but the coefficient is
   exposed as --ap-alpha because it was tuned against a different reward.


WHY THE ESTIMATES ARE REPLAYED RATHER THAN RECORDED
------------------------------------------------------------------------------
r_active needs L_est at every step of the rollout. Rather than accumulating
estimates during collection -- which means either reaching into SB3's
collection loop or carrying mutable state on the policy -- the whole rollout
is replayed through lstm_actor + est_net in ONE batched pass afterwards,
starting from the LSTM states snapshotted before collection began.

The weights have not changed between collection and replay (train() has not
run yet), so the replayed estimates are exactly the ones the policy produced.
This is checked by --selftest.


EVALUATION IS NOT v3a.evaluate_policy_full
------------------------------------------------------------------------------
That function calls `model.predict(obs, deterministic=True)` with no state
argument. For a recurrent policy that resets the hidden state on every step,
so the estimator would have no memory at all and V4 would evaluate as
broken while training fine.

evaluate_policy_recurrent() below carries the state explicitly. It also runs
the ACTOR PATH ONLY -- no critic, no privileged input -- so it exercises
exactly what would ship on the drone, and it reports the V4 estimator's own
error alongside the inherited constant-velocity baseline's on the same
frames.

Metric names and the CSV schema match V3a/V3b so the existing analysis
scripts keep working.


CHECKPOINTS
------------------------------------------------------------------------------
Per-level from the start. See V3b_REPORT.md section 5.5: the global
lexicographic rule lost seed 2's best policy by promoting onto a level it
could not fly. V4 keeps best_L<NN>.zip for every level in addition to the
global best, so that failure cannot recur here.


USAGE
------------------------------------------------------------------------------
    # structural self-test, ~30 s, no training
    python train_shin_v4.py --selftest

    # train
    python train_shin_v4.py --seed 1 --steps 600000

    # evaluate a checkpoint on an explicitly pinned task
    python train_shin_v4.py --eval --model results_shin_v4/run_*/best_L05.zip \\
        --episodes 100 --eval-seed-base 9000 \\
        --radius-min 0.11 --radius-max 0.23 --speed 0.0
"""

import argparse
import csv
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import explained_variance, obs_as_tensor

from sb3_contrib import RecurrentPPO

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.ShinLanderAviary import (                            # noqa: E402
    ShinLanderAviary, ACTOR_DIM, CRITIC_DIM, TARGET_DIM)
from policies.asymmetric_recurrent import (                    # noqa: E402
    AsymmetricRecurrentPolicy)
import train_aruco_v3a as v3a                                  # noqa: E402


# Paper values, Section III-C.
AP_ALPHA = 0.1
AP_BETA = 1.0
AP_TAU = 0.01


def make_kwargs(radius_min, radius_max, speed, frame='world'):
    """V3a's env kwargs plus the V4 frame setting.

    Built from v3a.make_env_kwargs so camera, marker, FOV, tilt limit and
    episode length cannot diverge from the stages V4 will be compared
    against.
    """
    kw = v3a.make_env_kwargs(radius_min, radius_max, speed)
    kw['frame'] = frame
    kw['verbose_source'] = False
    return kw


# ===========================================================================
# ALGORITHM
# ===========================================================================

class ShinRecurrentPPO(RecurrentPPO):
    """RecurrentPPO + auxiliary estimation loss + active-perception reward."""

    def __init__(self, *args, aux_coef: float = 1.0,
                 ap_alpha: float = AP_ALPHA, ap_beta: float = AP_BETA,
                 ap_tau: float = AP_TAU, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_coef = float(aux_coef)
        self.ap_alpha = float(ap_alpha)
        self.ap_beta = float(ap_beta)
        self.ap_tau = float(ap_tau)
        self._last_ap_mean = 0.0
        self._last_lest_mean = 0.0

    # ---------------------------------------------------------------
    # ROLLOUT: inject the active-perception reward before advantages
    # ---------------------------------------------------------------

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        # States at the START of collection. super() does
        # `lstm_states = deepcopy(self._last_lstm_states)` as its first act,
        # so this is the same tensor the rollout actually began from.
        start_states = deepcopy(self._last_lstm_states)

        ok = super().collect_rollouts(
            env, callback, rollout_buffer, n_rollout_steps)
        if not ok:
            return False

        if self.ap_alpha == 0.0:
            return True

        l_est = self._replay_estimation_loss(rollout_buffer, start_states)

        # r_active_t uses L_est at t+1.
        l_next = th.roll(l_est, shifts=-1, dims=0)
        r_active = -self.ap_alpha * th.clamp(
            self.ap_beta * (l_next - self.ap_tau), 0.0, 1.0)

        # Two places where t+1 is not a valid continuation of t:
        #   - the last step of the rollout (no t+1 collected)
        #   - any step whose successor begins a new episode
        # Zeroing is the honest choice; carrying the value across an episode
        # boundary would attribute a fresh episode's estimation error to the
        # previous episode's final action.
        eps = th.as_tensor(rollout_buffer.episode_starts,
                           device=self.device, dtype=th.float32)
        next_is_start = th.roll(eps, shifts=-1, dims=0)
        next_is_start[-1] = 1.0
        r_active = r_active * (1.0 - next_is_start)

        rollout_buffer.rewards = (
            rollout_buffer.rewards + r_active.cpu().numpy())

        # Advantages were computed by super() from the un-shaped rewards, so
        # they must be recomputed. This mirrors super()'s own final block
        # exactly -- same last_values, same dones.
        with th.no_grad():
            episode_starts = th.tensor(self._last_episode_starts,
                                       dtype=th.float32, device=self.device)
            last_values = self.policy.predict_values(
                obs_as_tensor(self._last_obs, self.device),
                self._last_lstm_states.vf, episode_starts)
        rollout_buffer.compute_returns_and_advantage(
            last_values=last_values, dones=self._last_episode_starts)

        self._last_ap_mean = float(r_active.mean().item())
        self._last_lest_mean = float(l_est.mean().item())
        return True

    def _replay_estimation_loss(self, rollout_buffer, start_states):
        """Per-step L_est for the whole rollout, shape (n_steps, n_envs).

        One batched pass through lstm_actor + est_net from the pre-rollout
        states. See the module docstring for why this is a replay and not a
        recording.

        _process_sequence expects an ENV-MAJOR flat batch -- it reshapes to
        (n_seq, seq_len, dim) with n_seq = n_envs -- while the rollout buffer
        stores time-major (n_steps, n_envs, dim). The permutes below are that
        conversion and back; getting them wrong silently interleaves
        different environments' trajectories into one sequence.
        """
        obs_a = th.as_tensor(rollout_buffer.observations['actor'],
                             device=self.device).float()      # (T, N, A)
        tgt = th.as_tensor(rollout_buffer.observations['target'],
                           device=self.device).float()        # (T, N, 6)
        eps = th.as_tensor(rollout_buffer.episode_starts,
                           device=self.device).float()        # (T, N)
        T, N = eps.shape

        feat = obs_a.permute(1, 0, 2).reshape(N * T, ACTOR_DIM)
        epsf = eps.permute(1, 0).reshape(N * T)

        with th.no_grad():
            lstm_out, _ = self.policy._process_sequence(
                feat, start_states.pi, epsf, self.policy.lstm_actor)
            est = self.policy.est_net(th.cat([lstm_out, feat], dim=1))

        est = est.reshape(N, T, TARGET_DIM).permute(1, 0, 2)  # (T, N, 6)
        return ((est - tgt) ** 2).mean(dim=-1)                # (T, N)

    # ---------------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------------

    def train(self) -> None:
        """RecurrentPPO.train() with the auxiliary estimation loss added.

        Copied verbatim from sb3_contrib 2.9.0 and changed in exactly two
        places, both marked `V4:`. If sb3-contrib is upgraded, re-diff this
        against the installed source before trusting a run.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        aux_losses = []                                        # V4:

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                # V4: evaluate_actions_with_estimate returns the extra head.
                values, log_prob, entropy, est = \
                    self.policy.evaluate_actions_with_estimate(
                        rollout_data.observations,
                        actions,
                        rollout_data.lstm_states,
                        rollout_data.episode_starts,
                    )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (
                        advantages[mask].std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(
                    th.min(policy_loss_1, policy_loss_2)[mask])

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf, clip_range_vf)
                value_loss = th.mean(
                    ((rollout_data.returns - values_pred) ** 2)[mask])
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])
                entropy_losses.append(entropy_loss.item())

                # V4: auxiliary state-estimation loss, Eq. 1. Masked the same
                # way every other term here is -- padded timesteps must not
                # enter the mean.
                target = rollout_data.observations['target']
                aux_loss = th.mean(
                    ((est - target) ** 2).mean(dim=-1)[mask])
                aux_losses.append(aux_loss.item())

                loss = (policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.aux_coef * aux_loss)            # V4:

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        ((th.exp(log_ratio) - 1) - log_ratio)[mask]
                    ).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and \
                        approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to "
                              f"reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                            self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item())

        # V4: estimator and active-perception diagnostics.
        self.logger.record("v4/aux_loss", float(np.mean(aux_losses)))
        self.logger.record("v4/l_est_rollout", self._last_lest_mean)
        self.logger.record("v4/r_active_mean", self._last_ap_mean)

        self.logger.record("train/n_updates", self._n_updates,
                           exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


# ===========================================================================
# EVALUATION -- recurrent, actor path only
# ===========================================================================

def evaluate_policy_recurrent(model, env_kwargs, n_episodes, seed_base,
                              collect_perception=True):
    """Deterministic evaluation carrying the recurrent state explicitly.

    Runs the ACTOR PATH ONLY: features -> lstm_actor -> mlp_extractor ->
    action, plus the estimation head. The critic, its LSTM and the privileged
    observation are never touched, so this is exactly the computation that
    would run on the drone.

    Metric names match v3a.evaluate_policy_full so the same analysis scripts
    apply. Two are added: v4_pos_err_m / v4_vel_err_ms, the V4 estimator's own
    error, alongside pos_err_m / vel_err_ms which remain the inherited
    constant-velocity baseline's on the same frames.
    """
    from collections import Counter

    policy = model.policy
    device = policy.device
    env = ShinLanderAviary(**env_kwargs)

    successes = 0
    precisions, returns = [], []
    reasons = Counter()
    near_miss = 0
    blind_fracs, longest_blinds, first_dets = [], [], []
    cv_pos, cv_vel = [], []
    v4_pos, v4_vel = [], []

    try:
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed_base + ep)
            # _process_sequence needs a real (h, c) pair; unlike
            # model.predict() it will not build one from None. Zeros is what
            # SB3 would create anyway, and ep_start=1 on the first step
            # zeroes them again regardless.
            n_l = policy.lstm_actor.num_layers
            n_h = policy.lstm_actor.hidden_size
            states = (th.zeros(n_l, 1, n_h, device=device),
                      th.zeros(n_l, 1, n_h, device=device))
            ep_start = th.ones(1, dtype=th.float32, device=device)
            ep_return = 0.0
            low_run = low_run_max = 0
            ep_cv_p, ep_cv_v, ep_v4_p, ep_v4_v = [], [], [], []

            while True:
                with th.no_grad():
                    obs_t, _ = policy.obs_to_tensor(obs)
                    feats = policy.extract_features(obs_t)
                    pi_feat, _ = policy._split_features(feats)
                    lstm_out, states = policy._process_sequence(
                        pi_feat, states, ep_start, policy.lstm_actor)
                    est = policy.est_net(
                        th.cat([lstm_out, pi_feat], dim=1))[0].cpu().numpy()
                    latent_pi = policy.mlp_extractor.forward_actor(lstm_out)
                    dist = policy._get_action_dist_from_latent(latent_pi)
                    action = dist.get_actions(
                        deterministic=True)[0].cpu().numpy()

                ep_start = th.zeros(1, dtype=th.float32, device=device)

                # obs['target'] is the ground-truth s_rel for the frame the
                # estimate was produced from. Read here for METRICS only --
                # it never reached the actor, which consumed obs['actor'].
                tgt = np.asarray(obs['target'], dtype=np.float64)
                ep_v4_p.append(float(np.linalg.norm(est[:3] - tgt[:3])))
                ep_v4_v.append(float(np.linalg.norm(est[3:] - tgt[3:])))

                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)

                h = float(info["height_above_pad"])
                if h < 0.03:
                    low_run += 1
                    low_run_max = max(low_run_max, low_run)
                else:
                    low_run = 0

                if collect_perception and info.get("est_detected"):
                    ep_cv_p.append(info["est_err_pos_norm"])
                    ep_cv_v.append(info["est_err_vel_norm"])

                if terminated or truncated:
                    break

            returns.append(ep_return)
            summ = env.episodeEstimatorSummary()
            blind_fracs.append(summ["blind_fraction"])
            longest_blinds.append(summ["longest_blind_run"])
            first_dets.append(summ["steps_before_first_detection"])

            if info.get("success", False):
                successes += 1
                precisions.append(float(info["d_target"]) * 100.0)
            else:
                reasons[info.get("truncation_reason")
                        or "terminated_no_reason"] += 1
                if low_run_max >= 10:
                    near_miss += 1

            if ep_cv_p:
                cv_pos.append(float(np.mean(ep_cv_p)))
                cv_vel.append(float(np.mean(ep_cv_v)))
            if ep_v4_p:
                v4_pos.append(float(np.mean(ep_v4_p)))
                v4_vel.append(float(np.mean(ep_v4_v)))
    finally:
        env.close()

    def m(x):
        return float(np.mean(x)) if x else float("nan")

    return {
        "episodes": n_episodes,
        "successes": successes,
        "success_rate": 100.0 * successes / n_episodes,
        "precision_mean_cm": (float(np.mean(precisions)) if precisions
                              else float("inf")),
        "precision_std_cm": (float(np.std(precisions)) if precisions
                             else float("nan")),
        "mean_return": float(np.mean(returns)),
        "near_miss": near_miss,
        "reasons": dict(reasons),
        "blind_fraction": float(np.mean(blind_fracs)),
        "longest_blind": float(np.mean(longest_blinds)),
        "steps_before_first_detection": float(np.mean(first_dets)),
        "pos_err_m": m(cv_pos),        # inherited CV baseline
        "vel_err_ms": m(cv_vel),
        "v4_pos_err_m": m(v4_pos),     # V4's own estimator
        "v4_vel_err_ms": m(v4_vel),
    }


# ===========================================================================
# CALLBACK -- curriculum, global best, AND per-level bests
# ===========================================================================

class V4CurriculumCallback(BaseCallback):
    """v3a's curriculum rule with a recurrent-aware evaluator.

    The level table, promotion threshold, cadence, episode count and seed
    base are imported from train_aruco_v3a so they cannot drift. Only the
    evaluator differs, and it has to: see the module docstring.

    Per-level checkpoints are present from the first run, rather than being
    retrofitted after a promotion destroys a good policy as happened in V3b
    (V3b_REPORT.md section 5.5).
    """

    LEVELS = v3a.CURRICULUM_LEVELS

    def __init__(self, save_dir, curriculum_state, frame='world',
                 eval_every_timesteps=v3a.SUCCESS_EVAL_EVERY_TIMESTEPS,
                 n_eval_episodes=v3a.SUCCESS_EVAL_EPISODES,
                 seed_base=v3a.SUCCESS_EVAL_SEED_BASE,
                 promote_threshold=v3a.PROMOTE_THRESHOLD,
                 radius_min=v3a.RADIUS_MIN, verbose=1):
        super().__init__(verbose=verbose)
        self.save_dir = save_dir
        self.state = curriculum_state
        self.frame = frame
        self.eval_every_timesteps = int(eval_every_timesteps)
        self.n_eval_episodes = int(n_eval_episodes)
        self.seed_base = int(seed_base)
        self.promote_threshold = promote_threshold
        self.radius_min = radius_min

        self.level = 0
        self.last_eval_timestep = 0
        self.best_level = -1
        self.best_success_rate = -1.0
        self.best_precision = float("inf")
        self.best_by_level = {}
        self.best_any_rate = -1.0
        self.best_any_level = -1
        self.csv_path = os.path.join(save_dir, "success_evaluations.csv")

    def _apply_level(self, level):
        hi, speed = self.LEVELS[level]
        self.training_env.env_method('set_spawn_radius', self.radius_min, hi)
        self.training_env.env_method('set_platform_speed', speed)
        self.state.radius_min = self.radius_min
        self.state.radius_max = hi
        self.state.platform_speed = speed
        self.state.level = level
        if self.verbose:
            print(f"\n[curriculum] -> {self.state.describe()}\n")

    def _on_training_start(self):
        self._apply_level(self.level)

    def _write_row(self, r):
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r.keys()))
            if new:
                w.writeheader()
            w.writerow(r)

    def _write_level_manifest(self):
        path = os.path.join(self.save_dir, "best_per_level.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["level", "spawn_radius_min", "spawn_radius_max",
                        "platform_speed_range", "checkpoint",
                        "success_rate", "precision_cm"])
            for lvl in sorted(self.best_by_level):
                rate, neg_prec = self.best_by_level[lvl]
                hi, speed = self.LEVELS[lvl]
                prec = ("" if neg_prec == -float("inf")
                        else round(-neg_prec, 4))
                w.writerow([lvl + 1, self.radius_min, hi, speed,
                            f"best_L{lvl + 1:02d}.zip",
                            round(rate, 3), prec])

    def _on_step(self):
        if (self.num_timesteps - self.last_eval_timestep
                < self.eval_every_timesteps):
            return True
        self.last_eval_timestep = self.num_timesteps

        snapshot = {
            "level": self.state.level,
            "radius_min": self.state.radius_min,
            "radius_max": self.state.radius_max,
            "platform_speed": self.state.platform_speed,
        }
        kw = make_kwargs(snapshot["radius_min"], snapshot["radius_max"],
                         snapshot["platform_speed"], self.frame)

        res = evaluate_policy_recurrent(
            self.model, kw, self.n_eval_episodes, self.seed_base)
        reasons = res["reasons"]

        record = {
            "timestep": int(self.num_timesteps),
            "level": snapshot["level"] + 1,
            "spawn_radius_min": snapshot["radius_min"],
            "spawn_radius_max": snapshot["radius_max"],
            "platform_speed_range": snapshot["platform_speed"],
            "successes": res["successes"],
            "episodes": res["episodes"],
            "success_rate": round(res["success_rate"], 3),
            "mean_success_distance_cm": (
                "" if not np.isfinite(res["precision_mean_cm"])
                else round(res["precision_mean_cm"], 4)),
            "mean_return": round(res["mean_return"], 6),
            "near_miss": res["near_miss"],
            "timeout": reasons.get("timeout", 0),
            "below_platform": reasons.get("below_platform", 0),
            "tilt_bound": reasons.get("tilt_bound", 0),
            "position_bound": reasons.get("position_bound", 0),
            "terminated_no_reason": reasons.get("terminated_no_reason", 0),
            "blind_fraction": round(res["blind_fraction"], 4),
            "v4_pos_err_m": round(res["v4_pos_err_m"], 5),
            "v4_vel_err_ms": round(res["v4_vel_err_ms"], 5),
            "cv_pos_err_m": round(res["pos_err_m"], 5),
            "cv_vel_err_ms": round(res["vel_err_ms"], 5),
        }
        self._write_row(record)
        self.logger.record("success_eval/success_rate", res["success_rate"])
        self.logger.record("success_eval/level", snapshot["level"] + 1)
        self.logger.record("v4/eval_pos_err_m", res["v4_pos_err_m"])
        self.logger.record("v4/eval_vel_err_ms", res["v4_vel_err_ms"])

        if self.verbose:
            print(f"\n  [eval] {self.num_timesteps} steps | "
                  f"level {snapshot['level'] + 1}/{len(self.LEVELS)}  "
                  f"radius {snapshot['radius_min']:.2f}-"
                  f"{snapshot['radius_max']:.2f} m  "
                  f"speed {snapshot['platform_speed']:.2f} m/s")
            print(f"         {res['successes']}/{res['episodes']} "
                  f"({res['success_rate']:.1f}%)  "
                  f"near-miss {record['near_miss']}  "
                  f"timeout {record['timeout']}  "
                  f"below_platform {record['below_platform']}")
            print(f"         est err  V4 {res['v4_pos_err_m']:.4f} m / "
                  f"{res['v4_vel_err_ms']:.3f} m/s   "
                  f"CV {res['pos_err_m']:.4f} m / "
                  f"{res['vel_err_ms']:.3f} m/s   "
                  f"blind {res['blind_fraction'] * 100:.1f}%")

        # global best -- same lexicographic rule as V3a/V3b, kept for
        # comparability, with the same loud warning when level dominance
        # produces a regression
        key = (snapshot["level"], res["success_rate"],
               -res["precision_mean_cm"] if res["successes"]
               else -float("inf"))
        best = (self.best_level, self.best_success_rate, -self.best_precision)
        if res["successes"] > 0 and key > best:
            regressed = (self.best_any_rate >= 0.0
                         and res["success_rate"] < self.best_any_rate)
            self.best_level = snapshot["level"]
            self.best_success_rate = res["success_rate"]
            self.best_precision = res["precision_mean_cm"]
            self.model.save(os.path.join(self.save_dir, "best_success_model"))
            if self.verbose:
                print(f"         NEW BEST saved at level "
                      f"{snapshot['level'] + 1} "
                      f"({res['success_rate']:.1f}%)")
                if regressed:
                    print(f"         WARNING: below the best at any level "
                          f"({self.best_any_rate:.1f}% at level "
                          f"{self.best_any_level + 1}). Use best_L"
                          f"{self.best_any_level + 1:02d}.zip.")

        # per-level best -- nothing can overwrite across levels
        lvl = snapshot["level"]
        lvl_key = (res["success_rate"],
                   -res["precision_mean_cm"] if res["successes"]
                   else -float("inf"))
        if (res["successes"] > 0
                and lvl_key > self.best_by_level.get(
                    lvl, (-1.0, -float("inf")))):
            self.best_by_level[lvl] = lvl_key
            self.model.save(
                os.path.join(self.save_dir, f"best_L{lvl + 1:02d}"))
            self._write_level_manifest()
            if self.verbose:
                print(f"         per-level best: best_L{lvl + 1:02d}.zip "
                      f"({res['success_rate']:.1f}%)")

        if res["successes"] > 0 and res["success_rate"] > self.best_any_rate:
            self.best_any_rate = res["success_rate"]
            self.best_any_level = snapshot["level"]

        if self.level >= len(self.LEVELS) - 1:
            if self.verbose:
                print("         [final level]")
            return True
        if res["success_rate"] >= self.promote_threshold * 100.0:
            self.level += 1
            self._apply_level(self.level)
        return True


# ===========================================================================
# SELF-TEST -- the privilege boundary, checked rather than asserted in prose
# ===========================================================================

def selftest(args):
    print('=' * 74)
    print('  V4 STRUCTURAL SELF-TEST')
    print('=' * 74)

    kw = make_kwargs(0.11, 0.14, 0.0, args.frame)
    env = ShinLanderAviary(**kw)
    obs, info = env.reset(seed=1234)

    assert set(obs) == {'actor', 'critic', 'target'}, set(obs)
    assert obs['actor'].shape == (ACTOR_DIM,)
    assert obs['critic'].shape == (CRITIC_DIM,)
    assert obs['target'].shape == (TARGET_DIM,)
    print(f'  [OK  ] observation keys and shapes '
          f'({ACTOR_DIM}/{CRITIC_DIM}/{TARGET_DIM})')

    assert np.all(np.abs(obs['actor']) <= 1.0 + 1e-6), 'actor not normalised'
    assert np.all(np.abs(obs['critic']) <= 1.0 + 1e-6), 'critic not normalised'
    print('  [OK  ] actor and critic within [-1, 1]')
    env.close()

    venv = make_vec_env(ShinLanderAviary, n_envs=2, seed=0, env_kwargs=kw)
    model = ShinRecurrentPPO(
        AsymmetricRecurrentPolicy, venv, n_steps=16, batch_size=16,
        verbose=0, seed=0,
        policy_kwargs=dict(lstm_hidden_size=32, est_hidden=32,
                           net_arch=dict(pi=[32], vf=[32])))
    policy = model.policy
    print('  [OK  ] policy constructed')

    names = set(n for n, _ in policy.named_parameters())
    assert any(n.startswith('est_net') for n in names), 'est_net not registered'
    opt_params = {id(p) for g in policy.optimizer.param_groups
                  for p in g['params']}
    missing = [n for n, p in policy.named_parameters()
               if id(p) not in opt_params]
    assert not missing, f'parameters missing from optimizer: {missing}'
    print('  [OK  ] every parameter is in the optimizer '
          '(LSTM rebuild did not orphan it)')

    assert policy.lstm_actor.input_size == ACTOR_DIM, policy.lstm_actor.input_size
    assert policy.lstm_critic.input_size == CRITIC_DIM
    print(f'  [OK  ] lstm_actor takes {ACTOR_DIM}, '
          f'lstm_critic takes {CRITIC_DIM}')

    # ---- THE PRIVILEGE TEST ----------------------------------------
    obs_b = {k: np.stack([v, v]) for k, v in obs.items()}
    obs_t, _ = policy.obs_to_tensor(obs_b)
    states = None
    eps = th.zeros(2, dtype=th.float32, device=policy.device)

    def zero_states(lstm):
        z = th.zeros(lstm.num_layers, 2, lstm.hidden_size,
                     device=policy.device)
        return (z, z.clone())

    def act_and_value(o):
        """Actor and critic paths from identical zero states.

        Both start from zeroed hidden state so the comparison below isolates
        the observation: any difference in output comes from the input, not
        from recurrent history.
        """
        with th.no_grad():
            f = policy.extract_features(o)
            pi_f, vf_f = policy._split_features(f)

            lo, _ = policy._process_sequence(
                pi_f, zero_states(policy.lstm_actor), eps, policy.lstm_actor)
            a = policy._get_action_dist_from_latent(
                policy.mlp_extractor.forward_actor(lo)).mode()

            lv, _ = policy._process_sequence(
                vf_f, zero_states(policy.lstm_critic), eps,
                policy.lstm_critic)
            v = policy.value_net(policy.mlp_extractor.forward_critic(lv))
        return a.cpu().numpy(), v.cpu().numpy()

    a0, v0 = act_and_value(obs_t)

    corrupted = {k: t.clone() for k, t in obs_t.items()}
    corrupted['critic'] = th.randn_like(corrupted['critic'])
    a1, v1 = act_and_value(corrupted)
    assert np.allclose(a0, a1, atol=1e-6), (
        'ACTION CHANGED when obs[critic] was corrupted -- the actor is '
        'reading privileged state. This invalidates the entire stage.')
    assert not np.allclose(v0, v1, atol=1e-6), (
        'VALUE did not change when obs[critic] was corrupted -- the critic '
        'is ignoring its privileged input, so the asymmetry does nothing.')
    print('  [OK  ] corrupting obs[critic] changes the VALUE but not the '
          'ACTION')

    corrupted = {k: t.clone() for k, t in obs_t.items()}
    corrupted['target'] = th.randn_like(corrupted['target'])
    a2, v2 = act_and_value(corrupted)
    assert np.allclose(a0, a2, atol=1e-6) and np.allclose(v0, v2, atol=1e-6), (
        'obs[target] reached a network. It is a LABEL and must not be read.')
    print('  [OK  ] corrupting obs[target] changes nothing (label only)')

    # ---- short learn, exercising both new code paths ----------------
    model.learn(total_timesteps=64)
    print('  [OK  ] learn() ran: rollout replay, active-perception reward, '
          'aux loss')
    print(f'         L_est (rollout mean) {model._last_lest_mean:.5f}   '
          f'r_active mean {model._last_ap_mean:.6f}')
    venv.close()

    print('=' * 74)
    print('  ALL CHECKS PASSED')
    print('=' * 74)


# ===========================================================================
# TRAIN / EVAL / CLI
# ===========================================================================

def train(args):
    if not v3a.verify_curriculum_port():
        print('\nRefusing to train on a diverged curriculum port.')
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    run_dir = os.path.join(args.outdir, time.strftime('run_%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)

    print(f'\n{"=" * 74}')
    print('  V4 -- RECURRENT PPO, PRIVILEGED CRITIC, JOINT ESTIMATION,')
    print('        ACTIVE PERCEPTION')
    print(f'{"=" * 74}')
    print(f'  seed            : {args.seed}')
    print(f'  total timesteps : {args.steps}')
    print(f'  frame           : {args.frame}')
    print(f'  aux coef        : {args.aux_coef}')
    print(f'  active percep.  : alpha {args.ap_alpha}  beta {args.ap_beta}  '
          f'tau {args.ap_tau}')
    print(f'  lstm            : {args.lstm_hidden} x {args.lstm_layers}')
    print(f'  output          : {run_dir}')
    print('  FOUR variables move relative to V3b: joint training, PPO,')
    print('  privileged critic, active-perception reward. V3a and V3b are')
    print('  the ablation points.')
    print(f'{"=" * 74}\n')

    state = v3a.CurriculumState()
    kw = make_kwargs(state.radius_min, state.radius_max,
                     state.platform_speed, args.frame)

    train_env = make_vec_env(ShinLanderAviary, n_envs=v3a.N_TRAIN_ENVS,
                             seed=args.seed, env_kwargs=kw)

    model = ShinRecurrentPPO(
        AsymmetricRecurrentPolicy, train_env,
        verbose=1, seed=args.seed,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        aux_coef=args.aux_coef,
        ap_alpha=args.ap_alpha, ap_beta=args.ap_beta, ap_tau=args.ap_tau,
        policy_kwargs=dict(
            lstm_hidden_size=args.lstm_hidden,
            n_lstm_layers=args.lstm_layers,
            est_hidden=args.est_hidden,
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        ),
        tensorboard_log=os.path.join(run_dir, 'tb'),
    )

    callbacks = CallbackList([
        V4CurriculumCallback(run_dir, state, frame=args.frame),
    ])

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks)
    elapsed = time.time() - t0

    model.save(os.path.join(run_dir, 'final_model'))
    print(f'\n  done in {elapsed / 3600:.2f} h. saved to {run_dir}')
    print(f'    cat {run_dir}/best_per_level.csv')
    return run_dir


def evaluate(args):
    radius_min = (v3a.FINAL_TASK_KWARGS['spawn_radius_min']
                  if args.radius_min is None else args.radius_min)
    radius_max = (v3a.FINAL_TASK_KWARGS['spawn_radius_max']
                  if args.radius_max is None else args.radius_max)
    speed = (v3a.FINAL_TASK_KWARGS['platform_speed_range']
             if args.speed is None else args.speed)

    kw = make_kwargs(radius_min, radius_max, speed, args.frame)
    model = ShinRecurrentPPO.load(args.model)

    print('=' * 74)
    print('  V4 EVALUATION')
    print('=' * 74)
    print(f'  model    : {args.model}')
    print(f'  task     : radius {radius_min}-{radius_max} m, '
          f'speed {speed} m/s, frame {args.frame}')
    print(f'  episodes : {args.episodes}  (seeds {args.eval_seed_base}..)')
    print()

    r = evaluate_policy_recurrent(model, kw, args.episodes,
                                  args.eval_seed_base)

    print(f'  success              : {r["successes"]}/{r["episodes"]} '
          f'({r["success_rate"]:.1f}%)')
    if np.isfinite(r['precision_mean_cm']):
        print(f'  precision            : {r["precision_mean_cm"]:.3f} +/- '
              f'{r["precision_std_cm"]:.3f} cm')
    print(f'  near-miss            : {r["near_miss"]}')
    print(f'  failure reasons      : {r["reasons"]}')
    print(f'  blind fraction       : {r["blind_fraction"] * 100:.1f}%')
    print(f'  longest blind run    : {r["longest_blind"]:.1f} steps')
    print(f'  V4 est error         : {r["v4_pos_err_m"]:.5f} m  /  '
          f'{r["v4_vel_err_ms"]:.5f} m/s')
    print(f'  CV baseline error    : {r["pos_err_m"]:.5f} m  /  '
          f'{r["vel_err_ms"]:.5f} m/s   (same frames)')
    print('=' * 74)
    print(f'  Task was radius {radius_min}-{radius_max} m, speed {speed} m/s.')
    print('  Report this number WITH those settings attached.')
    print('=' * 74)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--outdir', default='results_shin_v4')
    ap.add_argument('--frame', default='world', choices=['world', 'body'],
                    help="'body' is the paper's configuration; 'world' keeps "
                         "the estimator error comparable with V3a/V3b")

    ap.add_argument('--selftest', action='store_true',
                    help='structural checks incl. the privilege boundary')
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--model', type=str, default=None)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--eval-seed-base', type=int, default=9000)
    ap.add_argument('--radius-min', type=float, default=None)
    ap.add_argument('--radius-max', type=float, default=None)
    ap.add_argument('--speed', type=float, default=None)

    ap.add_argument('--aux-coef', type=float, default=1.0)
    ap.add_argument('--ap-alpha', type=float, default=AP_ALPHA)
    ap.add_argument('--ap-beta', type=float, default=AP_BETA)
    ap.add_argument('--ap-tau', type=float, default=AP_TAU)

    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--n-steps', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--n-epochs', type=int, default=10)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--gae-lambda', type=float, default=0.95)
    ap.add_argument('--clip-range', type=float, default=0.2)
    ap.add_argument('--ent-coef', type=float, default=0.0)
    ap.add_argument('--lstm-hidden', type=int, default=128)
    ap.add_argument('--lstm-layers', type=int, default=1)
    ap.add_argument('--est-hidden', type=int, default=256)

    args = ap.parse_args()

    if args.selftest:
        selftest(args)
    elif args.eval:
        if not args.model:
            print('--eval requires --model')
            sys.exit(1)
        evaluate(args)
    else:
        if args.steps is None:
            print('--steps is required. Use 600000 to match V3a/V3b.')
            sys.exit(1)
        train(args)


if __name__ == '__main__':
    main()
