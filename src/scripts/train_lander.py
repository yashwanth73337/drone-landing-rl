"""
train_lander.py
================

Training and evaluation for LanderAIAviary, using TD3. This version supports curriculum training, starting with a stationary platform (stationary platform) while retaining the
algorithm and hyperparameters reported in Peter et al., "Lander.AI:
DRL-based Autonomous Drone Landing on Moving 3D Surface in the Presence of
Aerodynamic Disturbances" (ICUAS 2024), Table I.

Usage:
    python train_lander.py                      # train
    python train_lander.py --gui                # watch one landing
    python train_lander.py --eval                # 100 episodes, metrics
    python train_lander.py --eval --note "..."   # tag the CSV row


HYPERPARAMETERS -- PAPER FACT (Table I) vs. THIS FILE'S CHOICES
------------------------------------------------------------------------------
    Algorithm            TD3                          PAPER FACT
    Policy                MlpPolicy                    PAPER FACT
    Learning rate         0.0001                       PAPER FACT
    Buffer size            1,000,000                    PAPER FACT
    Batch size             100                          PAPER FACT
    Activation             ReLU                         PAPER FACT
    Optimizer              Adam                         PAPER FACT (SB3 TD3
                                                          default -- nothing
                                                          to configure)
    Network architecture   FC512 x2 -> FC256 -> FC128    PAPER FACT for the
                                                          ACTOR (Eq. 4). The
                                                          paper does not
                                                          describe a separate
                                                          critic architecture
                                                          -- the SAME
                                                          architecture is
                                                          applied to the
                                                          critic here too,
                                                          as a documented
                                                          assumption, not a
                                                          paper-stated fact.
    learning_starts=100    "Training begins after       PAPER FACT
                            100 steps"
    Episode duration        20 s                         PAPER FACT (also
                                                          baked into
                                                          LanderAIAviary's
                                                          EPISODE_LEN_SEC)
    Total training steps   5,000,000 initial, up to     PAPER FACT, but NOT
                            35,000,000 extended          USED HERE -- not
                                                          feasible in a single
                                                          session's remaining
                                                          time. --steps
                                                          defaults to a much
                                                          smaller number (see
                                                          CLI default below);
                                                          THIS IS A DOCUMENTED
                                                          DEVIATION driven by
                                                          time, not a claim
                                                          that this matches
                                                          the paper's own
                                                          training budget.

TD3 internals not stated in the paper remain at SB3 defaults except
gradient_steps=-1. With four parallel training environments this keeps the
number of gradient updates aligned with the number of collected transitions;
it is an implementation choice, not a paper-stated hyperparameter.
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
from collections import Counter

import torch
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import get_schedule_fn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.LanderAIAviary import LanderAIAviary


RESULTS_LOG = 'results_lander_log.csv'

# Eq. 4 of the paper (PAPER FACT for the actor; applied to the critic too
# as a documented assumption -- see module docstring).
NET_ARCH = [512, 512, 256, 128]

# ---------------------------------------------------------------------------
# CURRENT CURRICULUM ENVIRONMENT
# ---------------------------------------------------------------------------
# The platform is stationary. Its XY spawn distribution is controlled by
# LanderAIAviary._samplePlatformMotion(), so this same training script works
# for both the centered diagnostic and the randomized-XY curriculum stage.
#
# IMPORTANT: training, reward-evaluation, success-evaluation, and GUI playback
# must all use the SAME environment settings.
ENV_KWARGS = {
    "platform_speed_range": 0.0,
    "tilt_limit": 1.0,
}

N_TRAIN_ENVS = 4
REWARD_EVAL_EVERY_TIMESTEPS = 5_000
SUCCESS_EVAL_EVERY_TIMESTEPS = 5_000
SUCCESS_EVAL_EPISODES = 20

# Curriculum fine-tuning settings -- IMPLEMENTATION CHOICES.
#
# Loading an SB3 model ZIP restores the networks/optimizers, but not the
# replay buffer. Updating immediately from a nearly empty new replay buffer
# caused catastrophic forgetting in our Stage-2A experiment:
# 72% zero-shot success -> 0% final success.
#
# Fine-tuning therefore:
#   1) preserves/evaluates the incoming policy,
#   2) collects policy-generated experience with NO gradient updates first,
#   3) uses smaller exploration noise,
#   4) uses a lower learning rate,
#   5) uses fewer gradient updates per 4 collected VecEnv transitions.
FINETUNE_PREFILL_STEPS = 5_000
FINETUNE_NOISE_SIGMA = 0.05
FINETUNE_LEARNING_RATE = 1e-5
FINETUNE_GRADIENT_STEPS = 1


def _evaluate_success_rate(model,
                           env_kwargs,
                           n_eval_episodes=20,
                           seed_base=10_000):
    """Evaluate deterministic landing success on fixed seeds."""
    env = LanderAIAviary(**env_kwargs)

    successes = 0
    success_distances_cm = []
    episode_returns = []

    try:
        for ep in range(n_eval_episodes):
            obs, info = env.reset(seed=seed_base + ep)
            ep_return = 0.0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)

                if terminated or truncated:
                    break

            episode_returns.append(ep_return)

            if info.get("success", False):
                successes += 1
                success_distances_cm.append(
                    float(info["d_target"]) * 100.0
                )
    finally:
        env.close()

    success_rate = 100.0 * successes / n_eval_episodes
    mean_return = float(np.mean(episode_returns))

    if success_distances_cm:
        mean_success_distance = float(
            np.mean(success_distances_cm)
        )
    else:
        mean_success_distance = float("inf")

    return (
        success_rate,
        mean_success_distance,
        mean_return,
        successes,
    )


class SuccessRateCheckpointCallback(BaseCallback):
    """Periodically evaluate deterministic landing success and save the model
    with the highest success rate.

    EvalCallback's "best_model.zip" is selected by mean episode reward. That
    is useful, but reward and true landing success can diverge. This callback
    therefore keeps a separate "best_success_model.zip".

    Evaluation uses the same fixed seeds at every checkpoint so checkpoint
    comparisons are reproducible rather than being dominated by a different
    random set of platform positions each time.
    """

    def __init__(self,
                 save_dir,
                 env_kwargs,
                 eval_every_timesteps=5_000,
                 n_eval_episodes=20,
                 seed_base=10_000,
                 initial_best_success_rate=-1.0,
                 initial_best_mean_success_distance=float("inf"),
                 curriculum_state=None,
                 verbose=1):
        super().__init__(verbose=verbose)
        self.save_dir = save_dir
        self.env_kwargs = dict(env_kwargs)
        # When a curriculum is running, evaluate at the level currently being
        # trained rather than at whatever radius ENV_KWARGS happens to imply.
        # Without this the checkpoint was selected on the default 0.11-0.14 m
        # task for the entire run.
        self.curriculum_state = curriculum_state
        self.eval_every_timesteps = int(eval_every_timesteps)
        self.n_eval_episodes = int(n_eval_episodes)
        self.seed_base = int(seed_base)

        self.last_eval_timestep = 0
        self.best_level = -1
        self.best_success_rate = float(initial_best_success_rate)
        self.best_mean_success_distance = float(
            initial_best_mean_success_distance
        )

        self.csv_path = os.path.join(
            self.save_dir, "success_evaluations.csv"
        )

    def _current_env_kwargs(self):
        if self.curriculum_state is None:
            return self.env_kwargs
        return self.curriculum_state.as_env_kwargs(self.env_kwargs)

    def _evaluate_success(self):
        return _evaluate_success_rate(
            self.model,
            self._current_env_kwargs(),
            n_eval_episodes=self.n_eval_episodes,
            seed_base=self.seed_base,
        )

    def _write_csv(self,
                   success_rate,
                   mean_success_distance,
                   mean_return,
                   successes):
        new_file = not os.path.exists(self.csv_path)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timesteps",
                    "successes",
                    "episodes",
                    "success_rate",
                    "mean_success_distance_cm",
                    "mean_return",
                ],
            )

            if new_file:
                writer.writeheader()

            writer.writerow({
                "timesteps": int(self.num_timesteps),
                "successes": int(successes),
                "episodes": int(self.n_eval_episodes),
                "success_rate": round(success_rate, 3),
                "mean_success_distance_cm": (
                    ""
                    if not np.isfinite(mean_success_distance)
                    else round(mean_success_distance, 4)
                ),
                "mean_return": round(mean_return, 6),
            })

    def _on_step(self):
        # self.num_timesteps already counts transitions across all VecEnv
        # workers, so this frequency is expressed directly in TOTAL training
        # timesteps and does not need division by n_envs.
        if (
            self.num_timesteps - self.last_eval_timestep
            < self.eval_every_timesteps
        ):
            return True

        self.last_eval_timestep = self.num_timesteps

        (
            success_rate,
            mean_success_distance,
            mean_return,
            successes,
        ) = self._evaluate_success()

        self._write_csv(
            success_rate,
            mean_success_distance,
            mean_return,
            successes,
        )

        self.logger.record(
            "success_eval/success_rate", success_rate
        )
        self.logger.record(
            "success_eval/mean_return", mean_return
        )

        if np.isfinite(mean_success_distance):
            self.logger.record(
                "success_eval/mean_success_distance_cm",
                mean_success_distance,
            )

        if self.verbose:
            distance_text = (
                "n/a"
                if not np.isfinite(mean_success_distance)
                else f"{mean_success_distance:.2f} cm"
            )
            print(
                f"\nSuccess eval @ {self.num_timesteps} steps: "
                f"{successes}/{self.n_eval_episodes} "
                f"({success_rate:.1f}%), "
                f"mean successful distance={distance_text}, "
                f"mean return={mean_return:.3f}"
            )

        # Selection must be LEVEL-AWARE. The previous rule compared success
        # rate alone, so once 100% was recorded at an easy curriculum level
        # nothing could ever strictly beat it -- 100% is the ceiling. Every
        # later 100% at a harder level merely tied, and the distance
        # tie-breaker favoured the easy level (shorter approaches).
        #
        # Observed consequence (run_20260911_154749): the curriculum held
        # 20/20 at level 12 for ~200k steps and success_evaluations.csv
        # recorded it, yet only TWO saves occurred in 600k steps, and the
        # resulting best_success_model.zip scored 17/100 on the real task
        # while final_model.zip scored 100/100.
        #
        # Fix: compare (level, success_rate, -distance) lexicographically.
        # A harder level always wins; within a level the old rule applies.
        current_level = (
            self.curriculum_state.level
            if self.curriculum_state is not None else 0
        )
        current_key = (
            current_level,
            success_rate,
            -mean_success_distance if success_rate > 0.0 else -float("inf"),
        )
        best_key = (
            self.best_level,
            self.best_success_rate,
            -self.best_mean_success_distance,
        )

        improved = success_rate > 0.0 and current_key > best_key

        if improved:
            self.best_level = current_level
            self.best_success_rate = success_rate
            self.best_mean_success_distance = (
                mean_success_distance
            )

            save_path = os.path.join(
                self.save_dir, "best_success_model"
            )
            replay_path = os.path.join(
                self.save_dir, "best_success_replay_buffer.pkl"
            )

            self.model.save(save_path)

            if self.model.replay_buffer is not None:
                self.model.save_replay_buffer(replay_path)

            if self.verbose:
                print(
                    "New best SUCCESS checkpoint saved: "
                    f"{save_path}.zip"
                )

        return True


class CurriculumState:
    """Single source of truth for the current curriculum spawn radius.

    Previously SuccessRateCheckpointCallback and the SB3 EvalCallback both
    built their evaluation environments from bare ENV_KWARGS, which carries
    no radius, so both silently evaluated the DEFAULT 0.11-0.14 m annulus
    for the whole run regardless of the curriculum level actually being
    trained. best_success_model.zip was therefore selected on the easy task.
    Sharing one mutable state object keeps every evaluator aligned with the
    level currently being trained.
    """

    def __init__(self, radius_min=0.11, radius_max=0.14, platform_speed=0.0):
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.platform_speed = float(platform_speed)
        # Index of the curriculum level currently being trained. Checkpoint
        # selection must compare across levels, not just success rate -- see
        # SuccessRateCheckpointCallback._current_key().
        self.level = 0

    def as_env_kwargs(self, base_kwargs):
        kwargs = dict(base_kwargs)
        kwargs['spawn_radius_min'] = self.radius_min
        kwargs['spawn_radius_max'] = self.radius_max
        kwargs['platform_speed_range'] = self.platform_speed
        return kwargs


class RadiusCurriculumCallback(BaseCallback):
    """Widen the platform spawn annulus INSIDE one uninterrupted TD3 run.

    Rationale (see handoff report sections 13-14, 19-20): save/reload
    fine-tuning degraded every good checkpoint it was applied to, because
    reloading discards the replay buffer and hands the optimizer a state
    distribution it has no data for. Expanding the radius in-process keeps
    the TD3 object, replay buffer, optimizer moments and target networks
    continuous, so none of that discontinuity occurs.

    Promotion is gated on measured success at the CURRENT level, using the
    same fixed-seed evaluator as SuccessRateCheckpointCallback.
    """

    # (spawn_radius_max, platform_speed_range) per level.
    #
    # Phase 1 (levels 1-7) widens the spawn annulus on a STATIONARY pad --
    # byte-identical to the schedule that reached 100/100 across three seeds,
    # so the validated result is reproduced on the way through rather than
    # replaced.
    #
    # Phase 2 (levels 8-12) holds the widest annulus and ramps platform
    # speed to 0.25 m/s. Ramping rather than jumping matters: the original
    # 11-30 cm collapse was a discovery failure, and introducing full speed
    # in one step re-creates exactly that condition.
    LEVELS = [
        (0.14, 0.00), (0.16, 0.00), (0.18, 0.00), (0.20, 0.00),
        (0.23, 0.00), (0.26, 0.00), (0.30, 0.00),
        (0.30, 0.05), (0.30, 0.10), (0.30, 0.15),
        (0.30, 0.20), (0.30, 0.25),
    ]

    def __init__(self,
                 state,
                 eval_envs=(),
                 eval_every_timesteps=SUCCESS_EVAL_EVERY_TIMESTEPS,
                 promote_threshold=0.90,
                 n_eval_episodes=SUCCESS_EVAL_EPISODES,
                 radius_min=0.11,
                 verbose=1):
        super().__init__(verbose)
        self.state = state
        # Raw LanderAIAviary instances used by other evaluators. SB3 wraps
        # eval_env in a DummyVecEnv but keeps the same underlying object, so
        # mutating it here keeps those evaluators on the current level too.
        self.eval_envs = list(eval_envs)
        self.eval_every_timesteps = eval_every_timesteps
        self.promote_threshold = promote_threshold
        self.n_eval_episodes = n_eval_episodes
        self.radius_min = radius_min
        self.level = 0
        self.last_eval_timestep = 0

    def _apply_level(self, level):
        hi, speed = self.LEVELS[level]
        self.training_env.env_method('set_spawn_radius', self.radius_min, hi)
        self.training_env.env_method('set_platform_speed', speed)
        self.state.radius_min = self.radius_min
        self.state.radius_max = hi
        self.state.platform_speed = speed
        self.state.level = level
        for env in self.eval_envs:
            env.set_spawn_radius(self.radius_min, hi)
            env.set_platform_speed(speed)
        if self.verbose:
            print(f"\n[curriculum] level {level + 1}/{len(self.LEVELS)}  "
                  f"spawn radius {self.radius_min:.2f}-{hi:.2f} m, "
                  f"platform speed {speed:.2f} m/s\n")

    def _on_training_start(self):
        # Envs are constructed with the default annulus; force level 1
        # explicitly so the starting difficulty is unambiguous.
        self._apply_level(self.level)

    def _current_env_kwargs(self):
        return self.state.as_env_kwargs(ENV_KWARGS)

    def _on_step(self):
        if self.num_timesteps - self.last_eval_timestep < self.eval_every_timesteps:
            return True
        self.last_eval_timestep = self.num_timesteps

        # NOTE: evaluate BEFORE the final-level check. An earlier version
        # returned early at the widest level, which silenced reporting for
        # the rest of the run exactly when the hardest task was being
        # trained -- so the run ended with no measurement of 0.11-0.30 m.
        success_rate, _, _, n_successes = _evaluate_success_rate(
            self.model,
            self._current_env_kwargs(),
            n_eval_episodes=self.n_eval_episodes,
            seed_base=10_000,
        )

        at_final_level = self.level >= len(self.LEVELS) - 1

        if self.verbose:
            hi, speed = self.LEVELS[self.level]
            print(f"[curriculum] {self.num_timesteps} steps, level "
                  f"{self.level + 1}/{len(self.LEVELS)} "
                  f"(<= {hi:.2f} m, {speed:.2f} m/s): "
                  f"{n_successes}/{self.n_eval_episodes} "
                  f"({success_rate:.1f}%)"
                  + ("  [final level]" if at_final_level else ""))

        if at_final_level:
            return True

        if success_rate >= self.promote_threshold * 100.0:
            self.level += 1
            self._apply_level(self.level)

        return True


def train(total_timesteps=100_000, output_dir='results_lander', continue_from=None,
          seed=0):
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # Reproducibility. Previously make_vec_env was seeded but TD3 itself was
    # not, so torch network initialisation differed between otherwise
    # identical runs. Two such runs (v1, v2) reached level 7/7 and level 1/7
    # respectively on the same code, so the run-to-run spread here is large
    # and must be measured across seeds rather than reported from one run.
    print(f"\nSeed: {seed}\n")

    train_env = make_vec_env(
        LanderAIAviary,
        n_envs=N_TRAIN_ENVS,
        seed=seed,
        env_kwargs=ENV_KWARGS,
    )
    eval_env = LanderAIAviary(**ENV_KWARGS)

    n_actions = train_env.action_space.shape[-1]

    if continue_from:
        noise_sigma = FINETUNE_NOISE_SIGMA
    else:
        noise_sigma = 0.1

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=noise_sigma * np.ones(n_actions),
    )

    policy_kwargs = dict(
        net_arch=NET_ARCH,
        activation_fn=torch.nn.ReLU,
    )

    baseline_success_rate = -1.0
    baseline_mean_distance = float("inf")

    if continue_from:
        print(f"\nContinuing training from: {continue_from}\n")

        model = TD3.load(
            continue_from,
            env=train_env,
            tensorboard_log=os.path.join(run_dir, 'tb'),
        )

        # Preserve the exact incoming model inside this run directory.
        # This guarantees that curriculum fine-tuning can never erase the
        # checkpoint we started from.
        model.save(os.path.join(run_dir, "initial_model"))

        model.action_noise = action_noise
        model.learning_starts = 0

        # Fine-tuning uses a 10x smaller LR to reduce destructive updates.
        model.learning_rate = FINETUNE_LEARNING_RATE
        model.lr_schedule = get_schedule_fn(
            FINETUNE_LEARNING_RATE
        )

        # Evaluate the incoming checkpoint on the SAME fixed seeds used by
        # the success callback. Seed best_success_model with this baseline so
        # later training must actually beat it before overwriting it.
        (
            baseline_success_rate,
            baseline_mean_distance,
            baseline_mean_return,
            baseline_successes,
        ) = _evaluate_success_rate(
            model,
            ENV_KWARGS,
            n_eval_episodes=SUCCESS_EVAL_EPISODES,
            seed_base=10_000,
        )

        print(
            f"Incoming-policy success baseline: "
            f"{baseline_successes}/{SUCCESS_EVAL_EPISODES} "
            f"({baseline_success_rate:.1f}%), "
            f"mean return={baseline_mean_return:.3f}"
        )

        # Treat the incoming policy as the current best-success checkpoint.
        model.save(os.path.join(run_dir, "best_success_model"))

    else:
        model = TD3(
            'MlpPolicy',
            train_env,
            verbose=1,
            seed=seed,
            gradient_steps=-1,
            learning_rate=0.0001,       # PAPER FACT
            buffer_size=1_000_000,      # PAPER FACT
            batch_size=100,             # PAPER FACT
            learning_starts=100,        # PAPER FACT
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(run_dir, 'tb'),
        )

    reward_eval_freq_calls = max(
        REWARD_EVAL_EVERY_TIMESTEPS // N_TRAIN_ENVS,
        1,
    )

    reward_eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=reward_eval_freq_calls,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    curriculum_state = CurriculumState() if not continue_from else None

    success_eval_callback = SuccessRateCheckpointCallback(
        save_dir=run_dir,
        env_kwargs=ENV_KWARGS,
        eval_every_timesteps=SUCCESS_EVAL_EVERY_TIMESTEPS,
        n_eval_episodes=SUCCESS_EVAL_EPISODES,
        initial_best_success_rate=baseline_success_rate,
        initial_best_mean_success_distance=baseline_mean_distance,
        curriculum_state=curriculum_state,
        verbose=1,
    )

    callback_list = [
        reward_eval_callback,
        success_eval_callback,
    ]

    # Curriculum applies to FRESH training only. On the continue_from path
    # the incoming policy already has a task distribution baked into it, and
    # re-imposing level 1 would itself be a distribution shift -- the exact
    # failure mode this callback exists to avoid.
    if not continue_from:
        callback_list.append(RadiusCurriculumCallback(
            state=curriculum_state,
            # eval_env is the SB3 EvalCallback's environment. SB3 wraps it in
            # a DummyVecEnv but keeps this same object, so updating it here
            # keeps mean_reward/best_model aligned with the current level
            # instead of frozen at the default 0.11-0.14 m annulus.
            eval_envs=[eval_env],
            verbose=1,
        ))

    callbacks = CallbackList(callback_list)

    print(f"\nTraining for {total_timesteps} steps. Output: {run_dir}\n")
    print("CURRENT CURRICULUM: stationary platform "
          "(platform_speed_range=0.0).\n")

    if continue_from:
        prefill_steps = min(
            FINETUNE_PREFILL_STEPS,
            total_timesteps,
        )

        if prefill_steps > 0:
            print(
                f"Fine-tune replay-buffer prefill: collecting "
                f"{prefill_steps} transitions with the pretrained policy "
                f"and NO gradient updates.\n"
            )

            # Policy actions are used from the first step because
            # learning_starts=0. gradient_steps=0 means collect only.
            model.gradient_steps = 0
            model.learn(
                total_timesteps=prefill_steps,
                reset_num_timesteps=True,
            )

        remaining_steps = total_timesteps - prefill_steps

        if remaining_steps > 0:
            print(
                f"Starting gentle fine-tuning for the remaining "
                f"{remaining_steps} transitions: "
                f"lr={FINETUNE_LEARNING_RATE}, "
                f"noise_sigma={FINETUNE_NOISE_SIGMA}, "
                f"gradient_steps={FINETUNE_GRADIENT_STEPS}.\n"
            )

            model.gradient_steps = FINETUNE_GRADIENT_STEPS

            model.learn(
                total_timesteps=remaining_steps,
                callback=callbacks,
                reset_num_timesteps=False,
            )
    else:
        print(
            "NOTE: paper's own initial training budget is 5,000,000 steps "
            "(Table I) -- this run uses far fewer, a documented deviation "
            "driven by current experimental scope.\n"
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )

    model.save(os.path.join(run_dir, 'final_model'))

    if model.replay_buffer is not None:
        model.save_replay_buffer(
            os.path.join(run_dir, 'final_replay_buffer.pkl')
        )

    print(f"\nDone. Saved to {run_dir}\n")
    return run_dir


def _append_to_log(row):
    fields = list(row.keys())
    new_file = not os.path.exists(RESULTS_LOG)
    with open(RESULTS_LOG, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Appended to {RESULTS_LOG}")


def evaluate(model_path, n_episodes=100, note=""):
    """Run many episodes and report metrics comparable to the paper's own
    reported figures (Table I: success rate: Table II: mean distance to
    target at landing, in cm)."""
    env = LanderAIAviary(**ENV_KWARGS)
    model = TD3.load(model_path)

    successes = 0
    final_distances = []       # cm, successful landings only -- matches
                               # the paper's Table II metric
    reason_counts = Counter()
    plat_speeds_fail = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        if info.get("success"):
            successes += 1
            final_distances.append(info["d_target"] * 100.0)   # m -> cm
        else:
            reason = info.get("truncation_reason") or "terminated_no_reason"
            reason_counts[reason] += 1
            plat_speeds_fail.append(info.get("plat_speed", float('nan')))

    env.close()

    print("\n" + "=" * 58)
    print(f"  EVALUATION OVER {n_episodes} EPISODES")
    if note:
        print(f"  {note}")
    print("=" * 58)
    print(f"  Landed successfully       : {successes}/{n_episodes}"
          f"  ({100*successes/n_episodes:.1f}%)")
    if ENV_KWARGS.get("platform_speed_range", 0.0) == 0.0:
        print("  NOTE: current evaluation platform is stationary; "
              "do not directly compare this curriculum result with "
              "the paper's moving-platform LMPL result.")
    else:
        print("  Paper LMPL reference: 93.33% success "
              "(only meaningful when the evaluation setup matches).")

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_path,
        "note": note,
        "n_episodes": n_episodes,
        "success_rate": round(100.0 * successes / n_episodes, 2),
    }

    if final_distances:
        print("-" * 58)
        print("  Landing precision (successful landings only, cm)")
        print(f"    mean {np.mean(final_distances):.2f} +/- "
              f"{np.std(final_distances):.2f}   "
              f"(paper's LMPL: 6.50 +/- 2.14 cm)")
        row["dist_mean_cm"] = round(float(np.mean(final_distances)), 3)
        row["dist_std_cm"] = round(float(np.std(final_distances)), 3)
    else:
        row["dist_mean_cm"] = ""
        row["dist_std_cm"] = ""

    print("-" * 58)
    print("  Why failed episodes ended")
    for reason, count in reason_counts.most_common():
        print(f"    {reason:20s}: {count}")
    if plat_speeds_fail and not all(np.isnan(plat_speeds_fail)):
        print(f"    failed episodes' platform speed: "
              f"mean {np.nanmean(plat_speeds_fail):.3f} m/s")

    print("=" * 58)
    _append_to_log(row)
    print()


def play(model_path):
    """Watch a single landing. NOTE: GUI rendering has previously crashed
    on Intel/Mesa driver setups in this project -- try
    LIBGL_ALWAYS_SOFTWARE=1 python train_lander.py --gui if it crashes."""
    env = LanderAIAviary(gui=True, **ENV_KWARGS)
    model = TD3.load(model_path)

    obs, info = env.reset(seed=np.random.randint(0, 100000))
    for _ in range(int(env.EPISODE_LEN_SEC * env.CTRL_FREQ)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1 / env.CTRL_FREQ)
        if terminated or truncated:
            break
    env.close()

    print(f"\n  Platform speed: {info.get('plat_speed', 0):.3f} m/s")
    print(f"  Success: {info.get('success')}")
    print(f"  Final distance to target: {info.get('d_target', float('nan'))*100:.2f} cm\n")


def _latest_model():
    root = 'results_lander'

    if not os.path.isdir(root):
        print("No results_lander directory found. Train a model first.")
        sys.exit(1)

    run_dirs = [
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.startswith("run_")
        and os.path.isdir(os.path.join(root, name))
    ]

    if not run_dirs:
        print("No training run directories found. Train a model first.")
        sys.exit(1)

    latest_run = max(run_dirs, key=os.path.getmtime)

    # Prefer the checkpoint selected by TRUE landing success. Fall back to
    # reward-best, then final model for older runs.
    candidates = [
        os.path.join(latest_run, 'best_success_model.zip'),
        os.path.join(latest_run, 'best_model.zip'),
        os.path.join(latest_run, 'final_model.zip'),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    print(f"No model ZIP found in latest run: {latest_run}")
    sys.exit(1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='watch one landing')
    parser.add_argument('--eval', action='store_true', help='run the metrics table')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=100_000,
                        help='paper uses 5,000,000 initial -- default here is '
                             'much smaller due to today\'s time constraints')
    parser.add_argument('--continue-from', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--note', type=str, default="")
    parser.add_argument('--seed', type=int, default=0,
                        help='training seed; run several to measure the '
                             'run-to-run spread instead of reporting one draw')
    args = parser.parse_args()

    if args.gui:
        play(args.model or _latest_model())
    elif args.eval:
        evaluate(args.model or _latest_model(), args.episodes, args.note)
    else:
        train(total_timesteps=args.steps, continue_from=args.continue_from,
              seed=args.seed)
