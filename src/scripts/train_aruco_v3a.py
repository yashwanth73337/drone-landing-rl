"""
train_aruco_v3a.py
==================

Stage V3a. Trains TD3 FROM SCRATCH on the ArUco + constant-velocity-prediction
observation used in V2, to answer exactly one question:

    How much of V2's ~45 pp privileged-to-vision gap is recovered simply by
    training the policy on the observation distribution it will actually
    encounter at test time?

V2 measured a frozen, privileged-trained policy flying on estimated state.
That conflates two things: the estimator's limitations, and the policy seeing
a distribution it was never trained on. V3a separates them.

ONE CONCEPTUAL VARIABLE CHANGES relative to the privileged Strand-B setup:
the source of the target-relative observation components. Everything else --
physics, platform geometry, contact-based success, dp = 0.1*c_t via
DSLPIDControl, reward function and constants, episode duration, failure
conditions, success tolerance, curriculum mechanism, TD3 architecture and
hyperparameters, parallel env count, seed handling -- is held fixed.

EXPLICITLY NOT IN V3a: LSTM, PPO, asymmetric critic, validity flag, estimate
age, active-perception reward, reward reshaping, new action semantics, warm
starting. Each of those is a separate experiment and would prevent a clean
answer to the question above.

NO WARM START. A fresh TD3 model is always constructed. There is deliberately
no --continue-from path: this project has already observed destructive
checkpoint fine-tuning, and warm-starting from a privileged checkpoint would
mix privileged pretraining with vision adaptation and make the result
uninterpretable.


CURRICULUM
------------------------------------------------------------------------------
Ported VERBATIM from train_lander.py's RadiusCurriculumCallback. The callbacks
could not simply be imported: train_lander.py's _evaluate_success_rate()
hardcodes LanderAIAviary, so importing and monkey-patching it would be more
fragile than copying. train_lander.py itself remains untouched.

VERIFY THE PORT BEFORE THE LONG RUN:
    python train_aruco_v3a.py --verify-curriculum
compares LEVELS and the associated constants against the live train_lander.py
and refuses to continue if they differ.


PRIVILEGED INFORMATION
------------------------------------------------------------------------------
The environment is constructed with strict_no_privileged=True, so no
privileged target-relative state can reach the actor -- enforced in
_computeObs(), not merely a consequence of call ordering.

Ground truth IS still used, legitimately, for: reward, termination, success
detection, curriculum scoring, and diagnostics. Those are simulator-side
training and evaluation mechanisms, not deployed policy observations.
"""

import os
import sys
import csv
import time
import argparse
from collections import Counter

import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, EvalCallback)
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.ArucoLanderAviary import ArucoLanderAviary       # noqa: E402

# Environment class used for training AND for every evaluation environment.
# Swapped to DiagnosticArucoAviary by --diagnostic, which adds strict
# finite-value assertions and changes nothing else.
#
# WHY THIS EXISTS RATHER THAN A SEPARATE REPRO SCRIPT: SB3's NormalActionNoise
# draws from numpy's GLOBAL RNG, and constructing an environment touches that
# stream. A repro that omits the callbacks therefore draws a DIFFERENT
# exploration noise sequence, visits different states, and may never reach the
# failing one -- which is exactly what happened on the first attempt. The only
# faithful reproduction is the real trainer with the real callbacks and the
# same seed, with the env class swapped underneath.
ENV_CLASS = ArucoLanderAviary
ENV_EXTRA_KWARGS = {}


# ===========================================================================
# CONFIGURATION -- all values ported from train_lander.py
# ===========================================================================

NET_ARCH = [512, 512, 256, 128]          # Eq. 4 (PAPER FACT for the actor)

N_TRAIN_ENVS = 4
REWARD_EVAL_EVERY_TIMESTEPS = 5_000
SUCCESS_EVAL_EVERY_TIMESTEPS = 5_000
SUCCESS_EVAL_EPISODES = 20
SUCCESS_EVAL_SEED_BASE = 10_000
PROMOTE_THRESHOLD = 0.90
RADIUS_MIN = 0.11

# Perception diagnostics are expensive (they need per-step estimator stats),
# so they run on their own coarser cadence rather than inside every rollout.
PERCEPTION_EVAL_EVERY_TIMESTEPS = 25_000
PERCEPTION_EVAL_EPISODES = 10

# (spawn_radius_max, platform_speed_range) per level -- VERBATIM port.
CURRICULUM_LEVELS = [
    (0.14, 0.00), (0.16, 0.00), (0.18, 0.00), (0.20, 0.00),
    (0.23, 0.00), (0.26, 0.00), (0.30, 0.00),
    (0.30, 0.05), (0.30, 0.10), (0.30, 0.15),
    (0.30, 0.20), (0.30, 0.25),
]

# Base env kwargs. Note these carry NO spawn radius or platform speed: those
# come exclusively from CurriculumState, so no evaluator can silently fall
# back to a default (easier) task. This is the structural fix for the
# checkpoint-selection bug.
BASE_ENV_KWARGS = {
    "tilt_limit": 1.0,
    "episode_len_sec": 10.0,
}

# Perception configuration -- IDENTICAL to V2. Training and evaluation must
# use the same pipeline or the comparison is meaningless.
VISION_KWARGS = {
    "camera_res": (128, 128),
    "camera_fov_deg": 90.0,
    "marker_size": 0.25,
    "estimator_mode": "predict",
    "strict_no_privileged": True,
}

# The final moving task, stated explicitly rather than inherited from any
# script's default. This is the configuration the moving_025 results in
# V1/V2 correspond to.
FINAL_TASK_KWARGS = {
    "spawn_radius_min": 0.11,
    "spawn_radius_max": 0.30,
    "platform_speed_range": 0.25,
}


def make_env_kwargs(radius_min, radius_max, platform_speed):
    kw = dict(BASE_ENV_KWARGS)
    kw.update(VISION_KWARGS)
    kw["spawn_radius_min"] = radius_min
    kw["spawn_radius_max"] = radius_max
    kw["platform_speed_range"] = platform_speed
    return kw


# ===========================================================================
# CURRICULUM PORT VERIFICATION
# ===========================================================================

def verify_curriculum_port():
    """Compare this file's ported constants against the live train_lander.py.

    The port is a copy, and copies drift. This makes drift loud instead of
    silent.
    """
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'train_lander.py')
    if not os.path.exists(path):
        print(f"  train_lander.py not found at {path} -- cannot verify.")
        return False

    spec = importlib.util.spec_from_file_location('_tl', path)
    tl = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(tl)
    except Exception as e:                                  # noqa: BLE001
        print(f"  could not import train_lander.py: {e}")
        return False

    ok = True
    checks = [
        ('LEVELS', CURRICULUM_LEVELS,
         list(tl.RadiusCurriculumCallback.LEVELS)),
        ('N_TRAIN_ENVS', N_TRAIN_ENVS, tl.N_TRAIN_ENVS),
        ('SUCCESS_EVAL_EVERY_TIMESTEPS', SUCCESS_EVAL_EVERY_TIMESTEPS,
         tl.SUCCESS_EVAL_EVERY_TIMESTEPS),
        ('SUCCESS_EVAL_EPISODES', SUCCESS_EVAL_EPISODES,
         tl.SUCCESS_EVAL_EPISODES),
        ('NET_ARCH', NET_ARCH, list(tl.NET_ARCH)),
    ]
    print("=" * 74)
    print("  CURRICULUM PORT VERIFICATION vs train_lander.py")
    print("=" * 74)
    for name, mine, theirs in checks:
        mine_n = [tuple(x) for x in mine] if name == 'LEVELS' else mine
        theirs_n = [tuple(x) for x in theirs] if name == 'LEVELS' else theirs
        match = mine_n == theirs_n
        ok &= match
        print(f"  [{'OK  ' if match else 'DIFF'}] {name}")
        if not match:
            print(f"         v3a          : {mine_n}")
            print(f"         train_lander : {theirs_n}")
    print("=" * 74)
    print(f"  {'PORT MATCHES' if ok else 'PORT DIVERGED -- DO NOT TRAIN'}")
    print("=" * 74)
    return ok


# ===========================================================================
# EVALUATION
# ===========================================================================

def evaluate_policy_full(model, env_kwargs, n_episodes, seed_base,
                         collect_perception=False):
    """Deterministic evaluation with failure-reason and perception breakdown.

    env_kwargs must be fully explicit -- there is no default task here to
    fall back to.
    """
    env = ENV_CLASS(**env_kwargs, **ENV_EXTRA_KWARGS)

    successes = 0
    precisions = []
    returns = []
    reasons = Counter()
    near_miss = 0
    blind_fracs = []
    longest_blinds = []
    first_dets = []
    pos_errs = []
    vel_errs = []

    try:
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed_base + ep)
            ep_return = 0.0
            low_run = 0
            low_run_max = 0
            ep_pos = []
            ep_vel = []

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)

                h = float(info["height_above_pad"])
                if h < 0.03:
                    low_run += 1
                    low_run_max = max(low_run_max, low_run)
                else:
                    low_run = 0

                if collect_perception and info.get("est_detected"):
                    ep_pos.append(info["est_err_pos_norm"])
                    ep_vel.append(info["est_err_vel_norm"])

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

            if ep_pos:
                pos_errs.append(float(np.mean(ep_pos)))
                vel_errs.append(float(np.mean(ep_vel)))
    finally:
        env.close()

    return {
        "episodes": n_episodes,
        "successes": successes,
        "success_rate": 100.0 * successes / n_episodes,
        "precision_mean_cm": float(np.mean(precisions)) if precisions else float("inf"),
        "precision_std_cm": float(np.std(precisions)) if precisions else float("nan"),
        "mean_return": float(np.mean(returns)),
        "near_miss": near_miss,
        "reasons": dict(reasons),
        "blind_fraction": float(np.mean(blind_fracs)),
        "longest_blind": float(np.mean(longest_blinds)),
        "steps_before_first_detection": float(np.mean(first_dets)),
        "pos_err_m": float(np.mean(pos_errs)) if pos_errs else float("nan"),
        "vel_err_ms": float(np.mean(vel_errs)) if vel_errs else float("nan"),
    }


# ===========================================================================
# CURRICULUM STATE -- single source of truth
# ===========================================================================

class CurriculumState:
    """Shared, mutable curriculum position.

    Every evaluator builds its env kwargs from THIS object. Ported from
    train_lander.py, where its absence caused each evaluator to silently
    score the default 0.11-0.14 m annulus for an entire run regardless of
    the level actually being trained.
    """

    def __init__(self, radius_min=RADIUS_MIN, radius_max=0.14,
                 platform_speed=0.0):
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.platform_speed = float(platform_speed)
        self.level = 0

    def env_kwargs(self):
        return make_env_kwargs(
            self.radius_min, self.radius_max, self.platform_speed)

    def describe(self):
        return (f"level {self.level + 1}/{len(CURRICULUM_LEVELS)}  "
                f"radius {self.radius_min:.2f}-{self.radius_max:.2f} m  "
                f"speed {self.platform_speed:.2f} m/s")


# ===========================================================================
# CALLBACKS
# ===========================================================================

class CurriculumAndCheckpointCallback(BaseCallback):
    """ONE evaluation, consumed by both checkpoint selection and promotion.

    WHY THESE ARE MERGED (Option A -- deduplication only, no science changed).
    In train_lander.py these were two separate callbacks. Both ran 20
    deterministic episodes, on the same model, at the same cadence, at the
    same fixed seeds, at the same curriculum level -- so the second
    evaluation recomputed, at full rendering cost, a result the first had
    already produced. Under privileged training that duplication was nearly
    free. With real rendering it was measured at roughly half of all
    evaluation time, which was two thirds of total wall clock.

    Nothing scientific changes: same seeds, same cadence, same episode count,
    same promotion rule, same selection rule. Only the duplicate computation
    is removed.

    ATOMICITY -- this is the part that matters.
    The evaluation is performed against a SNAPSHOT of the curriculum state
    taken before it runs, and the promotion decision is applied only AFTER
    checkpoint selection and CSV logging have completed for that snapshot.
    If promotion were allowed to mutate CurriculumState first, the checkpoint
    and the CSV row would be labelled with a level the model was never
    evaluated at -- which is a fresh variant of the checkpoint-selection bug
    this project already spent time diagnosing once.

    Order is therefore strictly:
        1. snapshot (level, radius_min, radius_max, platform_speed)
        2. evaluate once against that snapshot
        3. write the CSV row, labelled with the snapshot
        4. checkpoint selection, keyed on the snapshot's level
        4b. per-level checkpoint, which nothing can overwrite across levels
        5. only now, decide promotion and mutate state


    PER-LEVEL CHECKPOINTS (step 4b) -- added 19 Sep 2026
    ------------------------------------------------------------------------
    The step-4 rule is lexicographic on (level, success_rate, -precision), so
    curriculum level dominates absolutely. That has now failed twice, in
    opposite directions:

        - the original ceiling bug: 100% at an easy level can never be
          beaten, so saves froze. Fixed by making level dominant.
        - V3b seed 2: at 570k the rule compared level 7 @ 0% against the
          saved level 6 @ 90% and saved the level-7 one. Evaluated on one
          fixed task afterwards, the discarded policy scored 67.0% and the
          saved one 17.0%. The run's designated artefact was four times
          worse than the policy it replaced.

    Both follow from treating a STATE VARIABLE as a PERFORMANCE MEASURE.

    The step-4 rule is deliberately left UNCHANGED here. It produced every
    best_success_model already reported, and altering it would silently
    invalidate cross-run comparisons against V3a -- a frozen baseline. The
    failure is instead made non-destructive: a separate best checkpoint is
    kept for every curriculum level, so a promotion onto a level the policy
    cannot fly can no longer destroy the best policy at the level it could.

    Model only, no replay buffer: 12 levels of buffer would be ~1.6 GB per
    run, and a buffer is needed only to CONTINUE training, not to evaluate.

    Also added: a loud warning when the newly-saved global best scores BELOW
    the best seen at any earlier level. That is exactly the seed-2 situation,
    and it was previously silent -- the log line read "NEW BEST saved at
    level 7 (20.0%)", which looks like progress.

    Nothing here draws from any RNG stream (model.save() is pure
    serialisation), so seed-matched reproduction is unaffected.
    See V3b_REPORT.md section 5.5.
    """

    LEVELS = CURRICULUM_LEVELS

    def __init__(self, save_dir, curriculum_state, eval_envs=(),
                 eval_every_timesteps=SUCCESS_EVAL_EVERY_TIMESTEPS,
                 n_eval_episodes=SUCCESS_EVAL_EPISODES,
                 seed_base=SUCCESS_EVAL_SEED_BASE,
                 promote_threshold=PROMOTE_THRESHOLD,
                 radius_min=RADIUS_MIN, verbose=1):
        super().__init__(verbose=verbose)
        self.save_dir = save_dir
        self.state = curriculum_state
        self.eval_envs = list(eval_envs)
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
        self.csv_path = os.path.join(save_dir, "success_evaluations.csv")

        # Per-level bests. Additive -- the global best above is untouched.
        self.best_by_level = {}        # level -> (success_rate, -precision)
        self.best_any_rate = -1.0      # highest success rate at ANY level
        self.best_any_level = -1

    # ---- curriculum ---------------------------------------------------
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
            print(f"\n[curriculum] -> {self.state.describe()}\n")

    def _on_training_start(self):
        self._apply_level(self.level)

    # ---- the shared record --------------------------------------------
    def _build_record(self, snapshot, res):
        reasons = res["reasons"]
        return {
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
        }

    def _write_row(self, r):
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r.keys()))
            if new:
                w.writeheader()
            w.writerow(r)

    def _write_level_manifest(self):
        """Rewrite the per-level checkpoint index.

        Rewritten rather than appended so it always shows the CURRENT best
        per level, which is what a later evaluation needs in order to pick a
        file. The full evaluation history is already in
        success_evaluations.csv.

        The task columns are included deliberately: every evaluation in this
        project must be reported WITH the task it was measured on, and this
        file is where a later reader finds the right --radius-max / --speed
        for each checkpoint.
        """
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

        # ---- 1. SNAPSHOT, before anything can mutate it ----------------
        snapshot = {
            "level": self.state.level,
            "radius_min": self.state.radius_min,
            "radius_max": self.state.radius_max,
            "platform_speed": self.state.platform_speed,
        }
        eval_kwargs = make_env_kwargs(
            snapshot["radius_min"], snapshot["radius_max"],
            snapshot["platform_speed"])

        # ---- 2. EVALUATE ONCE ------------------------------------------
        res = evaluate_policy_full(
            self.model, eval_kwargs, self.n_eval_episodes, self.seed_base)
        record = self._build_record(snapshot, res)

        # ---- 3. LOG, labelled with the snapshot ------------------------
        self._write_row(record)
        self.logger.record("success_eval/success_rate", res["success_rate"])
        self.logger.record("success_eval/level", snapshot["level"] + 1)
        self.logger.record("success_eval/mean_return", res["mean_return"])
        self.logger.record("success_eval/near_miss", res["near_miss"])

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
                  f"below_platform {record['below_platform']}  "
                  f"tilt {record['tilt_bound']}")

        # ---- 4. CHECKPOINT SELECTION, keyed on the SNAPSHOT level ------
        # Lexicographic (level, success_rate, -precision): a harder level
        # always wins. Comparing success rate alone means a 100% at an easy
        # level can never be beaten, because 100% is the ceiling.
        #
        # LEFT UNCHANGED ON PURPOSE -- this rule produced every
        # best_success_model already reported. Its failure mode is handled
        # additively in 4b rather than by altering it. See the class
        # docstring.
        key = (snapshot["level"], res["success_rate"],
               -res["precision_mean_cm"] if res["successes"]
               else -float("inf"))
        best = (self.best_level, self.best_success_rate, -self.best_precision)

        if res["successes"] > 0 and key > best:
            # Captured before best_any_* is updated below, so it compares
            # against the historical maximum, not this evaluation.
            regressed = (self.best_any_rate >= 0.0
                         and res["success_rate"] < self.best_any_rate)
            self.best_level = snapshot["level"]
            self.best_success_rate = res["success_rate"]
            self.best_precision = res["precision_mean_cm"]
            self.model.save(os.path.join(self.save_dir, "best_success_model"))
            if self.model.replay_buffer is not None:
                self.model.save_replay_buffer(
                    os.path.join(self.save_dir,
                                 "best_success_replay_buffer.pkl"))
            if self.verbose:
                print(f"         NEW BEST saved at level "
                      f"{snapshot['level'] + 1} "
                      f"({res['success_rate']:.1f}%)")
                if regressed:
                    print(f"         WARNING: this scores BELOW the best "
                          f"seen at any level "
                          f"({self.best_any_rate:.1f}% at level "
                          f"{self.best_any_level + 1}). Curriculum level "
                          f"dominates the selection key, so "
                          f"best_success_model.zip is NOT the best policy "
                          f"this run produced.")
                    print(f"         Use best_L"
                          f"{self.best_any_level + 1:02d}.zip instead, and "
                          f"see best_per_level.csv.")

        # ---- 4b. PER-LEVEL BEST, which nothing can overwrite -----------
        # The step-4 rule can only ever move to a HIGHER level, so once a
        # promotion happens the best policy at the previous level becomes
        # unrecoverable. These files make that impossible.
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

        # ---- 5. ONLY NOW may promotion mutate the state ----------------
        at_final = self.level >= len(self.LEVELS) - 1
        if at_final:
            if self.verbose:
                print("         [final level]")
            return True

        if res["success_rate"] >= self.promote_threshold * 100.0:
            self.level += 1
            self._apply_level(self.level)
        return True


class PerceptionDiagnosticCallback(BaseCallback):
    """Fixed-seed perception metrics on a coarser cadence.

    Kept out of the main rollout loop so logging does not slow training.
    """

    def __init__(self, save_dir, curriculum_state,
                 eval_every_timesteps=PERCEPTION_EVAL_EVERY_TIMESTEPS,
                 n_eval_episodes=PERCEPTION_EVAL_EPISODES,
                 seed_base=20_000, verbose=1):
        super().__init__(verbose=verbose)
        self.save_dir = save_dir
        self.state = curriculum_state
        self.eval_every_timesteps = int(eval_every_timesteps)
        self.n_eval_episodes = int(n_eval_episodes)
        self.seed_base = int(seed_base)
        self.last_eval_timestep = 0
        self.csv_path = os.path.join(save_dir, "perception_evaluations.csv")

    def _on_step(self):
        if (self.num_timesteps - self.last_eval_timestep
                < self.eval_every_timesteps):
            return True
        self.last_eval_timestep = self.num_timesteps

        res = evaluate_policy_full(
            self.model, self.state.env_kwargs(), self.n_eval_episodes,
            self.seed_base, collect_perception=True)

        row = {
            "timestep": int(self.num_timesteps),
            "level": self.state.level + 1,
            "platform_speed_range": self.state.platform_speed,
            "success_rate": round(res["success_rate"], 3),
            "blind_fraction": round(res["blind_fraction"], 4),
            "longest_blind_run": round(res["longest_blind"], 2),
            "steps_before_first_detection":
                round(res["steps_before_first_detection"], 2),
            "pos_err_m": round(res["pos_err_m"], 5),
            "vel_err_ms": round(res["vel_err_ms"], 5),
        }
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

        if self.verbose:
            print(f"  [perception] blind {res['blind_fraction'] * 100:.1f}%  "
                  f"longest {res['longest_blind']:.0f} steps  "
                  f"pos err {res['pos_err_m']:.4f} m  "
                  f"vel err {res['vel_err_ms']:.3f} m/s")
        return True


# ===========================================================================
# TRAIN
# ===========================================================================

def train(total_timesteps, seed, output_dir):
    if not verify_curriculum_port():
        print("\nRefusing to train on a diverged curriculum port.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'=' * 74}")
    print("  V3a -- FRESH TD3 ON ARUCO + CONSTANT-VELOCITY PREDICTION")
    print(f"{'=' * 74}")
    print(f"  seed            : {seed}")
    print(f"  total timesteps : {total_timesteps}")
    print(f"  output          : {run_dir}")
    print(f"  estimator_mode  : {VISION_KWARGS['estimator_mode']}")
    print(f"  strict privileged guard : "
          f"{VISION_KWARGS['strict_no_privileged']}")
    print(f"  camera          : {VISION_KWARGS['camera_res']} @ "
          f"{VISION_KWARGS['camera_fov_deg']} deg, marker "
          f"{VISION_KWARGS['marker_size']} m")
    print("  NO WARM START -- a fresh model is always constructed.")
    print(f"{'=' * 74}\n")

    state = CurriculumState()
    start_kwargs = state.env_kwargs()

    train_env = make_vec_env(
        ENV_CLASS, n_envs=N_TRAIN_ENVS, seed=seed,
        env_kwargs=dict(start_kwargs, **ENV_EXTRA_KWARGS))
    eval_env = ENV_CLASS(**start_kwargs, **ENV_EXTRA_KWARGS)

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(
        'MlpPolicy', train_env, verbose=1, seed=seed,
        gradient_steps=-1,
        learning_rate=0.0001,        # PAPER FACT
        buffer_size=1_000_000,       # PAPER FACT
        batch_size=100,              # PAPER FACT
        learning_starts=100,         # PAPER FACT
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=NET_ARCH, activation_fn=torch.nn.ReLU),
        tensorboard_log=os.path.join(run_dir, 'tb'),
    )

    # Option A: ONE shared evaluation drives both checkpoint selection and
    # curriculum promotion. See CurriculumAndCheckpointCallback for why, and
    # for the atomicity guarantee.
    callbacks = CallbackList([
        EvalCallback(
            eval_env, best_model_save_path=run_dir, log_path=run_dir,
            eval_freq=max(REWARD_EVAL_EVERY_TIMESTEPS // N_TRAIN_ENVS, 1),
            n_eval_episodes=5, deterministic=True, render=False),
        CurriculumAndCheckpointCallback(run_dir, state,
                                        eval_envs=[eval_env]),
        PerceptionDiagnosticCallback(run_dir, state),
    ])

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    elapsed = time.time() - t0

    model.save(os.path.join(run_dir, 'final_model'))
    if model.replay_buffer is not None:
        model.save_replay_buffer(
            os.path.join(run_dir, 'final_replay_buffer.pkl'))

    print(f"\nDone in {elapsed / 3600:.2f} h. Saved to {run_dir}")
    print("\nEvaluate the BEST SUCCESS checkpoint on the final moving task:")
    print(f"  python train_aruco_v3a.py --eval "
          f"--model {run_dir}/best_success_model.zip --episodes 100")
    return run_dir


# ===========================================================================
# FINAL EVALUATION
# ===========================================================================

def evaluate(model_path, n_episodes, seed_base,
             radius_min=None, radius_max=None, speed=None):
    """Evaluate a checkpoint on an explicitly-stated task.

    Defaults to the FINAL moving 0.25 m/s task, with seed base 9000 to match
    V2's evaluation seeds, so the three-way comparison (privileged /
    frozen+vision / vision-trained) is genuinely seed-matched.

    WHY THE OVERRIDES EXIST.
    A curriculum run can stop short of level 12. Evaluating such a checkpoint
    only on the final task conflates two very different failures:

        (a) the policy cannot solve the task it TRAINED on, or
        (b) the policy solves its own task but does not generalise to a
            harder one it never saw.

    V3a's first run stalled at level 4 (radius <= 0.20 m, STATIONARY) and
    scored 5% on the final moving task. Without also measuring it at level 4,
    (a) and (b) are indistinguishable -- and they support opposite
    conclusions about whether memoryless TD3 on estimated state is viable.

    So: --radius-max / --speed let the checkpoint be scored on its OWN
    training distribution. Any such evaluation must be reported WITH its task
    settings, never alongside the final-task numbers as if comparable.
    """
    kw = make_env_kwargs(
        FINAL_TASK_KWARGS["spawn_radius_min"] if radius_min is None
        else radius_min,
        FINAL_TASK_KWARGS["spawn_radius_max"] if radius_max is None
        else radius_max,
        FINAL_TASK_KWARGS["platform_speed_range"] if speed is None
        else speed)

    is_final = (radius_min is None and radius_max is None and speed is None)

    print("=" * 74)
    if is_final:
        print("  V3a FINAL EVALUATION -- moving 0.25 m/s task")
    else:
        print("  V3a EVALUATION -- CUSTOM TASK (not the final benchmark)")
    print("=" * 74)
    print(f"  model    : {model_path}")
    print(f"  episodes : {n_episodes}  (seeds {seed_base}..)")
    print(f"  task     : radius {kw['spawn_radius_min']}-"
          f"{kw['spawn_radius_max']} m, speed "
          f"{kw['platform_speed_range']} m/s, "
          f"{kw['episode_len_sec']} s episodes")
    print()

    model = TD3.load(model_path)
    r = evaluate_policy_full(model, kw, n_episodes, seed_base,
                             collect_perception=True)

    print(f"  success              : {r['successes']}/{r['episodes']} "
          f"({r['success_rate']:.1f}%)")
    if np.isfinite(r["precision_mean_cm"]):
        print(f"  precision            : {r['precision_mean_cm']:.3f} +/- "
              f"{r['precision_std_cm']:.3f} cm")
    print(f"  near-miss            : {r['near_miss']}")
    print(f"  failure reasons      : {r['reasons']}")
    print(f"  blind fraction       : {r['blind_fraction'] * 100:.1f}%")
    print(f"  longest blind run    : {r['longest_blind']:.1f} steps")
    print(f"  steps before 1st det : "
          f"{r['steps_before_first_detection']:.2f}")
    print(f"  pos estimation error : {r['pos_err_m']:.5f} m")
    print(f"  vel estimation error : {r['vel_err_ms']:.5f} m/s")
    print("=" * 74)
    if is_final:
        print("  Compare against V2_REPORT.md section 4.2 (same task,")
        print("  same seed base, therefore directly comparable):")
        print("    privileged-trained + privileged obs : 98.0 +/- 1.6%")
        print("    privileged-trained + vision predict : 53.3 +/- 5.7%")
        print("    vision-trained     + vision predict : this run")
    else:
        print("  NOT COMPARABLE to the V2_REPORT.md section 4.2 numbers.")
        print("  Those were measured on the final moving task (radius")
        print("  0.11-0.30 m, 0.25 m/s). This run used:")
        print(f"    radius {kw['spawn_radius_min']}-"
              f"{kw['spawn_radius_max']} m, "
              f"speed {kw['platform_speed_range']} m/s")
        print("  Report this number WITH those settings attached, and only")
        print("  to answer whether the policy solved the task it actually")
        print("  trained on.")
    print("=" * 74)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=None,
                    help='total training transitions (required for training)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--outdir', default='results_aruco_v3a')
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--model', type=str, default=None)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--eval-seed-base', type=int, default=9000)
    ap.add_argument('--radius-min', type=float, default=None,
                    help='override spawn_radius_min for --eval')
    ap.add_argument('--radius-max', type=float, default=None,
                    help='override spawn_radius_max for --eval, e.g. 0.20 to '
                         'score a checkpoint on curriculum level 4')
    ap.add_argument('--speed', type=float, default=None,
                    help='override platform_speed_range for --eval, e.g. 0.0 '
                         'for the stationary phase')
    ap.add_argument('--verify-curriculum', action='store_true')
    ap.add_argument('--diagnostic', action='store_true',
                    help='use DiagnosticArucoAviary: strict finite-value '
                         'assertions at every boundary, dumping full context '
                         'at the FIRST non-finite value. Changes nothing '
                         'else -- same seed, curriculum, camera, estimator '
                         'and TD3 configuration.')
    ap.add_argument('--dump-dir', default='nan_dumps')
    args = ap.parse_args()

    if args.diagnostic:
        global ENV_CLASS, ENV_EXTRA_KWARGS
        from envs.DiagnosticArucoAviary import DiagnosticArucoAviary
        ENV_CLASS = DiagnosticArucoAviary
        ENV_EXTRA_KWARGS = {'dump_dir': args.dump_dir}
        print()
        print('*' * 74)
        print('  DIAGNOSTIC MODE -- strict finite-value assertions active.')
        print('  The run is expected to STOP at the first non-finite value.')
        print(f'  Dumps will be written to {args.dump_dir}/')
        print('  Nothing is masked: no nan_to_num, no substitution.')
        print('*' * 74)
        print()

    if args.verify_curriculum:
        sys.exit(0 if verify_curriculum_port() else 1)
    if args.eval:
        if not args.model:
            print("--eval requires --model")
            sys.exit(1)
        evaluate(args.model, args.episodes, args.eval_seed_base,
                 radius_min=args.radius_min, radius_max=args.radius_max,
                 speed=args.speed)
    else:
        if args.steps is None:
            print("--steps is required. Run bench_v3a.py first and choose a")
            print("budget deliberately; there is no default here on purpose.")
            sys.exit(1)
        train(args.steps, args.seed, args.outdir)


if __name__ == '__main__':
    main()
