"""
bench_v3a.py
============

PHASE A of V3a: measure what training with real rendering actually costs,
BEFORE committing to a long run.

Reports rollout throughput and evaluation overhead SEPARATELY, because with
rendering + ArUco the existing 5,000-step evaluation cadence may cost more
than the rollouts it is measuring. That is the specific thing this benchmark
exists to find out.

It changes nothing. Camera rate, env count, evaluation frequency, curriculum
and TD3 update ratio are all exactly as train_aruco_v3a.py will use them. If
the numbers say there is a compute problem, that is a conversation, not an
automatic fix.

USAGE
    python bench_v3a.py --steps 8000 --envs 4
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.ArucoLanderAviary import ArucoLanderAviary          # noqa: E402
import train_aruco_v3a as v3a                                 # noqa: E402


def bench_rollout(steps, n_envs, seed):
    """Time pure training: rollouts + gradient updates, no callbacks."""
    state = v3a.CurriculumState()
    kw = state.env_kwargs()

    train_env = make_vec_env(
        ArucoLanderAviary, n_envs=n_envs, seed=seed, env_kwargs=kw)
    n_actions = train_env.action_space.shape[-1]

    model = TD3(
        'MlpPolicy', train_env, verbose=0, seed=seed,
        gradient_steps=-1, learning_rate=0.0001, buffer_size=1_000_000,
        batch_size=100, learning_starts=100,
        action_noise=NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)),
        policy_kwargs=dict(net_arch=v3a.NET_ARCH,
                           activation_fn=torch.nn.ReLU),
    )

    t0 = time.time()
    model.learn(total_timesteps=steps)
    elapsed = time.time() - t0

    # Render accounting. _total_renders is cumulative per worker and is NOT
    # reset between episodes, so summing across workers gives the true count.
    renders = sum(train_env.get_attr('_total_renders'))

    # Reset accounting. Monitor records one entry per COMPLETED episode, and
    # every completed episode is followed by an auto-reset; add n_envs for
    # the initial reset each worker performs.
    try:
        completed = sum(len(x) for x in train_env.get_attr('episode_returns'))
    except Exception:                                       # noqa: BLE001
        completed = None
    resets = None if completed is None else completed + n_envs

    train_env.close()
    return model, elapsed, renders, resets


def bench_eval(model, n_episodes, seed_base):
    """Time one success-evaluation of the kind the callbacks run."""
    state = v3a.CurriculumState()
    t0 = time.time()
    res = v3a.evaluate_policy_full(
        model, state.env_kwargs(), n_episodes, seed_base)
    return time.time() - t0, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--envs', type=int, default=v3a.N_TRAIN_ENVS)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    print("=" * 74)
    print("  V3a PHASE A -- THROUGHPUT BENCHMARK")
    print("=" * 74)
    print(f"  transitions : {args.steps}")
    print(f"  envs        : {args.envs}")
    print(f"  camera      : {v3a.VISION_KWARGS['camera_res']} @ "
          f"{v3a.VISION_KWARGS['camera_fov_deg']} deg")
    print(f"  estimator   : {v3a.VISION_KWARGS['estimator_mode']}")
    print("  Nothing is being tuned here. This only measures.")
    print()

    print("--- rollout + gradient updates (no callbacks) " + "-" * 28)
    model, roll_s, renders, resets = bench_rollout(
        args.steps, args.envs, args.seed)
    fps = args.steps / roll_s

    # ctrl_freq=30, pyb_freq=240 -> one render per control step.
    ctrl_hz = 30
    expected = args.steps
    print(f"  wall clock            : {roll_s:.1f} s")
    print(f"  SB3 FPS               : {fps:.1f} transitions/s")
    print(f"  camera renders        : {renders}")
    print(f"  control steps         : {expected}")

    # EXACT invariant, no tolerance. Verified empirically:
    #
    #     renders == control_steps + 2 * episode_resets
    #
    # Two renders occur per reset, and both are accounted for:
    #   1. BaseRLAviary.reset() calls _computeObs() while PLAT_ID is None.
    #      Strict mode materialises the platform -- but at the PREVIOUS
    #      episode's plat_pos0, because _samplePlatformMotion() has not run
    #      yet. This render is against a stale platform and its result is
    #      deliberately DISCARDED.
    #   2. ArucoLanderAviary.reset() then bumps _det_epoch and recomputes
    #      after the platform has been re-sampled. This is the render whose
    #      estimate the episode actually starts from, and it is why
    #      steps_before_first_detection is 0.
    #
    # Within a step, exactly one render occurs: _refreshDetection() caches on
    # _det_epoch, which _preprocessAction() bumps once per control step, and
    # both _computeObs() and _computeInfo() consume that same cache.
    if resets is None:
        print("  renders == steps      : UNVERIFIED (reset count unavailable)")
    else:
        predicted = expected + 2 * resets
        match = renders == predicted
        print(f"  episode resets        : {resets}")
        print(f"  renders == steps+2*resets : "
              f"{'EXACT' if match else 'MISMATCH'}"
              f"  ({renders} vs {predicted}, delta {renders - predicted:+d})")
        if not match:
            print("    ^ investigate before the long run: the visual state")
            print("      for a step may not correspond to that physical state.")
    print(f"  renders / sim-second  : {ctrl_hz} (one per control step at "
          f"{ctrl_hz} Hz)")
    print(f"  simulated seconds     : {args.steps / ctrl_hz:.1f} s")
    print(f"  realtime factor       : {(args.steps / ctrl_hz) / roll_s:.2f}x")
    print()

    print("--- one success-evaluation (20 episodes, fixed seeds) " + "-" * 20)
    eval_s, res = bench_eval(model, v3a.SUCCESS_EVAL_EPISODES,
                             v3a.SUCCESS_EVAL_SEED_BASE)
    print(f"  wall clock            : {eval_s:.1f} s")
    print(f"  success rate          : {res['success_rate']:.1f}%  "
          f"(untrained -- expected near 0)")
    print(f"  blind fraction        : {res['blind_fraction'] * 100:.1f}%")
    print()

    # Callbacks firing every SUCCESS_EVAL_EVERY_TIMESTEPS steps, AFTER the
    # Option A deduplication:
    #   CurriculumAndCheckpointCallback  20 episodes (shared: drives BOTH
    #                                    checkpoint selection and promotion)
    #   EvalCallback                      5 episodes
    # Previously this was 45 episodes, because curriculum and checkpoint
    # selection each ran their own identical 20-episode evaluation.
    per_cycle_eps = v3a.SUCCESS_EVAL_EPISODES + 5
    per_ep = eval_s / v3a.SUCCESS_EVAL_EPISODES
    eval_per_cycle = per_ep * per_cycle_eps
    cycle_steps = v3a.SUCCESS_EVAL_EVERY_TIMESTEPS
    roll_per_cycle = cycle_steps / fps
    overhead = 100.0 * eval_per_cycle / (roll_per_cycle + eval_per_cycle)

    print("--- projected cost " + "-" * 55)
    print(f"  evaluation episodes per {cycle_steps}-step cycle : "
          f"{per_cycle_eps}")
    print(f"  rollout time per cycle     : {roll_per_cycle / 60:.1f} min")
    print(f"  evaluation time per cycle  : {eval_per_cycle / 60:.1f} min")
    print(f"  EVALUATION OVERHEAD        : {overhead:.1f}% of wall clock")
    print()

    eff = fps * (1.0 - overhead / 100.0)
    print(f"  {'budget':>12} {'rollout only':>16} {'with evaluation':>18}")
    print("  " + "-" * 50)
    for budget in (100_000, 300_000, 1_000_000, 1_500_000):
        print(f"  {budget:>12,} {budget / fps / 3600:>14.1f} h "
              f"{budget / eff / 3600:>16.1f} h")
    print()
    print("  Privileged-training FPS for comparison: check the old run logs.")
    print("  If none survive, that comparison is unavailable -- say so rather")
    print("  than estimating it.")
    print("=" * 74)
    print("  STOP HERE. Report these numbers before launching a long run.")
    print("  Do NOT reduce the camera rate, env count, or evaluation cadence")
    print("  on the basis of this benchmark without discussing it first --")
    print("  each would be a new experimental parameter.")
    print("=" * 74)


if __name__ == '__main__':
    main()
