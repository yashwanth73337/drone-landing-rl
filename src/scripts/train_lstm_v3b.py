"""
train_lstm_v3b.py
=================

Stage V3b. Trains TD3 from scratch on the FROZEN LSTM temporal estimator's
output, to answer one question:

    Does replacing the hand-coded constant-velocity predictor with a learned
    temporal estimator let a memoryless TD3 policy get further through the
    curriculum than V3a managed?

V3a reached curriculum level 4/12 in 600k transitions, level 9/12 at 1.8M,
then regressed at level 9 (40.2% -> 21.2% over the final 600k) with the
decline driven entirely by rising timeouts. The diagnosis was that a
memoryless actor cannot tell a fresh estimate from a 181-step-old
extrapolation, so it learns not to commit to touchdown. V3b tests whether a
better estimate changes that.


EXACTLY ONE VARIABLE CHANGES FROM V3a
------------------------------------------------------------------------------
How the relative state is propagated between detections:

    V3a   constant-velocity dead reckoning   (hand-coded)
    V3b   frozen LSTM temporal estimator     (learned offline)

Everything else is imported directly from train_aruco_v3a rather than
reimplemented, so it cannot silently drift: the 12-level curriculum, the
promotion threshold, the shared atomic evaluation callback, the
level-aware lexicographic checkpoint selection, TD3 architecture and all
hyperparameters, reward, episode duration, success criterion, camera,
marker, ArUco front-end, 4 parallel envs, evaluation cadence and seeds.

Seed 1 and a 600k budget to match V3a's first block exactly.


THE ACTOR OBSERVATION IS STILL 15-D
------------------------------------------------------------------------------
    [theta, v, omega, d_LSTM, delta_v_LSTM]

No validity bit, no estimate age, no hidden state, no extra dimension. The
actor still cannot tell whether its estimate is fresh or stale -- that
limitation is deliberately UNCHANGED from V3a, so this experiment isolates
estimate QUALITY rather than smuggling in an uncertainty signal.

actor_obs_source='lstm' is set explicitly and printed at construction. It is
never inferred from estimator_mode.

The estimator is FROZEN: no RL gradient reaches it. It is part of the
observation pipeline, not part of the learned policy.


WHAT THE ESTIMATOR PASSED BEFORE THIS RUN
------------------------------------------------------------------------------
Held-out validation (round 2, merged base + hard-regime dataset):

    blind 61-120    CV 0.6492 m   LSTM 0.1768 m
    blind >120      CV 1.1623 m   LSTM 0.2635 m
    velocity        4-5x better in every bin

Closed-loop 2x2 cross-check, LSTM passive, all four cells:

    privileged @ L9, privileged @ L12, v3a @ L9, v3a @ L12
    18 moving motion bins, 0 lost

A caveat that matters for interpreting this run: in all of that, the LSTM was
PASSIVE. It never drove a trajectory. Once the policy flies on its output, it
creates a state distribution neither collection round has seen. If V3b
underperforms, distributional feedback is the first hypothesis to test, not a
post-hoc excuse -- it is written here in advance for that reason.


USAGE
------------------------------------------------------------------------------
    python train_lstm_v3b.py --seed 1 --steps 600000 \\
        --estimator ../models/temporal_estimator_v3b_r2.pt

    # evaluate on the final moving task (V2/V3a seed base, comparable)
    python train_lstm_v3b.py --eval \\
        --model results_lstm_v3b/run_*/best_success_model.zip \\
        --estimator ../models/temporal_estimator_v3b_r2.pt --episodes 100

    # evaluate on the level it actually reached, as V3a was evaluated
    python train_lstm_v3b.py --eval --model ... --estimator ... \\
        --episodes 100 --radius-max 0.20 --speed 0.0
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.LSTMLanderAviary import LSTMLanderAviary          # noqa: E402
import train_aruco_v3a as v3a                               # noqa: E402


# Captured at import, BEFORE any rebinding. train() temporarily replaces
# v3a.make_env_kwargs so the evaluation callbacks pick up the LSTM config;
# if make_kwargs() below called the module attribute instead of this saved
# reference, it would call its own replacement and recurse forever. That is
# exactly what happened on the first launch attempt.
_V3A_MAKE_ENV_KWARGS = v3a.make_env_kwargs


def make_kwargs(radius_min, radius_max, speed, estimator_path):
    """V3a's env kwargs, with the LSTM estimator attached.

    Built from the ORIGINAL v3a.make_env_kwargs so camera, marker, FOV, tilt
    limit and episode length cannot diverge from V3a -- and so rebinding the
    module attribute cannot make this recurse.
    """
    kw = _V3A_MAKE_ENV_KWARGS(radius_min, radius_max, speed)
    # LSTMLanderAviary forces estimator_mode='predict' internally so the
    # constant-velocity baseline stays live for logging, and raises if asked
    # for anything else. What the ACTOR sees is actor_obs_source alone.
    kw.pop('estimator_mode', None)
    kw['estimator_path'] = estimator_path
    kw['actor_obs_source'] = 'lstm'
    return kw


def _swap_env_class(estimator_path):
    """Point train_aruco_v3a's env factory at LSTMLanderAviary.

    The evaluation callbacks build their environments through
    v3a.evaluate_policy_full, which instantiates the module-level ENV_CLASS
    with ENV_EXTRA_KWARGS. Rebinding only the ArucoLanderAviary name is NOT
    enough -- the evaluator does not read that name, so evaluation would
    silently keep using the constant-velocity estimator while training used
    the LSTM. The policy would then be scored, and the curriculum promoted,
    on a different observation from the one it learned on.

    ENV_EXTRA_KWARGS is emptied on purpose: estimator_path and
    actor_obs_source already arrive through env_kwargs from make_kwargs(),
    and passing them twice raises a duplicate-keyword TypeError.

    getattr is used so this still works against an older revision of
    train_aruco_v3a that lacks the ENV_CLASS indirection.

    Returns a token for _restore_env_class().
    """
    token = {
        'had_cls': hasattr(v3a, 'ENV_CLASS'),
        'cls': getattr(v3a, 'ENV_CLASS', None),
        'extra': dict(getattr(v3a, 'ENV_EXTRA_KWARGS', {}) or {}),
        'aruco': v3a.ArucoLanderAviary,
    }

    def mk(radius_min, radius_max, speed):
        return make_kwargs(radius_min, radius_max, speed, estimator_path)

    if token['had_cls']:
        v3a.ENV_CLASS = LSTMLanderAviary
        v3a.ENV_EXTRA_KWARGS = {}
    v3a.ArucoLanderAviary = LSTMLanderAviary
    v3a.make_env_kwargs = mk
    return token


def _restore_env_class(token):
    if token['had_cls']:
        v3a.ENV_CLASS = token['cls']
        v3a.ENV_EXTRA_KWARGS = token['extra']
    v3a.ArucoLanderAviary = token['aruco']
    v3a.make_env_kwargs = _V3A_MAKE_ENV_KWARGS


# ===========================================================================
# TRAIN
# ===========================================================================

def train(args):
    if not os.path.exists(args.estimator):
        print(f'ERROR: estimator not found: {args.estimator}')
        sys.exit(1)
    if not v3a.verify_curriculum_port():
        print('\nRefusing to train on a diverged curriculum port.')
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    run_dir = os.path.join(args.outdir, time.strftime('run_%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)

    print(f'\n{"=" * 74}')
    print('  V3b -- FRESH TD3 ON A FROZEN LSTM TEMPORAL ESTIMATOR')
    print(f'{"=" * 74}')
    print(f'  seed            : {args.seed}')
    print(f'  total timesteps : {args.steps}')
    print(f'  estimator       : {os.path.basename(args.estimator)} (FROZEN)')
    print(f'  actor obs source: LSTM  (15-D, no validity bit)')
    print(f'  output          : {run_dir}')
    print('  NO WARM START. No PPO, no asymmetric critic, no active')
    print('  perception, no reward change. Only the estimator differs from')
    print('  V3a, which reached level 4/12 in this same 600k budget.')
    print(f'{"=" * 74}\n')

    state = v3a.CurriculumState()
    start_kwargs = make_kwargs(state.radius_min, state.radius_max,
                               state.platform_speed, args.estimator)

    train_env = make_vec_env(LSTMLanderAviary, n_envs=v3a.N_TRAIN_ENVS,
                             seed=args.seed, env_kwargs=start_kwargs)
    eval_env = LSTMLanderAviary(**start_kwargs)

    n_actions = train_env.action_space.shape[-1]
    from stable_baselines3.common.noise import NormalActionNoise
    action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                     sigma=0.1 * np.ones(n_actions))

    model = TD3(
        'MlpPolicy', train_env, verbose=1, seed=args.seed,
        gradient_steps=-1,
        learning_rate=0.0001,        # PAPER FACT
        buffer_size=1_000_000,       # PAPER FACT
        batch_size=100,              # PAPER FACT
        learning_starts=100,         # PAPER FACT
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=v3a.NET_ARCH,
                           activation_fn=torch.nn.ReLU),
        tensorboard_log=os.path.join(run_dir, 'tb'),
    )

    # The evaluators build environments through v3a.evaluate_policy_full,
    # which constructs v3a.ArucoLanderAviary by module-global name. Rebinding
    # it here routes every evaluation env through LSTMLanderAviary too, so
    # training and evaluation use the SAME perception and estimation
    # pipeline. Train/eval mismatch here would invalidate the whole run.
    _swap_token = _swap_env_class(args.estimator)

    try:
        callbacks = CallbackList([
            EvalCallback(
                eval_env, best_model_save_path=run_dir, log_path=run_dir,
                eval_freq=max(v3a.REWARD_EVAL_EVERY_TIMESTEPS
                              // v3a.N_TRAIN_ENVS, 1),
                n_eval_episodes=5, deterministic=True, render=False),
            v3a.CurriculumAndCheckpointCallback(run_dir, state,
                                                eval_envs=[eval_env]),
            v3a.PerceptionDiagnosticCallback(run_dir, state),
        ])

        t0 = time.time()
        model.learn(total_timesteps=args.steps, callback=callbacks)
        elapsed = time.time() - t0
    finally:
        _restore_env_class(_swap_token)

    model.save(os.path.join(run_dir, 'final_model'))
    if model.replay_buffer is not None:
        model.save_replay_buffer(
            os.path.join(run_dir, 'final_replay_buffer.pkl'))

    print(f'\n  done in {elapsed / 3600:.2f} h. saved to {run_dir}')
    print('\n  Compare the level reached against V3a block 1 (level 4/12):')
    print(f'    tail -5 {run_dir}/success_evaluations.csv')
    return run_dir


# ===========================================================================
# EVALUATE
# ===========================================================================

def evaluate(args):
    radius_min = (v3a.FINAL_TASK_KWARGS['spawn_radius_min']
                  if args.radius_min is None else args.radius_min)
    radius_max = (v3a.FINAL_TASK_KWARGS['spawn_radius_max']
                  if args.radius_max is None else args.radius_max)
    speed = (v3a.FINAL_TASK_KWARGS['platform_speed_range']
             if args.speed is None else args.speed)
    is_final = (args.radius_min is None and args.radius_max is None
                and args.speed is None)

    kw = make_kwargs(radius_min, radius_max, speed, args.estimator)

    print('=' * 74)
    print('  V3b EVALUATION' if is_final
          else '  V3b EVALUATION -- CUSTOM TASK (not the final benchmark)')
    print('=' * 74)
    print(f'  model     : {args.model}')
    print(f'  estimator : {os.path.basename(args.estimator)} (FROZEN)')
    print(f'  task      : radius {radius_min}-{radius_max} m, '
          f'speed {speed} m/s')
    print(f'  episodes  : {args.episodes}  (seeds {args.eval_seed_base}..)')
    print()

    _swap_token = _swap_env_class(args.estimator)
    try:
        model = TD3.load(args.model)
        r = v3a.evaluate_policy_full(model, kw, args.episodes,
                                     args.eval_seed_base,
                                     collect_perception=True)
    finally:
        _restore_env_class(_swap_token)

    print(f'  success              : {r["successes"]}/{r["episodes"]} '
          f'({r["success_rate"]:.1f}%)')
    if np.isfinite(r['precision_mean_cm']):
        print(f'  precision            : {r["precision_mean_cm"]:.3f} +/- '
              f'{r["precision_std_cm"]:.3f} cm')
    print(f'  near-miss            : {r["near_miss"]}')
    print(f'  failure reasons      : {r["reasons"]}')
    print(f'  blind fraction       : {r["blind_fraction"] * 100:.1f}%')
    print(f'  longest blind run    : {r["longest_blind"]:.1f} steps')
    print(f'  steps before 1st det : '
          f'{r["steps_before_first_detection"]:.2f}')
    print(f'  pos estimation error : {r["pos_err_m"]:.5f} m')
    print(f'  vel estimation error : {r["vel_err_ms"]:.5f} m/s')
    print('=' * 74)

    if is_final:
        print('  Final moving task -- comparable to:')
        print('    privileged-trained + privileged obs : 98.0 +/- 1.6%')
        print('    privileged-trained + CV predict     : 53.3 +/- 5.7%')
        print('    vision-trained (V3a) + CV predict   : 5.0% *')
        print('    vision-trained (V3b) + LSTM         : this run')
        print('    * V3a only reached level 4 and never trained on a moving')
        print('      platform, so its 5.0% is NOT a like-for-like number.')
    else:
        print('  NOT comparable to the final-task numbers in V2/V3a.')
        print(f'  This run used radius {radius_min}-{radius_max} m, '
              f'speed {speed} m/s.')
        print('  Report this number WITH those settings attached.')
    print('=' * 74)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--estimator', required=True,
                    help='frozen estimator .pt from train_estimator.py')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--outdir', default='results_lstm_v3b')
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--model', type=str, default=None)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--eval-seed-base', type=int, default=9000)
    ap.add_argument('--radius-min', type=float, default=None)
    ap.add_argument('--radius-max', type=float, default=None)
    ap.add_argument('--speed', type=float, default=None)
    args = ap.parse_args()

    if args.eval:
        if not args.model:
            print('--eval requires --model')
            sys.exit(1)
        evaluate(args)
    else:
        if args.steps is None:
            print('--steps is required. Use 600000 to match V3a block 1.')
            sys.exit(1)
        train(args)


if __name__ == '__main__':
    main()
