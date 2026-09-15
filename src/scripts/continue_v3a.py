"""
continue_v3a.py
===============

TRUE continuation of an existing V3a run, for the extended-budget experiment:

    Can memoryless TD3 trained on ArUco + constant-velocity prediction
    eventually solve the final moving 0.25 m/s task, and how much additional
    sample complexity does it need relative to privileged TD3?

The privileged agent cleared 12/12 curriculum levels in 600k transitions.
V3a's fresh vision run reached 4/12 in the same budget. This script extends
seed 1 toward a hard ceiling of ~1.8-2.0M total transitions to find out
whether it converges at all.


WHY THIS IS A SEPARATE FILE
------------------------------------------------------------------------------
train_aruco_v3a.py deliberately has NO --continue-from path, because V3a's
main result depends on being able to state that the policy was trained from
scratch with no privileged pretraining mixed in. That guarantee is preserved
by not touching that file. This script is the extended-budget experiment and
must be reported as such -- NOT as part of V3a's headline result.


WHAT IS RESTORED, AND WHAT IS NOT
------------------------------------------------------------------------------
Restored exactly (from the model zip and buffer pickle):

    actor, critic, actor_target, critic_target weights
    actor.optimizer and critic.optimizer state (Adam moments)
    replay buffer -- all 600k transitions
    num_timesteps, via learn(reset_num_timesteps=False)
    action noise configuration

Reconstructed from success_evaluations.csv:

    curriculum level        last row's 'level' column, minus 1 for the index
    spawn radius min/max    last row
    platform speed          last row
    best-checkpoint key     recomputed by replaying the same lexicographic
                            rule (level, success_rate, -precision) over every
                            row of the CSV, so a later level cannot be
                            outranked by an earlier 100%

NOT restored -- and this is the one honest gap:

    numpy / torch RNG state.

SB3 does not checkpoint the exploration random stream. The LEARNED state
carries over exactly, but the random stream restarts. This continuation is
therefore NOT bit-identical to an uninterrupted 1.8M run, and the result must
be described as "600k fresh + N continued", never as "1.8M uninterrupted".


WHAT IS DELIBERATELY NOT CHANGED
------------------------------------------------------------------------------
reward, camera, marker, estimator, TD3 architecture and hyperparameters,
curriculum levels and thresholds, observation dimensionality, number of
parallel environments, evaluation cadence and seeds.

The solvePnP non-finite rejection applied before the original run stands as a
NUMERICAL ROBUSTNESS FIX -- it prevents a NaN reaching DSLPIDControl. It is
not an estimator improvement and must not be described as one.


A HAZARD WORTH NAMING IN ADVANCE
------------------------------------------------------------------------------
The restored buffer holds 600k transitions collected entirely on levels 1-4,
all STATIONARY. When the curriculum reaches level 8 and platform speed ramps
in, the buffer is dominated by off-distribution data. TD3 is off-policy so
this is survivable, and with buffer_size=1e6 the earliest data begins evicting
past roughly 1M total transitions. But if progress stalls precisely as speed
is introduced, buffer composition is a plausible cause -- named here in
advance so it is a hypothesis to test rather than a post-hoc explanation.


USAGE
------------------------------------------------------------------------------
Always inspect first; this prints everything it would restore and exits:

    python continue_v3a.py --run-dir results_aruco_v3a/run_20260912_201312 \\
        --verify

Then continue. Splitting into two ~600k blocks rather than one long run gives
a checkpoint and a decision point midway:

    python continue_v3a.py --run-dir results_aruco_v3a/run_20260912_201312 \\
        --additional-steps 600000
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.ArucoLanderAviary import ArucoLanderAviary        # noqa: E402
import train_aruco_v3a as v3a                               # noqa: E402


# Hard ceiling on TOTAL transitions for seed 1, agreed in advance.
TOTAL_CEILING = 2_000_000


# ===========================================================================
# RESTORE
# ===========================================================================

def read_curriculum_state(run_dir):
    """Recover level, radius and speed from the last evaluation row.

    The CSV is written by CurriculumAndCheckpointCallback with the SNAPSHOT
    taken before the evaluation ran, so the level recorded on each row is the
    level the model was genuinely evaluated at -- promotion happens only
    after the row is written. That ordering is what makes this reconstruction
    trustworthy.
    """
    path = os.path.join(run_dir, 'success_evaluations.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found -- cannot recover the '
                                f'curriculum state.')

    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'{path} is empty.')

    last = rows[-1]
    return {
        'level_index': int(last['level']) - 1,      # CSV is 1-based
        'radius_min': float(last['spawn_radius_min']),
        'radius_max': float(last['spawn_radius_max']),
        'platform_speed': float(last['platform_speed_range']),
        'last_timestep': int(last['timestep']),
        'rows': rows,
    }


def replay_best_key(rows):
    """Rebuild the best-checkpoint selection key from the CSV history.

    Replays the exact lexicographic rule used during training:
    (level, success_rate, -precision), successes > 0 required. Without this,
    the continuation would start with best_* unset and could overwrite a
    genuinely harder-level checkpoint with an easier one.
    """
    best_level, best_rate, best_prec = -1, -1.0, float('inf')
    for r in rows:
        successes = int(r['successes'])
        if successes <= 0:
            continue
        level = int(r['level']) - 1
        rate = float(r['success_rate'])
        prec_raw = r['mean_success_distance_cm']
        prec = float(prec_raw) if prec_raw not in ('', None) else float('inf')
        key = (level, rate, -prec)
        if key > (best_level, best_rate, -best_prec):
            best_level, best_rate, best_prec = level, rate, prec
    return best_level, best_rate, best_prec


def describe(run_dir, state, model_path, buffer_path, additional):
    best_level, best_rate, best_prec = replay_best_key(state['rows'])
    total_after = state['last_timestep'] + additional

    print('=' * 74)
    print('  V3a CONTINUATION -- RESTORE PLAN')
    print('=' * 74)
    print(f'  source run   : {run_dir}')
    print(f'  model        : {os.path.basename(model_path)} '
          f'({os.path.getsize(model_path) / 1e6:.1f} MB)')
    print(f'  replay buffer: {os.path.basename(buffer_path)} '
          f'({os.path.getsize(buffer_path) / 1e6:.1f} MB)')
    print()
    print('  RESTORED EXACTLY')
    print('    actor / critic / target networks     from model zip')
    print('    actor + critic optimizer state       from model zip')
    print('    replay buffer                        from pickle')
    print(f'    num_timesteps                        '
          f'{state["last_timestep"]} (reset_num_timesteps=False)')
    print()
    print('  RECONSTRUCTED FROM success_evaluations.csv')
    print(f'    curriculum level                     '
          f'{state["level_index"] + 1}/{len(v3a.CURRICULUM_LEVELS)}')
    print(f'    spawn radius                         '
          f'{state["radius_min"]:.2f}-{state["radius_max"]:.2f} m')
    print(f'    platform speed                       '
          f'{state["platform_speed"]:.2f} m/s')
    print(f'    best-checkpoint key                  '
          f'level {best_level + 1}, {best_rate:.1f}%, '
          f'{best_prec:.3f} cm')
    print()
    print('  NOT RESTORED')
    print('    numpy / torch RNG state -- SB3 does not checkpoint it.')
    print('    The continuation is NOT bit-identical to an uninterrupted')
    print('    run. Report as "600k fresh + continued", never as a single')
    print('    uninterrupted budget.')
    print()
    print(f'  additional steps : {additional}')
    print(f'  total after run  : {total_after}')
    if total_after > TOTAL_CEILING:
        print(f'  WARNING: exceeds the agreed {TOTAL_CEILING} ceiling.')
    print('=' * 74)
    return best_level, best_rate, best_prec


# ===========================================================================
# CONTINUE
# ===========================================================================

def continue_run(args):
    run_dir = args.run_dir
    model_path = os.path.join(run_dir, 'final_model.zip')
    buffer_path = os.path.join(run_dir, 'final_replay_buffer.pkl')

    for p in (model_path, buffer_path):
        if not os.path.exists(p):
            print(f'ERROR: {p} not found.')
            print('A true continuation needs BOTH the final model and its')
            print('matching replay buffer. Resuming from the model alone')
            print('with an empty buffer is the destructive path this')
            print('project already established degrades good policies.')
            sys.exit(1)

    state = read_curriculum_state(run_dir)
    best_level, best_rate, best_prec = describe(
        run_dir, state, model_path, buffer_path, args.additional_steps)

    if args.verify:
        print('\n  --verify: nothing was loaded or trained. Exiting.')
        return

    total_after = state['last_timestep'] + args.additional_steps
    if total_after > TOTAL_CEILING and not args.allow_over_ceiling:
        print(f'\nRefusing to exceed the {TOTAL_CEILING} total ceiling.')
        print('Pass --allow-over-ceiling to override deliberately.')
        sys.exit(1)

    out_dir = os.path.join(
        args.outdir, time.strftime('cont_%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n  output: {out_dir}')
    print('  the source run directory is NEVER modified.\n')

    # Environments at the RESTORED level, not at level 1.
    env_kwargs = v3a.make_env_kwargs(
        state['radius_min'], state['radius_max'], state['platform_speed'])

    train_env = make_vec_env(
        ArucoLanderAviary, n_envs=v3a.N_TRAIN_ENVS, seed=args.seed,
        env_kwargs=env_kwargs)
    eval_env = ArucoLanderAviary(**env_kwargs)

    print('  loading model...')
    model = TD3.load(model_path, env=train_env,
                     tensorboard_log=os.path.join(out_dir, 'tb'))
    print(f'  loaded. num_timesteps = {model.num_timesteps}')

    print('  loading replay buffer (144 MB, takes a moment)...')
    model.load_replay_buffer(buffer_path)
    print(f'  loaded. buffer size = {model.replay_buffer.size()}')

    if model.replay_buffer.size() == 0:
        print('\nERROR: replay buffer loaded empty. Refusing to continue --')
        print('this is exactly the destructive fine-tuning condition.')
        sys.exit(1)

    # Curriculum state restored to the level actually reached.
    cur = v3a.CurriculumState(
        radius_min=state['radius_min'],
        radius_max=state['radius_max'],
        platform_speed=state['platform_speed'])
    cur.level = state['level_index']

    callback = v3a.CurriculumAndCheckpointCallback(
        out_dir, cur, eval_envs=[eval_env])
    # Resume the curriculum at the reached level rather than level 1.
    callback.level = state['level_index']
    # Carry the best key forward so an easier level cannot overwrite a
    # harder-level checkpoint.
    callback.best_level = best_level
    callback.best_success_rate = best_rate
    callback.best_precision = best_prec
    # Suppress an immediate spurious evaluation on the first step.
    callback.last_eval_timestep = model.num_timesteps

    perception = v3a.PerceptionDiagnosticCallback(out_dir, cur)
    perception.last_eval_timestep = model.num_timesteps

    callbacks = CallbackList([
        EvalCallback(
            eval_env, best_model_save_path=out_dir, log_path=out_dir,
            eval_freq=max(
                v3a.REWARD_EVAL_EVERY_TIMESTEPS // v3a.N_TRAIN_ENVS, 1),
            n_eval_episodes=5, deterministic=True, render=False),
        callback,
        perception,
    ])

    print(f'\n  resuming at level {state["level_index"] + 1}/'
          f'{len(v3a.CURRICULUM_LEVELS)} '
          f'(radius {state["radius_min"]:.2f}-{state["radius_max"]:.2f} m, '
          f'speed {state["platform_speed"]:.2f} m/s)')
    print(f'  training {args.additional_steps} additional transitions '
          f'-> {total_after} total\n')

    t0 = time.time()
    model.learn(total_timesteps=args.additional_steps,
                callback=callbacks,
                reset_num_timesteps=False)
    elapsed = time.time() - t0

    model.save(os.path.join(out_dir, 'final_model'))
    model.save_replay_buffer(os.path.join(out_dir, 'final_replay_buffer.pkl'))

    print(f'\n  done in {elapsed / 3600:.2f} h. saved to {out_dir}')
    print(f'  total transitions now: {model.num_timesteps}')
    print('\n  Check the level reached:')
    print(f'    tail -5 {out_dir}/success_evaluations.csv')
    print('\n  Only if level 12 was reached and trained on is a comparison')
    print('  against V2 (98.0% privileged / 53.3% frozen+vision) valid.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True,
                    help='source run directory containing final_model.zip '
                         'and final_replay_buffer.pkl')
    ap.add_argument('--additional-steps', type=int, default=600_000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--outdir', default='results_aruco_v3a_cont')
    ap.add_argument('--verify', action='store_true',
                    help='print the restore plan and exit without training')
    ap.add_argument('--allow-over-ceiling', action='store_true')
    args = ap.parse_args()
    continue_run(args)


if __name__ == '__main__':
    main()
