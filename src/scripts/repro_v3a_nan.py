"""
repro_v3a_nan.py
================

Reproduces the V3a crash with strict finite-value assertions at every
boundary, so the FIRST non-finite value is reported rather than the final
SciPy SVD failure.

Identical to train_aruco_v3a.py's configuration in every respect that could
affect the result -- same seed, same curriculum start, same camera, same
estimator mode, same TD3 hyperparameters, same 4 workers. The ONLY difference
is that DiagnosticArucoAviary is used in place of ArucoLanderAviary, adding
assertions and nothing else.

TWO MODES, and the difference matters more than it looks.

  --mode bare       No callbacks. Reaches 40k faster, but DOES NOT reproduce
                    the original trajectory: SB3's NormalActionNoise draws
                    from numpy's GLOBAL RNG, and constructing/resetting the
                    evaluation environments consumes global draws. Removing
                    the callbacks therefore changes the exploration noise and
                    hence the entire rollout. A clean run in this mode proves
                    only that no SYSTEMATIC non-finite value exists; it says
                    nothing about the specific crash.

  --mode faithful   The real trainer, unmodified, with DiagnosticArucoAviary
                    swapped in for ArucoLanderAviary in both the training and
                    the evaluation environments. Same callbacks, same
                    cadence, same global-RNG consumption, same trajectory.
                    This is the mode that can actually reproduce the crash.

NOTHING IS MASKED. The run is expected to stop with NonFiniteError.

USAGE
    python repro_v3a_nan.py --steps 50000 --seed 1

On failure it writes nan_dumps/nan_w<worker>_<timestamp>.json/.npz and then,
per requirement 6, inspects the actor, critic and replay buffer for
non-finite values -- but only after the environment boundaries have already
reported which side the problem started on.
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

from envs.DiagnosticArucoAviary import (                    # noqa: E402
    DiagnosticArucoAviary, NonFiniteError)
import train_aruco_v3a as v3a                               # noqa: E402


def inspect_model(model):
    """Requirement 6: only meaningful once the env is proven finite."""
    print('=' * 74)
    print('  MODEL / REPLAY-BUFFER INSPECTION')
    print('=' * 74)

    bad = []
    for label, net in (('actor', model.actor), ('critic', model.critic),
                       ('actor_target', model.actor_target),
                       ('critic_target', model.critic_target)):
        for name, param in net.named_parameters():
            if not torch.isfinite(param).all():
                bad.append(f'{label}.{name}')
    if bad:
        print(f'  NON-FINITE PARAMETERS in: {bad}')
    else:
        print('  all actor/critic parameters finite')

    rb = model.replay_buffer
    if rb is None:
        print('  no replay buffer')
    else:
        n = rb.size()
        print(f'  replay buffer entries: {n}')
        for field in ('observations', 'next_observations', 'actions',
                      'rewards', 'dones'):
            arr = getattr(rb, field, None)
            if arr is None:
                continue
            sl = np.asarray(arr[:n], dtype=float)
            finite = np.isfinite(sl).all()
            print(f'    {field:18s}: '
                  f'{"finite" if finite else "NON-FINITE"}'
                  f'   min={np.nanmin(sl):+.4g} max={np.nanmax(sl):+.4g}')
            if not finite:
                idx = np.argwhere(~np.isfinite(sl))
                print(f'      first non-finite index: {idx[0].tolist()} '
                      f'(of {len(idx)} total)')
    print('=' * 74)


def run_faithful(args):
    """Run the REAL trainer with only the environment class swapped.

    evaluate_policy_full() and train() both reference the module-global name
    ArucoLanderAviary, so rebinding it here routes every environment the
    trainer creates -- training workers and evaluation envs alike -- through
    DiagnosticArucoAviary. Nothing else about the run changes: same
    callbacks, same cadence, same curriculum, same global-RNG consumption,
    same trajectory.
    """
    print('=' * 74)
    print('  V3a NaN REPRODUCTION -- FAITHFUL MODE')
    print('=' * 74)
    print('  The real trainer, with DiagnosticArucoAviary substituted for')
    print('  ArucoLanderAviary. Callbacks, evaluation cadence and RNG')
    print('  consumption are all identical to the run that crashed.')
    print(f'  seed  : {args.seed}')
    print(f'  steps : {args.steps}')
    print(f'  dumps : {args.dump_dir}/')
    print('=' * 74)
    print()

    v3a.ArucoLanderAviary = DiagnosticArucoAviary

    original = v3a.make_env_kwargs

    def make_env_kwargs_with_dump(*a, **k):
        kw = original(*a, **k)
        kw['dump_dir'] = args.dump_dir
        return kw

    v3a.make_env_kwargs = make_env_kwargs_with_dump

    try:
        v3a.train(args.steps, args.seed, args.outdir)
        print()
        print('=' * 74)
        print(f'  Completed {args.steps} steps with NO non-finite value.')
        print('  The crash is intermittent. Either re-run, or accept that it')
        print('  is rare and decide separately whether a guard is warranted.')
        print('=' * 74)
    except NonFiniteError as e:
        print()
        print('=' * 74)
        print('  REPRODUCED -- first non-finite value caught at its source')
        print('=' * 74)
        print(f'  {e}')
        print()
        if os.path.isdir(args.dump_dir):
            print('  Dump files:')
            for f in sorted(os.listdir(args.dump_dir)):
                print(f'    {os.path.join(args.dump_dir, f)}')
        print()
        print('  NOTE: the model object is owned by v3a.train() and is not')
        print('  reachable from here, so the requirement-6 parameter and')
        print('  replay-buffer inspection is not run in faithful mode. The')
        print('  environment-side dump identifies which side the problem')
        print('  started on, which is the question that comes first.')
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=50_000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--envs', type=int, default=v3a.N_TRAIN_ENVS)
    ap.add_argument('--dump-dir', default='nan_dumps')
    ap.add_argument('--mode', default='faithful',
                    choices=['bare', 'faithful'])
    ap.add_argument('--outdir', default='results_v3a_repro')
    args = ap.parse_args()

    if args.mode == 'faithful':
        return run_faithful(args)

    print('=' * 74)
    print('  V3a NaN REPRODUCTION -- strict assertions, nothing masked')
    print('=' * 74)
    print(f'  seed   : {args.seed}')
    print(f'  steps  : {args.steps}  (crash previously observed near 40k)')
    print(f'  envs   : {args.envs}')
    print(f'  dumps  : {args.dump_dir}/')
    print('  Expected outcome: NonFiniteError with a first-failure dump.')
    print('=' * 74)
    print()

    state = v3a.CurriculumState()
    env_kwargs = dict(state.env_kwargs())
    env_kwargs['dump_dir'] = args.dump_dir

    train_env = make_vec_env(
        DiagnosticArucoAviary, n_envs=args.envs, seed=args.seed,
        env_kwargs=env_kwargs)

    n_actions = train_env.action_space.shape[-1]
    model = TD3(
        'MlpPolicy', train_env, verbose=1, seed=args.seed,
        gradient_steps=-1, learning_rate=0.0001, buffer_size=1_000_000,
        batch_size=100, learning_starts=100,
        action_noise=NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)),
        policy_kwargs=dict(net_arch=v3a.NET_ARCH,
                           activation_fn=torch.nn.ReLU),
    )

    t0 = time.time()
    try:
        model.learn(total_timesteps=args.steps)
        print()
        print('=' * 74)
        print(f'  Completed {args.steps} steps with NO non-finite value.')
        print('  The crash did not reproduce at this seed/budget. Either it')
        print('  is intermittent, or the callbacks (absent here) are')
        print('  implicated. Re-run for longer before concluding anything.')
        print('=' * 74)
    except NonFiniteError as e:
        print()
        print('=' * 74)
        print('  REPRODUCED -- first non-finite value caught at its source')
        print('=' * 74)
        print(f'  {e}')
        print(f'  elapsed: {time.time() - t0:.0f} s')
        print()
        print('  Dump files:')
        for f in sorted(os.listdir(args.dump_dir)):
            print(f'    {os.path.join(args.dump_dir, f)}')
        print()
        inspect_model(model)
    finally:
        train_env.close()


if __name__ == '__main__':
    main()
