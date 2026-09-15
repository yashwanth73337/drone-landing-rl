"""
check_estimator_closed_loop.py
==============================

Step 10 of the V3b plan. Runs the FROZEN estimator online, inside the real
environment, and compares its error against the offline held-out numbers from
train_estimator.py.

WHY THIS EXISTS
------------------------------------------------------------------------------
The estimator was trained on trajectories flown by privileged and scripted
policies, because no vision policy can fly levels 8-12. It will be deployed
under a vision policy that flies differently. That is a behavioural-cloning
style distribution shift, and offline held-out error can look excellent while
online error is much worse.

Discovering that AFTER a six-hour TD3 run would be the expensive way to find
out. This check costs minutes.

Ground truth is used for diagnostics only. The estimator consumes the same
13-D input it was trained on and nothing else.

WHAT IS COMPARED
------------------------------------------------------------------------------
Both estimators run on the SAME frames of the SAME episodes:

    frozen LSTM
    constant-velocity predictor (the V3a deployed logic, still live in the
                                 parent class)

binned by blind age, in physical units, against the offline table.

INTERPRETING THE RESULT
------------------------------------------------------------------------------
Some degradation is expected and acceptable. What matters is whether the
LSTM's advantage over constant velocity SURVIVES online, particularly in the
long blind bins. If online LSTM error in the >120 bin approaches or exceeds
constant velocity, the offline advantage was an artefact of the training
distribution and a DAgger-style collection round is needed before any RL.

USAGE
------------------------------------------------------------------------------
    # privileged policy, the distribution the estimator was trained on
    python check_estimator_closed_loop.py \\
        --estimator ../models/temporal_estimator_v3b.pt \\
        --policy results_lander/moving_025_seed1_final.zip \\
        --level 12 --episodes 30

    # V3a vision policy -- the harder and more informative test
    python check_estimator_closed_loop.py \\
        --estimator ../models/temporal_estimator_v3b.pt \\
        --policy results_aruco_v3a_cont/cont_20260913_150310/best_success_model.zip \\
        --level 9 --episodes 30 --drive-with-lstm
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.LSTMLanderAviary import LSTMLanderAviary          # noqa: E402
from models.temporal_estimator import BLIND_AGE_BINS        # noqa: E402
import train_aruco_v3a as v3a                               # noqa: E402


def stat(a):
    if not a:
        return None
    a = np.asarray(a)
    return {'n': int(a.size), 'mean': float(a.mean()),
            'rmse': float(np.sqrt((a ** 2).mean())),
            'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--estimator', required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--level', type=int, default=12)
    ap.add_argument('--episodes', type=int, default=30)
    ap.add_argument('--seed-base', type=int, default=41000)
    ap.add_argument('--drive-with-lstm', action='store_true',
                    help='policy flies on the LSTM observation. Without '
                         'this, the policy flies on privileged state and '
                         'the estimator is observed only -- use that for '
                         'checkpoints that expect privileged input.')
    ap.add_argument('--offline-json', default=None,
                    help='temporal_estimator_v3b_validation.json, for a '
                         'side-by-side offline/online comparison')
    ap.add_argument('--out', default='v3b_closed_loop.json')
    args = ap.parse_args()

    from stable_baselines3 import TD3
    model = TD3.load(args.policy)

    radius_max, speed = v3a.CURRICULUM_LEVELS[args.level - 1]
    kw = v3a.make_env_kwargs(v3a.RADIUS_MIN, radius_max, speed)
    kw.pop('strict_no_privileged', None)
    kw['estimator_mode'] = 'predict' if args.drive_with_lstm else 'off'
    kw['estimator_path'] = args.estimator

    print('=' * 92)
    print('  V3b CLOSED-LOOP ESTIMATOR CHECK')
    print('=' * 92)
    print(f'  estimator : {os.path.basename(args.estimator)}  (FROZEN)')
    print(f'  policy    : {os.path.basename(args.policy)}')
    print(f'  level     : {args.level}/12  '
          f'(radius {v3a.RADIUS_MIN}-{radius_max} m, speed {speed} m/s)')
    print(f'  episodes  : {args.episodes}')
    print(f'  driving   : {"LSTM observation" if args.drive_with_lstm else "PRIVILEGED observation (estimator observed only)"}')
    print()

    env = LSTMLanderAviary(**kw)
    bins = {n: {'lstm_p': [], 'lstm_v': [], 'cv_p': [], 'cv_v': []}
            for n, _, _ in BLIND_AGE_BINS}
    successes = 0

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed_base + ep)
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)

                age = int(info.get('est_steps_since_detection', 0))
                lp = info.get('lstm_err_pos_norm')
                lv = info.get('lstm_err_vel_norm')
                cp = info.get('cv_err_pos_norm')
                cv = info.get('cv_err_vel_norm')

                if lp is not None and cp is not None:
                    for name, lo, hi in BLIND_AGE_BINS:
                        if lo <= age <= hi:
                            bins[name]['lstm_p'].append(lp)
                            bins[name]['lstm_v'].append(lv)
                            bins[name]['cv_p'].append(cp)
                            bins[name]['cv_v'].append(cv)
                            break

                if term or trunc:
                    successes += bool(info.get('success', False))
                    break
    finally:
        env.close()

    print(f'  policy success during the check: {successes}/{args.episodes}')
    print()
    print(f'  {"bin":<14}{"n":>8}'
          f'{"CV pos":>11}{"LSTM pos":>11}{"CV p95":>10}{"LSTM p95":>10}'
          f'{"CV vel":>11}{"LSTM vel":>11}')
    print('  ' + '-' * 84)

    out, verdict_ok = {}, True
    for name, _, _ in BLIND_AGE_BINS:
        b = bins[name]
        lp, lv, cp, cv = (stat(b['lstm_p']), stat(b['lstm_v']),
                          stat(b['cv_p']), stat(b['cv_v']))
        if lp is None:
            continue
        out[name] = {'lstm_pos': lp, 'lstm_vel': lv,
                     'cv_pos': cp, 'cv_vel': cv}
        print(f'  {name:<14}{lp["n"]:>8}'
              f'{cp["mean"]:>11.4f}{lp["mean"]:>11.4f}'
              f'{cp["p95"]:>10.4f}{lp["p95"]:>10.4f}'
              f'{cv["mean"]:>11.4f}{lv["mean"]:>11.4f}')
        # The advantage only has to survive where it matters: long dropout.
        if name in ('blind 61-120', 'blind >120'):
            if lp['mean'] >= cp['mean']:
                verdict_ok = False
    print('  ' + '-' * 84)

    if args.offline_json and os.path.exists(args.offline_json):
        off = json.load(open(args.offline_json))['bins']
        print()
        print('  OFFLINE (held-out) vs ONLINE (closed-loop), LSTM position:')
        print(f'  {"bin":<14}{"offline":>12}{"online":>12}{"ratio":>10}')
        print('  ' + '-' * 48)
        for name in out:
            if name in off:
                o = off[name]['lstm_pos']['mean']
                n_ = out[name]['lstm_pos']['mean']
                print(f'  {name:<14}{o:>12.4f}{n_:>12.4f}'
                      f'{(n_ / o if o > 0 else float("nan")):>10.2f}x')

    print()
    print('=' * 92)
    if verdict_ok:
        print('  VERDICT: the LSTM advantage SURVIVES closed-loop in the long')
        print('  blind bins. Proceed to the V3b TD3 run.')
    else:
        print('  VERDICT: the LSTM advantage does NOT survive closed-loop in')
        print('  at least one long blind bin. This is dataset shift. Report')
        print('  it and consider a DAgger-style collection round BEFORE')
        print('  spending hours on TD3.')
    print()
    print('  Reminder: the platform moves at constant velocity in this')
    print('  environment, so the constant-velocity predictor has the correct')
    print('  model class. Any LSTM advantage is denoising, multi-measurement')
    print('  integration and dropout bridging -- NOT evidence about')
    print('  non-constant target dynamics.')
    print('=' * 92)

    with open(args.out, 'w') as f:
        json.dump({'level': args.level, 'episodes': args.episodes,
                   'policy': args.policy,
                   'drive_with_lstm': bool(args.drive_with_lstm),
                   'successes': successes, 'bins': out}, f, indent=2)
    print(f'  wrote {args.out}')


if __name__ == '__main__':
    main()
