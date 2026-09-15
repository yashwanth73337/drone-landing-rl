"""
compare_error_distributions.py
==============================

Tests the hypothesis left open in V3b_REPORT.md section 5.4.

THE PUZZLE
------------------------------------------------------------------------------
V3b beat V3a -- level 5 vs 4, 85.0% vs 73.0% on own task, and
`below_platform` failures eliminated (8 -> 0 per 100 episodes). But its
estimator showed NO measurable mean-accuracy advantage in deployment:

    closed-loop position error   V3a (CV) 0.0120 m   V3b (LSTM) 0.0118 m
    closed-loop velocity error   V3a (CV) 0.180-0.226 m/s   V3b 0.206 m/s

So the improvement cannot be attributed to "better estimates" on mean error.

THE HYPOTHESIS
------------------------------------------------------------------------------
The LSTM's estimate is better BEHAVED rather than more accurate: smoother,
without the large excursions unbounded constant-velocity integration
produces. Offline this was visible -- at >120 blind age, CV's p95 reached
7.2 m against the LSTM's 0.86 m, while the means differed by only ~4x. A mean
hides that completely, and a policy that descends onto one catastrophic
outlier crashes regardless of how good the average was.

That would explain the elimination of `below_platform` failures without any
improvement in mean accuracy.

WHAT THIS MEASURES
------------------------------------------------------------------------------
Both estimators run side by side on the SAME camera stream, in the same
episodes, driven by the same policy -- LSTMLanderAviary keeps the
constant-velocity estimator live regardless of what the actor consumes, so
this is a paired comparison on identical frames, not two separate runs.

Reported per blind-age bin:

    mean, p50, p90, p95, p99, max
    fraction of frames exceeding 0.1 m  (the success radius)
    fraction exceeding 0.3 m and 1.0 m  (excursions that would cause a crash)

The tail statistics are the point. If the LSTM's p99 and max are much lower
at equal mean, the hypothesis holds. If the tails match too, the mechanism
behind V3b's advantage is NOT estimator behaviour and the report should say
so plainly rather than keep a hypothesis it cannot support.

NOTE ON WHAT IS BEING COMPARED
------------------------------------------------------------------------------
This runs BOTH estimators under ONE policy at a time, so it isolates the
estimators on matched frames. It does not reproduce V3a's and V3b's own
trajectories -- those differ, and comparing errors across different
trajectories would confound estimator behaviour with policy behaviour.

Run it with the V3b policy (the states V3b actually visits) and again with
the V3a policy, and report both.

USAGE
    python compare_error_distributions.py \\
        --estimator ../models/temporal_estimator_v3b_r2.pt \\
        --policy results_lstm_v3b/run_20260914_195837/best_success_model.zip \\
        --obs-source lstm --radius-max 0.23 --speed 0.0 --episodes 50 \\
        --out dist_v3b_policy.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.LSTMLanderAviary import LSTMLanderAviary          # noqa: E402
import train_aruco_v3a as v3a                               # noqa: E402

AGE_BINS = [('visible', 0, 0), ('1-15', 1, 15), ('16-60', 16, 60),
            ('61-120', 61, 120), ('>120', 121, 10 ** 9), ('ALL blind', 1, 10 ** 9)]

# Thresholds chosen against the task, not the data: 0.1 m is the success
# radius, so an error above it means the policy cannot know it is on target;
# 0.3 m and 1.0 m are excursions large enough to drive a descent into empty
# space.
THRESHOLDS = [0.1, 0.3, 1.0]


def dist(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)])
    if a.size == 0:
        return None
    d = {'n': int(a.size), 'mean': float(a.mean()),
         'p50': float(np.percentile(a, 50)),
         'p90': float(np.percentile(a, 90)),
         'p95': float(np.percentile(a, 95)),
         'p99': float(np.percentile(a, 99)),
         'max': float(a.max())}
    for t in THRESHOLDS:
        d[f'frac_gt_{t}'] = float((a > t).mean())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--estimator', required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--obs-source', default='lstm',
                    choices=['privileged', 'predict', 'lstm'],
                    help="what the ACTOR consumes. Use 'lstm' for a V3b "
                         "policy, 'predict' for a V3a policy -- each must "
                         "fly on the observation it was trained with.")
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--seed-base', type=int, default=9000)
    ap.add_argument('--radius-min', type=float, default=0.11)
    ap.add_argument('--radius-max', type=float, default=0.23)
    ap.add_argument('--speed', type=float, default=0.0)
    ap.add_argument('--out', default='error_distributions.json')
    args = ap.parse_args()

    from stable_baselines3 import TD3
    model = TD3.load(args.policy)

    kw = v3a.make_env_kwargs(args.radius_min, args.radius_max, args.speed)
    kw.pop('estimator_mode', None)
    kw.pop('strict_no_privileged', None)
    kw['estimator_path'] = args.estimator
    kw['actor_obs_source'] = args.obs_source
    kw['verbose_source'] = False

    print('=' * 96)
    print('  CLOSED-LOOP ERROR DISTRIBUTIONS -- CV vs LSTM on identical frames')
    print('=' * 96)
    print(f'  policy      : {os.path.basename(args.policy)}')
    print(f'  actor input : {args.obs_source.upper()}')
    print(f'  task        : radius {args.radius_min}-{args.radius_max} m, '
          f'speed {args.speed} m/s')
    print(f'  episodes    : {args.episodes}')
    print('  Both estimators run on the SAME camera stream in the SAME')
    print('  episodes -- this is a paired comparison, not two runs.')
    print()

    env = LSTMLanderAviary(**kw)
    rows = []
    successes = 0
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed_base + ep)
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                rows.append({
                    'age': int(info.get('est_steps_since_detection', 0)),
                    'lstm_p': info.get('lstm_err_pos_norm'),
                    'cv_p': info.get('cv_err_pos_norm'),
                    'lstm_v': info.get('lstm_err_vel_norm'),
                    'cv_v': info.get('cv_err_vel_norm'),
                })
                if term or trunc:
                    successes += bool(info.get('success', False))
                    break
    finally:
        env.close()

    print(f'  policy success: {successes}/{args.episodes}   '
          f'frames: {len(rows)}')

    out = {}
    for name, lo, hi in AGE_BINS:
        sel = [r for r in rows if lo <= r['age'] <= hi]
        cp, lp = (dist([r['cv_p'] for r in sel]),
                  dist([r['lstm_p'] for r in sel]))
        if cp is None or lp is None:
            continue
        out[name] = {'cv_pos': cp, 'lstm_pos': lp,
                     'cv_vel': dist([r['cv_v'] for r in sel]),
                     'lstm_vel': dist([r['lstm_v'] for r in sel])}

        print()
        print(f'  --- {name}   (n = {cp["n"]}) '
              + '-' * max(0, 58 - len(name)))
        print(f'  {"position (m)":<16}{"mean":>9}{"p50":>9}{"p90":>9}'
              f'{"p95":>9}{"p99":>9}{"max":>10}')
        for lbl, d in (('CV', cp), ('LSTM', lp)):
            print(f'  {lbl:<16}{d["mean"]:>9.4f}{d["p50"]:>9.4f}'
                  f'{d["p90"]:>9.4f}{d["p95"]:>9.4f}{d["p99"]:>9.4f}'
                  f'{d["max"]:>10.4f}')
        rm = lp['mean'] / cp['mean'] if cp['mean'] > 0 else np.nan
        r99 = lp['p99'] / cp['p99'] if cp['p99'] > 0 else np.nan
        rmx = lp['max'] / cp['max'] if cp['max'] > 0 else np.nan
        print(f'  {"LSTM/CV ratio":<16}{rm:>9.2f}{"":>9}{"":>9}{"":>9}'
              f'{r99:>9.2f}{rmx:>10.2f}')
        print(f'  {"frac > 0.1 m":<16}  CV {cp["frac_gt_0.1"]:.3f}'
              f'   LSTM {lp["frac_gt_0.1"]:.3f}')
        print(f'  {"frac > 0.3 m":<16}  CV {cp["frac_gt_0.3"]:.3f}'
              f'   LSTM {lp["frac_gt_0.3"]:.3f}')
        print(f'  {"frac > 1.0 m":<16}  CV {cp["frac_gt_1.0"]:.3f}'
              f'   LSTM {lp["frac_gt_1.0"]:.3f}')

    print()
    print('=' * 96)
    ab = out.get('ALL blind')
    if ab:
        mr = ab['lstm_pos']['mean'] / max(ab['cv_pos']['mean'], 1e-9)
        tr = ab['lstm_pos']['p99'] / max(ab['cv_pos']['p99'], 1e-9)
        print(f'  Over all blind frames:  mean ratio {mr:.2f}   '
              f'p99 ratio {tr:.2f}   max ratio '
              f'{ab["lstm_pos"]["max"] / max(ab["cv_pos"]["max"], 1e-9):.2f}')
        print()
        if tr < mr * 0.6:
            print('  TAIL MUCH SHORTER THAN THE MEAN SUGGESTS.')
            print('  Consistent with V3b section 5.4: the LSTM advantage is')
            print('  in error BEHAVIOUR, not accuracy. Large excursions are')
            print('  what drive descent into empty space, and they are what')
            print('  the elimination of below_platform failures looks like.')
        elif tr > mr * 1.4:
            print('  TAIL RELATIVELY WORSE THAN THE MEAN.')
            print('  This CONTRADICTS the section 5.4 hypothesis.')
        else:
            print('  TAIL AND MEAN SCALE TOGETHER.')
            print('  The section 5.4 hypothesis is NOT supported: error')
            print("  behaviour does not explain V3b's advantage, and the")
            print('  report should say the mechanism is unexplained rather')
            print('  than keep a hypothesis the data does not carry.')
    print('=' * 96)

    with open(args.out, 'w') as f:
        json.dump({'policy': args.policy, 'obs_source': args.obs_source,
                   'episodes': args.episodes, 'successes': successes,
                   'bins': out}, f, indent=2)
    print(f'  wrote {args.out}')


if __name__ == '__main__':
    main()
