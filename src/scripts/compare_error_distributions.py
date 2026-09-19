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
    print('  VERDICT -- suppression of crash-scale excursions, per blind-age bin')
    print('=' * 96)
    print()
    print('  below_platform is caused by a SINGLE large position error during')
    print('  descent, not by average error. The measurement is therefore the')
    print('  fraction of frames exceeding a crash-relevant threshold.')
    print()
    print('  Reported per bin and never pooled: section 4.2 records a pooled')
    print('  check passing while the estimator lost in four of eight bins.')
    print()

    DANGER = [0.3, 1.0]
    cells = {}

    print(f'  {"bin":<12}{"thresh":>8}{"CV":>9}{"LSTM":>9}{"abs drop":>11}'
          f'   verdict')
    print('  ' + '-' * 64)
    for name, _, _ in AGE_BINS:
        if name in ('visible', 'ALL blind') or name not in out:
            continue
        for t in DANGER:
            cvf = out[name]['cv_pos'][f'frac_gt_{t}']
            lsf = out[name]['lstm_pos'][f'frac_gt_{t}']
            drop = cvf - lsf
            if cvf < 0.05:
                v = 'n/a  (CV already <5%)'
            elif lsf <= 0.5 * cvf:
                v = 'SUPPRESSED'
                cells[(name, t)] = True
            elif lsf >= cvf:
                v = 'NOT suppressed'
                cells[(name, t)] = False
            else:
                v = 'partial'
                cells[(name, t)] = None
            print(f'  {name:<12}{t:>8.1f}{cvf:>9.3f}{lsf:>9.3f}{drop:>11.3f}'
                  f'   {v}')

    won = sum(1 for v in cells.values() if v is True)
    lost = sum(1 for v in cells.values() if v is False)
    part = sum(1 for v in cells.values() if v is None)

    print()
    if not cells:
        print('  NO TESTABLE CELLS. CV never exceeds the crash thresholds here,')
        print('  so the mechanism cannot be tested on these frames.')
    elif lost == 0 and won > 0:
        print(f'  SUPPORTED.  LSTM suppresses crash-scale excursions in {won} of')
        print(f'  {len(cells)} testable cells and loses in none ({part} partial).')
        print('  The section 5.4 mechanism is carried by the tail of the error')
        print('  distribution, not its centre.')
    elif won > lost:
        print(f'  PARTIALLY SUPPORTED.  wins {won}, loses {lost}, partial {part}.')
        print('  Report per bin. Do not state the mechanism unconditionally.')
    else:
        print(f'  NOT SUPPORTED.  wins {won}, loses {lost}, partial {part}.')
        print('  Error behaviour does not explain the below_platform result and')
        print('  the report should record the mechanism as unexplained.')

    ab = out.get('ALL blind')
    if ab:
        print()
        print('  Context only (pooled across blind ages -- NOT a verdict):')
        mr = ab['lstm_pos']['mean'] / max(ab['cv_pos']['mean'], 1e-9)
        tr = ab['lstm_pos']['p99'] / max(ab['cv_pos']['p99'], 1e-9)
        mx = ab['lstm_pos']['max'] / max(ab['cv_pos']['max'], 1e-9)
        print(f'    LSTM/CV all blind frames:  mean {mr:.2f}   p99 {tr:.2f}'
              f'   max {mx:.2f}')
        print(f'    worst single error:  CV {ab["cv_pos"]["max"]:.3f} m'
              f'   LSTM {ab["lstm_pos"]["max"]:.3f} m')
    print('=' * 96)

    with open(args.out, 'w') as f:
        json.dump({'policy': args.policy, 'obs_source': args.obs_source,
            'episodes': args.episodes, 'successes': successes,
            'excursion_cells': {f'{k[0]}|{k[1]}': v
                                for k, v in cells.items()},
            'bins': out}, f, indent=2)
    print(f'  wrote {args.out}')


if __name__ == '__main__':
    main()
