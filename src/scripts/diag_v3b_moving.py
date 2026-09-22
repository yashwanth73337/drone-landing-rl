"""
diag_v3b_moving.py
==================

Diagnostic, NO TRAINING. Answers one question about V3b on a moving pad:

    When the frozen-LSTM V3b policy descends beside a moving platform
    (below_platform), is that because the ESTIMATE was wrong, or because the
    POLICY descended knowing the pad was not underneath it?

Motivation (22-23 Sep 2026, best_success_model.zip @ 560k, seed 1,
radius 0.11-0.23 m, 100 episodes, seed base 9000):

    speed 0.00   85%   timeout 15   below_platform  0
    speed 0.05   73%   timeout 22   below_platform  5
    speed 0.10   44%   timeout 29   below_platform 27

The 'pos estimation error' printed by train_lstm_v3b.py --eval does not
answer this. v3a.evaluate_policy_full averages info['est_err_pos_norm'] --
the parent's CONSTANT-VELOCITY estimate -- and only on frames where the
marker was DETECTED. It never looks at the LSTM, and never at blind frames.


TWO INDEPENDENT CHECKS, ONE RUN
------------------------------------------------------------------------------
1. OBSERVATION-SOURCE SWAP (one variable).
   The SAME policy, SAME seeds, SAME task, flown with
       actor_obs_source = 'lstm'        what it was trained on
       actor_obs_source = 'privileged'  true relative position/velocity
   LSTMLanderAviary supports this natively; both estimators keep running
   passively either way. Nothing else changes.

       below_platform collapses under 'privileged'  -> estimate-driven
       below_platform persists under 'privileged'   -> policy-driven
                                                       (it cannot land on a
                                                        moving pad even when
                                                        told exactly where
                                                        it is)

   Caveat, stated in advance: 'privileged' is an input the policy never
   trained on. It is close to what the LSTM gives on visible frames, but a
   policy can react oddly to cleaner inputs. Read the outcome counts
   together with check 2, not alone.

2. BELIEF vs TRUTH AT THE MOMENT OF COMMITMENT.
   For every episode, over the last 30 steps (1 s at 30 Hz) before it ends:
       believed lateral offset   |LSTM d_xy|  (where the drone thinks the pad is)
       true lateral offset       |true d_xy|
       LSTM lateral error        |LSTM d_xy - true d_xy|
       blind fraction            of those 30 steps
   Grouped by outcome.

       below_platform: believed SMALL, true LARGE -> the estimate put the pad
                       under the drone when it was not (estimate-driven)
       below_platform: believed ~ true, both LARGE -> the drone knew it was
                       off the pad and descended anyway (policy-driven)

   Plus the LSTM's error binned by blind age (visible / 1-15 / 16-60 /
   61-120 / >120 steps since the last detection), the same bins used in
   the V3b report, next to the constant-velocity estimate on the SAME frames.

   Blind age uses info['est_detected'], the flag evaluate_policy_full already
   uses. d_true is recovered as LSTM d - LSTM error, from the info keys
   LSTMLanderAviary publishes (lstm_d_*, lstm_err_d*).


CONVENTIONS
------------------------------------------------------------------------------
Task is required on the command line (--radius-min, --radius-max, --speeds);
there is no default task. Seed base defaults to 9000 so the lstm arm
reproduces the train_lstm_v3b.py --eval numbers exactly -- check that it
does; if it does not, stop and tell me before reading anything else.

The estimator stays frozen. Nothing is trained, nothing is saved except
CSVs in a new directory under --outdir.


USAGE (from src/scripts)
------------------------------------------------------------------------------
    python diag_v3b_moving.py \\
        --model results_lstm_v3b/run_20260914_195837/best_success_model.zip \\
        --estimator ../models/temporal_estimator_v3b_r2.pt \\
        --radius-min 0.11 --radius-max 0.23 --speeds 0.10 \\
        --sources lstm privileged --episodes 100
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from stable_baselines3 import TD3

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.LSTMLanderAviary import LSTMLanderAviary          # noqa: E402
import train_lstm_v3b as v3b                                # noqa: E402


LAST_K = 30                     # steps before termination = 1 s at 30 Hz
NEAR_MISS_HEIGHT = 0.03         # identical to v3a.evaluate_policy_full
NEAR_MISS_STEPS = 10
AGE_BINS = [                    # the V3b report's blind-age bins
    (0, 0, 'visible'),
    (1, 15, 'blind 1-15'),
    (16, 60, 'blind 16-60'),
    (61, 120, 'blind 61-120'),
    (121, 10 ** 9, 'blind >120'),
]
BIN_ORDER = ['pre-detection'] + [b[2] for b in AGE_BINS]
OUTCOME_ORDER = ['success', 'below_platform', 'timeout', 'tilt_bound',
                 'position_bound', 'terminated_no_reason']


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float('nan')
    return v if math.isfinite(v) else float('nan')


def age_bin(age):
    if age is None:
        return 'pre-detection'
    for lo, hi, name in AGE_BINS:
        if lo <= age <= hi:
            return name
    return 'blind >120'


def nanstats(xs):
    a = np.asarray([x for x in xs if math.isfinite(x)], dtype=float)
    if a.size == 0:
        return {'n': 0, 'mean': float('nan'), 'p95': float('nan'),
                'max': float('nan')}
    return {'n': int(a.size), 'mean': float(a.mean()),
            'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def nanmean(xs):
    a = [x for x in xs if math.isfinite(x)]
    return float(np.mean(a)) if a else float('nan')


# ===========================================================================
# ONE CELL = one (actor_obs_source, speed)
# ===========================================================================

def run_cell(model, source, speed, args):
    kw = v3b.make_kwargs(args.radius_min, args.radius_max, speed,
                         os.path.abspath(args.estimator))
    kw['actor_obs_source'] = source         # the ONLY thing that differs
    env = LSTMLanderAviary(**kw)

    episodes = []
    by_age = defaultdict(lambda: {'lstm': [], 'lstm_lat': [], 'cv': []})

    try:
        for ep in range(args.episodes):
            seed = args.seed_base + ep
            obs, info = env.reset(seed=seed)
            age = None
            low_run = low_max = 0
            hist = []

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _r, terminated, truncated, info = env.step(action)

                det = bool(info.get('est_detected'))
                age = 0 if det else (None if age is None else age + 1)

                ld = np.array([fnum(info.get(f'lstm_d_{c}')) for c in 'xy'])
                le = np.array([fnum(info.get(f'lstm_err_d{c}')) for c in 'xy'])
                true_xy = ld - le
                believed_lat = float(np.hypot(*ld))
                true_lat = float(np.hypot(*true_xy))
                lstm_lat = float(np.hypot(*le))
                lstm_3d = fnum(info.get('lstm_err_pos_norm'))
                cv_3d = fnum(info.get('est_err_pos_norm'))

                b = age_bin(age)
                by_age[b]['lstm'].append(lstm_3d)
                by_age[b]['lstm_lat'].append(lstm_lat)
                by_age[b]['cv'].append(cv_3d)
                hist.append((det, believed_lat, true_lat, lstm_lat))

                h = fnum(info.get('height_above_pad'))
                if math.isfinite(h) and h < NEAR_MISS_HEIGHT:
                    low_run += 1
                    low_max = max(low_max, low_run)
                else:
                    low_run = 0

                if terminated or truncated:
                    break

            success = bool(info.get('success', False))
            outcome = ('success' if success else
                       (info.get('truncation_reason') or
                        'terminated_no_reason'))
            tail = hist[-LAST_K:]
            episodes.append({
                'source': source,
                'speed': speed,
                'seed': seed,
                'outcome': outcome,
                'near_miss': int((not success) and low_max >= NEAR_MISS_STEPS),
                'steps': len(hist),
                'blind_frac': round(
                    1.0 - sum(t[0] for t in hist) / max(len(hist), 1), 4),
                'tail_blind_frac': round(
                    1.0 - sum(t[0] for t in tail) / max(len(tail), 1), 4),
                'tail_believed_lat_m': round(nanmean(t[1] for t in tail), 5),
                'tail_true_lat_m': round(nanmean(t[2] for t in tail), 5),
                'tail_lstm_lat_err_m': round(nanmean(t[3] for t in tail), 5),
                'end_believed_lat_m': round(tail[-1][1], 5) if tail else '',
                'end_true_lat_m': round(tail[-1][2], 5) if tail else '',
                'end_lstm_lat_err_m': round(tail[-1][3], 5) if tail else '',
                'end_height_m': round(fnum(info.get('height_above_pad')), 5),
                'end_d_target': round(fnum(info.get('d_target')), 5),
            })
    finally:
        env.close()

    return episodes, by_age


# ===========================================================================
# REPORTING
# ===========================================================================

def print_outcomes(cells):
    print('\n' + '=' * 78)
    print('  1. OUTCOMES -- same policy, same seeds, only actor_obs_source differs')
    print('=' * 78)
    print(f'  {"source":<11}{"speed":>6}{"success":>9}{"timeout":>9}'
          f'{"below_pl":>10}{"other":>7}{"near-miss":>11}{"blind":>8}')
    for (source, speed), (eps, _) in cells.items():
        c = Counter(e['outcome'] for e in eps)
        n = len(eps)
        other = n - c['success'] - c['timeout'] - c['below_platform']
        print(f'  {source:<11}{speed:>6.2f}'
              f'{100.0 * c["success"] / n:>8.1f}%'
              f'{c["timeout"]:>9}{c["below_platform"]:>10}{other:>7}'
              f'{sum(e["near_miss"] for e in eps):>11}'
              f'{100.0 * np.mean([e["blind_frac"] for e in eps]):>7.1f}%')


def print_commitment(cells):
    print('\n' + '=' * 78)
    print(f'  2. BELIEF vs TRUTH over the last {LAST_K} steps, by outcome '
          f'(means, metres)')
    print('=' * 78)
    print('  below_platform with believed SMALL and true LARGE  -> estimate')
    print('  below_platform with believed ~ true, both LARGE    -> policy')
    print(f'\n  {"source":<11}{"speed":>6}  {"outcome":<16}{"n":>4}'
          f'{"blind":>8}{"believed":>10}{"true":>9}{"LSTM err":>10}')
    for (source, speed), (eps, _) in cells.items():
        groups = defaultdict(list)
        for e in eps:
            groups[e['outcome']].append(e)
        for oc in OUTCOME_ORDER:
            g = groups.get(oc)
            if not g:
                continue
            print(f'  {source:<11}{speed:>6.2f}  {oc:<16}{len(g):>4}'
                  f'{100.0 * np.mean([e["tail_blind_frac"] for e in g]):>7.1f}%'
                  f'{nanmean(e["tail_believed_lat_m"] for e in g):>10.4f}'
                  f'{nanmean(e["tail_true_lat_m"] for e in g):>9.4f}'
                  f'{nanmean(e["tail_lstm_lat_err_m"] for e in g):>10.4f}')


def print_by_age(cells):
    print('\n' + '=' * 78)
    print('  3. ESTIMATION ERROR BY BLIND AGE (3-D position error, metres)')
    print('     LSTM = what the actor sees under source=lstm.')
    print('     CV   = constant-velocity estimate on the SAME frames (passive).')
    print('=' * 78)
    print(f'  {"source":<11}{"speed":>6}  {"bin":<14}{"frames":>8}'
          f'{"LSTM mean":>11}{"p95":>8}{"max":>8}'
          f'{"CV mean":>10}{"p95":>8}{"max":>8}')
    for (source, speed), (_, by_age) in cells.items():
        for b in BIN_ORDER:
            if b not in by_age:
                continue
            l = nanstats(by_age[b]['lstm'])
            c = nanstats(by_age[b]['cv'])
            if l['n'] == 0:
                continue
            print(f'  {source:<11}{speed:>6.2f}  {b:<14}{l["n"]:>8}'
                  f'{l["mean"]:>11.4f}{l["p95"]:>8.3f}{l["max"]:>8.3f}'
                  f'{c["mean"]:>10.4f}{c["p95"]:>8.3f}{c["max"]:>8.3f}')


def write_csvs(out_dir, cells):
    all_eps = [e for eps, _ in cells.values() for e in eps]
    with open(os.path.join(out_dir, 'episodes.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_eps[0].keys()))
        w.writeheader()
        w.writerows(all_eps)

    with open(os.path.join(out_dir, 'error_by_blind_age.csv'), 'w',
              newline='') as f:
        w = csv.writer(f)
        w.writerow(['source', 'speed', 'bin', 'frames',
                    'lstm_mean_m', 'lstm_p95_m', 'lstm_max_m',
                    'lstm_lat_mean_m', 'lstm_lat_p95_m', 'lstm_lat_max_m',
                    'cv_mean_m', 'cv_p95_m', 'cv_max_m'])
        for (source, speed), (_, by_age) in cells.items():
            for b in BIN_ORDER:
                if b not in by_age:
                    continue
                l = nanstats(by_age[b]['lstm'])
                ll = nanstats(by_age[b]['lstm_lat'])
                c = nanstats(by_age[b]['cv'])
                w.writerow([source, speed, b, l['n'],
                            *(round(x, 5) for x in (l['mean'], l['p95'],
                                                    l['max'])),
                            *(round(x, 5) for x in (ll['mean'], ll['p95'],
                                                    ll['max'])),
                            *(round(x, 5) for x in (c['mean'], c['p95'],
                                                    c['max']))])


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    ap.add_argument('--model', required=True)
    ap.add_argument('--estimator', required=True)
    ap.add_argument('--radius-min', type=float, required=True)
    ap.add_argument('--radius-max', type=float, required=True)
    ap.add_argument('--speeds', type=float, nargs='+', required=True)
    ap.add_argument('--sources', nargs='+', default=['lstm', 'privileged'],
                    choices=['lstm', 'privileged', 'predict'])
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed-base', type=int, default=9000)
    ap.add_argument('--outdir', default='eval_logs')
    args = ap.parse_args()

    for p in (args.model, args.estimator):
        if not os.path.exists(p):
            print(f'ERROR: not found: {p}')
            sys.exit(1)

    out_dir = os.path.join(args.outdir,
                           time.strftime('diag_moving_%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 78)
    print('  V3b MOVING-PAD DIAGNOSTIC -- no training')
    print('=' * 78)
    print(f'  model     : {args.model}')
    print(f'  estimator : {os.path.basename(args.estimator)} (FROZEN)')
    print(f'  task      : radius {args.radius_min}-{args.radius_max} m, '
          f'speeds {args.speeds} m/s')
    print(f'  sources   : {args.sources}')
    print(f'  episodes  : {args.episodes} per cell (seeds {args.seed_base}..)')
    print(f'  output    : {out_dir}')
    print('=' * 78)

    model = TD3.load(args.model)
    cells = {}
    for speed in args.speeds:
        for source in args.sources:
            t0 = time.time()
            print(f'\n  running source={source} speed={speed:.2f} ...',
                  flush=True)
            cells[(source, speed)] = run_cell(model, source, speed, args)
            eps = cells[(source, speed)][0]
            c = Counter(e['outcome'] for e in eps)
            print(f'  done in {(time.time() - t0) / 60:.1f} min: '
                  f'{dict(c)}', flush=True)

    print_outcomes(cells)
    print_commitment(cells)
    print_by_age(cells)
    write_csvs(out_dir, cells)

    print('\n' + '=' * 78)
    print(f'  CSVs: {out_dir}/episodes.csv, {out_dir}/error_by_blind_age.csv')
    print('  CHECK FIRST: the lstm rows in table 1 must match the')
    print('  train_lstm_v3b.py --eval result for the same model/task/seeds.')
    print('=' * 78)


if __name__ == '__main__':
    main()
