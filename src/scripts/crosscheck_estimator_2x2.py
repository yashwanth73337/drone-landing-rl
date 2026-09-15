"""
crosscheck_estimator_2x2.py
===========================

Separates TASK/SPEED distribution shift from POLICY/TRAJECTORY distribution
shift, which the first closed-loop check confounded by changing both at once.

    privileged policy @ level 9      privileged policy @ level 12
    V3a policy        @ level 9      V3a policy        @ level 12

THE LSTM IS PASSIVE IN ALL FOUR CELLS.

Each policy flies on its OWN normal observation -- privileged checkpoints on
privileged state, V3a checkpoints on the constant-velocity estimate -- set
explicitly via actor_obs_source, never inferred. Both estimators run on the
same camera stream for diagnostics only. Neither can alter the trajectory, so
closed-loop estimator feedback cannot contaminate the comparison.

Reading the result:

    degradation follows LEVEL 12 under both policies
        -> shortage of high-speed (levels 11-12) training data

    degradation follows the V3a POLICY under both levels
        -> policy-induced dataset shift; DAgger is justified

    both
        -> collect stratified across level AND policy


MOTION CONDITIONING -- why blind age alone is not enough
------------------------------------------------------------------------------
The level-9 closed-loop result showed LSTM error DECREASING with blind age
(0.091 m at 61-120 steps, 0.073 m beyond 120), which is not how an estimator
normally behaves. The likely explanation is that at level 9 the V3a policy
hovers, so its long blind stretches occur while the relative state is barely
changing -- an estimator that has settled into "nothing is moving" is
trivially correct, and that looks like skill.

So error is reported conditioned on blind age AND on relative-motion
magnitude. If the LSTM only wins in low-motion bins, the advantage is an
artefact of when dropout happens rather than evidence it bridges dropout.

Per blind stretch this records: length, mean and max ground-truth |delta_v|,
mean ground-truth relative-state change rate ||d(t) - d(t-1)|| / dt, platform
speed, curriculum level and policy source.

USAGE
    python crosscheck_estimator_2x2.py \\
        --estimator ../models/temporal_estimator_v3b.pt \\
        --privileged results_lander/moving_025_seed1_final.zip \\
        --v3a results_aruco_v3a_cont/cont_20260913_150310/best_success_model.zip \\
        --episodes 30
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.LSTMLanderAviary import LSTMLanderAviary          # noqa: E402
import train_aruco_v3a as v3a                               # noqa: E402

DT = 1.0 / 30.0

AGE_BINS = [('visible', 0, 0), ('1-15', 1, 15), ('16-60', 16, 60),
            ('61-120', 61, 120), ('>120', 121, 10 ** 9)]

# Ground-truth |delta_v|, m/s. The platform tops out at ~0.35 m/s
# (0.25 per axis), and the drone's own motion adds to the relative term.
MOTION_BINS = [('still <0.05', 0.0, 0.05), ('slow 0.05-0.15', 0.05, 0.15),
               ('med 0.15-0.30', 0.15, 0.30), ('fast >0.30', 0.30, 1e9)]


def run_cell(estimator, policy_path, level, obs_source, episodes, seed_base):
    from stable_baselines3 import TD3
    model = TD3.load(policy_path)

    radius_max, speed = v3a.CURRICULUM_LEVELS[level - 1]
    kw = v3a.make_env_kwargs(v3a.RADIUS_MIN, radius_max, speed)
    kw.pop('strict_no_privileged', None)
    kw['estimator_path'] = estimator
    kw['actor_obs_source'] = obs_source      # EXPLICIT
    kw['verbose_source'] = False

    env = LSTMLanderAviary(**kw)
    steps, stretches = [], []
    successes = 0

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed_base + ep)
            prev_d = None
            cur = None

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)

                age = int(info.get('est_steps_since_detection', 0))
                dv = float(info.get('gt_dv_norm', np.nan))
                d_vec = np.array([info['est_d_true_x'], info['est_d_true_y'],
                                  info['est_d_true_z']])
                rate = (np.linalg.norm(d_vec - prev_d) / DT
                        if prev_d is not None else np.nan)
                prev_d = d_vec

                steps.append({
                    'age': age, 'gt_dv': dv, 'rate': rate,
                    'lstm_p': info['lstm_err_pos_norm'],
                    'lstm_v': info['lstm_err_vel_norm'],
                    'cv_p': info.get('cv_err_pos_norm'),
                    'cv_v': info.get('cv_err_vel_norm'),
                })

                # per-blind-stretch accounting
                if age > 0:
                    if cur is None:
                        cur = {'len': 0, 'dv': [], 'rate': [],
                               'lstm_p': [], 'cv_p': []}
                    cur['len'] += 1
                    cur['dv'].append(dv)
                    if np.isfinite(rate):
                        cur['rate'].append(rate)
                    cur['lstm_p'].append(info['lstm_err_pos_norm'])
                    cur['cv_p'].append(info.get('cv_err_pos_norm'))
                elif cur is not None:
                    stretches.append(cur)
                    cur = None

                if term or trunc:
                    if cur is not None:
                        stretches.append(cur)
                    successes += bool(info.get('success', False))
                    break
    finally:
        env.close()

    return steps, stretches, successes


def agg(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    a = np.asarray(vals)
    return {'n': int(a.size), 'mean': float(a.mean()),
            'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def age_table(steps, title):
    print(f'\n  {title}')
    print(f'  {"blind age":<12}{"n":>7}{"CV pos":>10}{"LSTM pos":>10}'
          f'{"ratio":>8}{"CV vel":>10}{"LSTM vel":>10}')
    print('  ' + '-' * 57)
    out = {}
    for name, lo, hi in AGE_BINS:
        sel = [s for s in steps if lo <= s['age'] <= hi]
        cp, lp = agg([s['cv_p'] for s in sel]), agg([s['lstm_p'] for s in sel])
        cv, lv = agg([s['cv_v'] for s in sel]), agg([s['lstm_v'] for s in sel])
        if lp is None:
            continue
        ratio = lp['mean'] / cp['mean'] if cp and cp['mean'] > 0 else np.nan
        out[name] = {'cv_pos': cp, 'lstm_pos': lp,
                     'cv_vel': cv, 'lstm_vel': lv}
        flag = '' if ratio < 1.0 else '  <-- LSTM worse'
        print(f'  {name:<12}{lp["n"]:>7}{cp["mean"]:>10.4f}'
              f'{lp["mean"]:>10.4f}{ratio:>8.2f}'
              f'{cv["mean"]:>10.4f}{lv["mean"]:>10.4f}{flag}')
    return out


def motion_table(steps, title):
    """Blind frames only, conditioned on ground-truth relative motion."""
    print(f'\n  {title}  (BLIND frames only, conditioned on |delta_v|)')
    print(f'  {"motion":<18}{"age":<10}{"n":>7}{"CV pos":>10}'
          f'{"LSTM pos":>10}{"ratio":>8}')
    print('  ' + '-' * 63)
    out = {}
    for mname, mlo, mhi in MOTION_BINS:
        for aname, alo, ahi in (('1-60', 1, 60), ('>60', 61, 10 ** 9)):
            sel = [s for s in steps
                   if alo <= s['age'] <= ahi
                   and np.isfinite(s['gt_dv'])
                   and mlo <= s['gt_dv'] < mhi]
            cp, lp = (agg([s['cv_p'] for s in sel]),
                      agg([s['lstm_p'] for s in sel]))
            if lp is None or lp['n'] < 20:
                continue
            ratio = lp['mean'] / cp['mean'] if cp and cp['mean'] > 0 else np.nan
            out[f'{mname}|{aname}'] = {'cv_pos': cp, 'lstm_pos': lp}
            flag = '' if ratio < 1.0 else '  <-- LSTM worse'
            print(f'  {mname:<18}{aname:<10}{lp["n"]:>7}'
                  f'{cp["mean"]:>10.4f}{lp["mean"]:>10.4f}'
                  f'{ratio:>8.2f}{flag}')
    return out


def stretch_summary(stretches, title):
    if not stretches:
        return {}
    print(f'\n  {title}  ({len(stretches)} blind stretches)')
    lens = np.array([s['len'] for s in stretches])
    print(f'    length      mean {lens.mean():6.1f}  p95 '
          f'{np.percentile(lens, 95):6.1f}  max {lens.max():5d}')
    for label, key in (('mean |dv|', 'dv'), ('rel. rate', 'rate')):
        m = np.array([np.mean(s[key]) if s[key] else np.nan
                      for s in stretches])
        m = m[np.isfinite(m)]
        if m.size:
            print(f'    {label:<11} mean {m.mean():6.4f}  p95 '
                  f'{np.percentile(m, 95):6.4f}  max {m.max():6.4f}')
    long_s = [s for s in stretches if s['len'] > 60]
    print(f'    stretches >60 steps: {len(long_s)}')
    if long_s:
        dv = np.array([np.mean(s['dv']) for s in long_s])
        print(f'      their mean |dv|: {dv.mean():.4f} m/s   '
              f'(vs {np.mean([np.mean(s["dv"]) for s in stretches]):.4f} '
              f'over all stretches)')
    return {'n': len(stretches), 'len_mean': float(lens.mean()),
            'len_max': int(lens.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--estimator', required=True)
    ap.add_argument('--privileged', required=True)
    ap.add_argument('--v3a', required=True)
    ap.add_argument('--episodes', type=int, default=30)
    ap.add_argument('--seed-base', type=int, default=51000)
    ap.add_argument('--out', default='v3b_crosscheck_2x2.json')
    args = ap.parse_args()

    cells = [
        ('privileged @ L9', args.privileged, 9, 'privileged'),
        ('privileged @ L12', args.privileged, 12, 'privileged'),
        ('v3a @ L9', args.v3a, 9, 'predict'),
        ('v3a @ L12', args.v3a, 12, 'predict'),
    ]

    print('=' * 78)
    print('  V3b 2x2 CROSS-CHECK -- LSTM PASSIVE IN ALL CELLS')
    print('=' * 78)
    print('  Each policy flies on its OWN observation (actor_obs_source set')
    print('  explicitly). Both estimators run on the same camera stream for')
    print('  diagnostics only; neither alters the trajectory.')
    print(f'  episodes per cell: {args.episodes}')

    results, summary = {}, {}
    for label, path, level, src in cells:
        print()
        print('=' * 78)
        print(f'  CELL: {label}   actor_obs_source={src.upper()}')
        print('=' * 78)
        steps, stretches, succ = run_cell(
            args.estimator, path, level, src, args.episodes, args.seed_base)
        print(f'  policy success: {succ}/{args.episodes}   '
              f'steps recorded: {len(steps)}')

        a = age_table(steps, 'By blind age')
        m = motion_table(steps, 'Motion-conditioned')
        st = stretch_summary(stretches, 'Blind-stretch statistics')

        # Pooled long-blind error is NOT a safe verdict. In a cell where
        # long dropout coincides with near-stationary relative motion, the
        # still-bin frames dominate the pool and drag the mean down, so a
        # cell that loses in every moving bin can still report "LSTM
        # better". The verdict is therefore computed PER MOTION BIN.
        long_sel = [s for s in steps if s['age'] > 60]
        cp, lp = (agg([s['cv_p'] for s in long_sel]),
                  agg([s['lstm_p'] for s in long_sel]))

        lose = [k for k, v in m.items()
                if v['cv_pos'] and v['lstm_pos']
                and v['lstm_pos']['mean'] >= v['cv_pos']['mean']]
        # Bins where the relative state is genuinely moving -- the only ones
        # that test dropout bridging rather than "nothing changed".
        moving = {k: v for k, v in m.items() if not k.startswith('still')}
        lose_moving = [k for k, v in moving.items()
                       if v['cv_pos'] and v['lstm_pos']
                       and v['lstm_pos']['mean'] >= v['cv_pos']['mean']]

        summary[label] = {
            'success': f'{succ}/{args.episodes}',
            'long_blind_cv_pos_POOLED': cp['mean'] if cp else None,
            'long_blind_lstm_pos_POOLED': lp['mean'] if lp else None,
            'motion_bins': len(m),
            'motion_bins_lost': len(lose),
            'moving_bins': len(moving),
            'moving_bins_lost': len(lose_moving),
            'lost_bins': lose,
            'pass': len(lose_moving) == 0 and len(moving) > 0,
        }
        results[label] = {'age': a, 'motion': m, 'stretches': st}

    print()
    print('=' * 78)
    print('  SUMMARY -- verdict is PER MOTION BIN, not pooled')
    print('=' * 78)
    print(f'  {"cell":<20}{"success":>9}{"moving bins":>13}{"lost":>7}'
          f'{"pooled ratio":>14}   verdict')
    print('  ' + '-' * 76)
    for label, sm in summary.items():
        cv_, l_ = (sm['long_blind_cv_pos_POOLED'],
                   sm['long_blind_lstm_pos_POOLED'])
        pooled = (f'{l_ / cv_:.2f}' if (cv_ and l_ and cv_ > 0) else 'n/a')
        if sm['moving_bins'] == 0:
            verdict = 'no moving bins'
        elif sm['pass']:
            verdict = 'PASS'
        else:
            verdict = f'FAIL ({sm["moving_bins_lost"]} moving bins lost)'
        print(f'  {label:<20}{sm["success"]:>9}{sm["moving_bins"]:>13}'
              f'{sm["moving_bins_lost"]:>7}{pooled:>14}   {verdict}')
    print('  ' + '-' * 76)
    print()
    print('  NOTE ON THE POOLED COLUMN. It is shown for reference only and')
    print('  must not be used as the verdict. Where long dropout coincides')
    print('  with a near-stationary relative state, the still-motion frames')
    print('  dominate the pool -- a cell can lose in every moving bin and')
    print('  still show a pooled ratio below 1.0. The 2x2 run that motivated')
    print('  this change did exactly that.')
    print()
    for label, sm in summary.items():
        if sm['lost_bins']:
            print(f'  {label}: lost in {sm["lost_bins"]}')
    print()
    print('  Degradation confined to L12 cells        -> high-speed data gap')
    print('  Degradation confined to v3a cells        -> policy-induced shift')
    print('  Degradation only where BOTH coincide     -> the hard regime is')
    print('    long dropout AT high platform speed, and it is missing from')
    print('    the training mixture. Collect stratified across both.')
    print('=' * 78)

    with open(args.out, 'w') as f:
        json.dump({'summary': summary, 'detail': results}, f,
                  indent=2, default=str)
    print(f'  wrote {args.out}')


if __name__ == '__main__':
    main()
