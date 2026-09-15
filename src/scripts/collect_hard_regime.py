"""
collect_hard_regime.py
======================

Targeted second collection round, filling the specific gap the 2x2 cross-check
identified.

THE GAP
------------------------------------------------------------------------------
The cross-check isolated the failure to a single cell: the V3a policy at level
12. Three of four cells were clean in every motion bin; that one lost in four
of eight, and lost specifically wherever the relative state was genuinely
moving:

    still <0.05,  >60 steps   ratio 0.18   LSTM wins (nothing is moving)
    slow 0.05-0.15, 1-60      ratio 1.38   LSTM LOSES
    med  0.15-0.30, >60       ratio 1.21   LSTM LOSES
    fast >0.30,     >60       ratio 1.03   LSTM LOSES

So the hard regime is the CONJUNCTION: long visual dropout WHILE the relative
state is moving fast. Neither factor alone breaks the estimator.

The original dataset contains almost none of it, and the reason is structural
rather than an oversight:

    privileged sources at levels 11-12 LAND TOO FAST to go blind for long.
        30 episodes at L12 produced only 5 stretches over 60 steps, max 94.

    V3a sources that hover for hundreds of steps live at levels 3-10, where
        the platform is slow or stationary.
        At L9 their >60-step stretches averaged |delta_v| = 0.089 m/s against
        0.161 m/s over all stretches -- long dropout there happens precisely
        when nothing is moving.

The one configuration that produces both at once is the V3a policy at levels
11-12, which fails slowly at high platform speed: 29 stretches over 60 steps,
max 280, mean |delta_v| 0.187 m/s.

WHAT THIS SCRIPT DOES
------------------------------------------------------------------------------
Collects episodes concentrated on that configuration, in the SAME format as
collect_estimator_data.py so the two datasets merge directly. It does NOT
replace the original dataset -- the estimator still needs the easy regimes, and
training only on hard cases would trade one distribution gap for another.

Episodes are filtered on a measured criterion rather than assumed to be hard:
an episode is kept only if it contains at least one blind stretch of
`--min-blind` steps during which mean ground-truth |delta_v| exceeds
`--min-motion`. Rejected episodes are counted and reported, so the yield of
each source is visible rather than inferred.

USAGE
------------------------------------------------------------------------------
    python collect_hard_regime.py \\
        --out estimator_data/v3b_hard.npz \\
        --v3a-block1 results_aruco_v3a/run_20260912_201312/best_success_model.zip \\
        --v3a-block3 results_aruco_v3a_cont/cont_20260913_150310/best_success_model.zip \\
        --episodes-per-cell 30

Then retrain on the merged dataset:

    python train_estimator.py \\
        --data estimator_data/v3b_dataset.npz,estimator_data/v3b_hard.npz \\
        --out ../models/temporal_estimator_v3b_r2.pt
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.ArucoLanderAviary import ArucoLanderAviary        # noqa: E402
import train_aruco_v3a as v3a                               # noqa: E402
from collect_estimator_data import ScriptedPolicy           # noqa: E402

DT = 1.0 / 30.0

# Levels where the platform actually moves fast. Level 10 is included because
# 0.15 m/s already sits in the 'med' motion bin where the estimator lost.
HARD_LEVELS = [10, 11, 12]


def rollout(env, policy, seed):
    obs, info = env.reset(seed=seed)
    rec = defaultdict(list)
    prev_action = np.zeros(3)

    while True:
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        obs, r, term, trunc, info = env.step(action)

        cache = env._det_cache or {}
        detected = bool(cache.get('detected', False))
        dm = cache.get('d_meas')
        dm = (np.zeros(3) if (dm is None or not detected)
              else np.asarray(dm, dtype=np.float64))

        state = env._getDroneStateVector(0)
        gt_d = np.asarray(env._getLandingTargetPos(), float) \
            - np.asarray(state[0:3], float)
        gt_dv = np.asarray(env._getPlatformVel(), float) \
            - np.asarray(state[10:13], float)

        rec['d_meas'].append(dm)
        rec['valid'].append(1.0 if detected else 0.0)
        rec['v'].append(np.asarray(state[10:13], float))
        rec['rpy'].append(np.asarray(state[7:10], float))
        rec['omega'].append(np.asarray(state[13:16], float))
        rec['prev_action'].append(prev_action.copy())
        rec['gt_d'].append(gt_d)
        rec['gt_dv'].append(gt_dv)
        prev_action = action.copy()

        if term or trunc:
            break

    return rec, env.episodeEstimatorSummary(), info


def hard_stretch_stats(rec, min_blind, min_motion):
    """Longest blind stretch and its mean |delta_v|. This is the measured
    criterion -- an episode is hard only if it demonstrably contains long
    dropout WHILE the relative state is moving."""
    valid = np.asarray(rec['valid'])
    dvn = np.linalg.norm(np.asarray(rec['gt_dv']), axis=1)

    best_len, best_motion = 0, 0.0
    run, run_dv = 0, []
    for t in range(len(valid)):
        if valid[t] == 0:
            run += 1
            run_dv.append(dvn[t])
        else:
            if run >= min_blind and run_dv:
                m = float(np.mean(run_dv))
                if m > best_motion or run > best_len:
                    best_len, best_motion = max(best_len, run), max(
                        best_motion, m)
            run, run_dv = 0, []
    if run >= min_blind and run_dv:
        m = float(np.mean(run_dv))
        best_len, best_motion = max(best_len, run), max(best_motion, m)

    is_hard = best_len >= min_blind and best_motion >= min_motion
    return is_hard, best_len, best_motion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='estimator_data/v3b_hard.npz')
    ap.add_argument('--episodes-per-cell', type=int, default=30)
    ap.add_argument('--v3a-block1', default=None)
    ap.add_argument('--v3a-block3', default=None)
    ap.add_argument('--min-blind', type=int, default=60,
                    help='blind-stretch length that counts as long dropout')
    ap.add_argument('--min-motion', type=float, default=0.12,
                    help='mean |delta_v| (m/s) during that stretch that '
                         'counts as genuine relative motion')
    ap.add_argument('--seed-base', type=int, default=61000)
    args = ap.parse_args()

    from stable_baselines3 import TD3

    sources = []
    for label, p in (('v3a_block1', args.v3a_block1),
                     ('v3a_block3', args.v3a_block3)):
        if p and os.path.exists(p):
            sources.append((label, TD3.load(p)))
        else:
            print(f'  [skip] {label} not provided/found')
    # A deliberately sloppy scripted controller: slow gains plus jitter make
    # it approach lazily, which generates long dropout at high platform speed
    # without needing a trained policy to fail in a particular way.
    sources.append(('scripted_slow', ScriptedPolicy(
        k_xy=2.5, k_z=1.0, jitter=0.15, rng=np.random.default_rng(77))))

    if not sources:
        print('No sources available. Provide at least one V3a checkpoint.')
        sys.exit(1)

    print('=' * 74)
    print('  V3b TARGETED COLLECTION -- long dropout AT high platform speed')
    print('=' * 74)
    print(f'  levels          : {HARD_LEVELS}')
    print(f'  sources         : {[s[0] for s in sources]}')
    print(f'  hard criterion  : blind stretch >= {args.min_blind} steps '
          f'with mean |delta_v| >= {args.min_motion} m/s')
    print(f'  episodes / cell : {args.episodes_per_cell}')
    print()

    eps, meta = [], []
    kept, rejected = Counter(), Counter()
    seed = args.seed_base
    t0 = time.time()

    for level in HARD_LEVELS:
        radius_max, speed = v3a.CURRICULUM_LEVELS[level - 1]
        kw = v3a.make_env_kwargs(v3a.RADIUS_MIN, radius_max, speed)
        kw['estimator_mode'] = 'predict'
        kw.pop('strict_no_privileged', None)

        env = ArucoLanderAviary(**kw)
        try:
            for name, pol in sources:
                for _ in range(args.episodes_per_cell):
                    rec, summ, info = rollout(env, pol, seed)
                    seed += 1
                    T = len(rec['valid'])
                    if T < 5:
                        continue

                    is_hard, blen, bmotion = hard_stretch_stats(
                        rec, args.min_blind, args.min_motion)
                    key = f'L{level}/{name}'
                    if not is_hard:
                        rejected[key] += 1
                        continue
                    kept[key] += 1

                    eps.append({k: np.asarray(v, dtype=np.float32)
                                for k, v in rec.items()})
                    meta.append({
                        'length': T, 'level': level,
                        'platform_speed': float(speed),
                        'source': name, 'seed': seed - 1,
                        'success': bool(info.get('success', False)),
                        'blind_fraction': float(summ['blind_fraction']),
                        'longest_blind_run': int(summ['longest_blind_run']),
                        'hard_stretch_len': int(blen),
                        'hard_stretch_motion': float(bmotion),
                    })
                print(f'  L{level:2d} {name:16s} kept {kept[key]:3d} / '
                      f'rejected {rejected[key]:3d}')
        finally:
            env.close()

    if not meta:
        print('\nNo episodes met the hard criterion. Loosen --min-blind or '
              '--min-motion, or check the checkpoints.')
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    lengths = np.array([m['length'] for m in meta], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    packed = {k: np.concatenate([e[k] for e in eps], axis=0)
              for k in ('d_meas', 'valid', 'v', 'rpy', 'omega',
                        'prev_action', 'gt_d', 'gt_dv')}

    np.savez_compressed(
        args.out, ep_start=starts, ep_length=lengths,
        ep_level=np.array([m['level'] for m in meta], dtype=np.int64),
        ep_speed=np.array([m['platform_speed'] for m in meta],
                          dtype=np.float32),
        ep_source=np.array([m['source'] for m in meta]),
        ep_seed=np.array([m['seed'] for m in meta], dtype=np.int64),
        ep_success=np.array([m['success'] for m in meta]),
        ep_blind_fraction=np.array([m['blind_fraction'] for m in meta],
                                   dtype=np.float32),
        ep_longest_blind=np.array([m['longest_blind_run'] for m in meta],
                                  dtype=np.int64),
        **packed)

    hl = np.array([m['hard_stretch_len'] for m in meta])
    hm = np.array([m['hard_stretch_motion'] for m in meta])

    with open(args.out.replace('.npz', '_manifest.json'), 'w') as f:
        json.dump({
            'episodes': len(meta), 'total_steps': int(lengths.sum()),
            'kept': dict(kept), 'rejected': dict(rejected),
            'min_blind': args.min_blind, 'min_motion': args.min_motion,
            'hard_stretch_len_mean': float(hl.mean()),
            'hard_stretch_motion_mean': float(hm.mean()),
            'note': 'Targeted supplement for the long-dropout-at-speed gap '
                    'found by the 2x2 cross-check. MERGE with the base '
                    'dataset; do not train on this alone.',
        }, f, indent=2)

    print()
    print('=' * 74)
    print(f'  kept      : {len(meta)} episodes, {int(lengths.sum())} steps')
    print(f'  rejected  : {sum(rejected.values())} (criterion not met)')
    print(f'  wall clock: {(time.time() - t0) / 60:.1f} min')
    print()
    print(f'  hard stretch length : mean {hl.mean():.0f}  max {hl.max()}')
    print(f'  its mean |delta_v|  : mean {hm.mean():.3f}  max {hm.max():.3f}')
    print()
    print(f'  wrote {args.out}')
    print()
    print('  MERGE with the base dataset when retraining -- training on the')
    print('  hard regime alone would trade one distribution gap for another.')
    print('=' * 74)


if __name__ == '__main__':
    main()
