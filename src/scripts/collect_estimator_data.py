"""
collect_estimator_data.py
=========================

Collects COMPLETE EPISODES for offline supervised training of the V3b
temporal estimator.

Episode boundaries are preserved throughout. The train/val/test split happens
BY EPISODE, never by timestep or overlapping window -- a sequence model split
at the timestep level leaks the hidden-state context across splits and
produces validation numbers that mean nothing.

Ground truth is recorded as a supervised TARGET only. It never enters any
actor observation: the data-collection policies here either fly on privileged
state (estimator_mode='off', which is what the trusted moving_025 checkpoints
expect) or are scripted. The estimator being trained consumes only the 13-D
input listed in temporal_estimator.py.


DATASET COMPOSITION AND WHY IT IS STRATIFIED
------------------------------------------------------------------------------
V3a never flew levels 10-12, and its replay buffers are useless here anyway:
they store the 15-D observation (the PROPAGATED d_est, not the raw d_meas),
with no validity signal and no ground truth. So data must be collected fresh.

The awkward part, stated plainly: we need trajectories at levels 8-12, and no
vision-trained policy can fly them. The trusted privileged moving_025
checkpoints can. They are therefore used as data-collection policies, flying
on privileged observations while the camera pipeline runs alongside and is
logged.

    This is supervised data collection, not policy learning. But it does mean
    the estimator learns on PRIVILEGED-policy trajectories and will be
    deployed under a VISION policy that flies differently. That is a
    behavioural-cloning-style distribution shift, and it is the reason the
    closed-loop check (step 10 of the V3b plan) exists before any RL run.

Mitigation is diversity. The mixture deliberately includes the V3a
checkpoints around the level-9 failure regime, where the deployed policy
actually spends its time, and a scripted controller for coverage. Random
actions are included only in small quantity and are capped, because a dataset
dominated by flailing trajectories would train an estimator for a state
distribution no policy visits.

Per-level and per-source episode and step counts are written to the manifest
and printed, so the composition is visible in the report rather than implied.


USAGE
------------------------------------------------------------------------------
    python collect_estimator_data.py --out estimator_data/v3b_dataset.npz

    # smaller smoke run first
    python collect_estimator_data.py --out /tmp/smoke.npz --episodes-per-cell 2
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


# Curriculum levels to sample, with how many episodes per source.
# Weighted toward 8-12: those are the levels the deployed policy must
# eventually handle and the ones V3a reached least.
LEVEL_WEIGHTS = {
    1: 1, 2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 7: 2,
    8: 3, 9: 4, 10: 3, 11: 3, 12: 4,
}


class ScriptedPolicy:
    """Proportional controller on the privileged observation.

    Provides coverage at levels no trained policy handles well, without the
    pathological state distribution of random actions.
    """
    name = 'scripted'

    def __init__(self, k_xy=8.0, k_z=3.0, jitter=0.0, rng=None):
        self.k_xy, self.k_z = k_xy, k_z
        self.jitter = jitter
        self.rng = rng or np.random.default_rng(0)

    def predict(self, obs, deterministic=True):
        d = np.asarray(obs, dtype=np.float64)[9:12] * 5.0
        a = np.array([self.k_xy * d[0], self.k_xy * d[1], self.k_z * d[2]])
        if self.jitter:
            a = a + self.rng.normal(0, self.jitter, 3)
        return np.clip(a, -1, 1).astype(np.float32), None


class RandomPolicy:
    name = 'random'

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(0)

    def predict(self, obs, deterministic=True):
        return self.rng.uniform(-1, 1, 3).astype(np.float32), None


def load_sb3(path):
    from stable_baselines3 import TD3
    return TD3.load(path)


def build_sources(args):
    """Assemble the stratified policy mixture, skipping anything absent."""
    sources = []

    for seed in (1, 2, 3):
        p = os.path.join(args.privileged_dir,
                         f'moving_025_seed{seed}_final.zip')
        if os.path.exists(p):
            sources.append({
                'name': f'privileged_seed{seed}',
                'policy': load_sb3(p),
                'levels': list(range(1, 13)),
                'weight': 1.0,
            })
        else:
            print(f'  [skip] {p} not found')

    for label, p in (('v3a_block1', args.v3a_block1),
                     ('v3a_block3', args.v3a_block3)):
        if p and os.path.exists(p):
            sources.append({
                'name': label,
                'policy': load_sb3(p),
                # V3a checkpoints are most informative around the regime the
                # deployed policy actually occupies.
                'levels': [3, 4, 5, 8, 9, 10],
                'weight': 1.0,
            })
        else:
            print(f'  [skip] {label} checkpoint not provided/found')

    sources.append({
        'name': 'scripted',
        'policy': ScriptedPolicy(jitter=0.05,
                                 rng=np.random.default_rng(11)),
        'levels': list(range(1, 13)),
        'weight': 1.0,
    })

    # Capped deliberately -- see the module docstring.
    sources.append({
        'name': 'random',
        'policy': RandomPolicy(np.random.default_rng(12)),
        'levels': [1, 4, 8, 12],
        'weight': 0.25,
    })

    return sources


def rollout_episode(env, policy, seed):
    """One complete episode. Returns per-step arrays, boundaries preserved."""
    obs, info = env.reset(seed=seed)

    rec = defaultdict(list)
    prev_action = np.zeros(3, dtype=np.float64)

    while True:
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float64).reshape(-1)

        obs, reward, terminated, truncated, info = env.step(action)

        # Raw per-frame measurement straight from the detection cache. NOT
        # the propagated estimate -- the estimator must learn propagation
        # itself, so it is never handed a pre-propagated value.
        cache = env._det_cache or {}
        detected = bool(cache.get('detected', False))
        d_meas = cache.get('d_meas')
        d_meas = (np.zeros(3) if (d_meas is None or not detected)
                  else np.asarray(d_meas, dtype=np.float64))

        state = env._getDroneStateVector(0)
        gt_d = np.asarray(env._getLandingTargetPos(), dtype=np.float64) \
            - np.asarray(state[0:3], dtype=np.float64)
        gt_dv = np.asarray(env._getPlatformVel(), dtype=np.float64) \
            - np.asarray(state[10:13], dtype=np.float64)

        rec['d_meas'].append(d_meas)
        rec['valid'].append(1.0 if detected else 0.0)
        rec['v'].append(np.asarray(state[10:13], dtype=np.float64))
        rec['rpy'].append(np.asarray(state[7:10], dtype=np.float64))
        rec['omega'].append(np.asarray(state[13:16], dtype=np.float64))
        rec['prev_action'].append(prev_action.copy())
        rec['gt_d'].append(gt_d)
        rec['gt_dv'].append(gt_dv)

        prev_action = action.copy()

        if terminated or truncated:
            break

    summary = env.episodeEstimatorSummary()
    return rec, summary, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='estimator_data/v3b_dataset.npz')
    ap.add_argument('--episodes-per-cell', type=int, default=4,
                    help='episodes per (source, level) before weighting')
    ap.add_argument('--privileged-dir', default='results_lander')
    ap.add_argument('--v3a-block1', default=None,
                    help='path to a V3a block-1 best_success_model.zip')
    ap.add_argument('--v3a-block3', default=None,
                    help='path to a V3a block-3 best_success_model.zip')
    ap.add_argument('--seed-base', type=int, default=31000)
    args = ap.parse_args()

    print('=' * 74)
    print('  V3b ESTIMATOR DATA COLLECTION')
    print('=' * 74)
    sources = build_sources(args)
    print(f'  sources: {[s["name"] for s in sources]}')
    print()

    ep_arrays = []
    meta = []
    per_level = Counter()
    per_source = Counter()
    steps_level = Counter()
    steps_source = Counter()
    ep_id = 0
    seed_counter = args.seed_base
    t0 = time.time()

    for level in sorted(LEVEL_WEIGHTS):
        radius_max, speed = v3a.CURRICULUM_LEVELS[level - 1]
        env_kwargs = v3a.make_env_kwargs(v3a.RADIUS_MIN, radius_max, speed)
        # Estimator runs and logs while the policy flies on privileged state.
        env_kwargs['estimator_mode'] = 'off'
        env_kwargs.pop('strict_no_privileged', None)

        env = ArucoLanderAviary(**env_kwargs)
        try:
            for src in sources:
                if level not in src['levels']:
                    continue
                n = max(1, int(round(args.episodes_per_cell
                                     * LEVEL_WEIGHTS[level]
                                     * src['weight'])))
                for _ in range(n):
                    rec, summ, info = rollout_episode(
                        env, src['policy'], seed_counter)
                    seed_counter += 1

                    T = len(rec['valid'])
                    if T < 5:
                        continue

                    ep_arrays.append({
                        k: np.asarray(v, dtype=np.float32)
                        for k, v in rec.items()})
                    meta.append({
                        'ep_id': ep_id,
                        'length': T,
                        'level': level,
                        'platform_speed': float(speed),
                        'radius_max': float(radius_max),
                        'source': src['name'],
                        'seed': seed_counter - 1,
                        'success': bool(info.get('success', False)),
                        'blind_fraction': float(summ['blind_fraction']),
                        'longest_blind_run': int(summ['longest_blind_run']),
                        'detections': int(summ['detections']),
                    })
                    per_level[level] += 1
                    per_source[src['name']] += 1
                    steps_level[level] += T
                    steps_source[src['name']] += T
                    ep_id += 1
                print(f'  level {level:2d}  {src["name"]:20s}  '
                      f'{n} episodes')
        finally:
            env.close()

    # ---- pack, preserving episode boundaries ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    lengths = np.array([m['length'] for m in meta], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)

    packed = {}
    for key in ('d_meas', 'valid', 'v', 'rpy', 'omega', 'prev_action',
                'gt_d', 'gt_dv'):
        packed[key] = np.concatenate([e[key] for e in ep_arrays], axis=0)

    np.savez_compressed(
        args.out,
        ep_start=starts, ep_length=lengths,
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

    manifest = {
        'episodes': len(meta),
        'total_steps': int(lengths.sum()),
        'episodes_per_level': dict(sorted(per_level.items())),
        'steps_per_level': dict(sorted(steps_level.items())),
        'episodes_per_source': dict(per_source),
        'steps_per_source': dict(steps_source),
        'dt': 1.0 / 30.0,
        'note': 'Estimator trains on privileged/scripted trajectories but '
                'deploys under a vision policy. Distribution shift is '
                'expected; see the closed-loop check before any RL run.',
    }
    with open(args.out.replace('.npz', '_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print()
    print('=' * 74)
    print(f'  episodes    : {len(meta)}')
    print(f'  total steps : {int(lengths.sum())}')
    print(f'  wall clock  : {(time.time() - t0) / 60:.1f} min')
    print()
    print('  episodes / steps per level:')
    for lv in sorted(per_level):
        print(f'    level {lv:2d}: {per_level[lv]:4d} eps, '
              f'{steps_level[lv]:7d} steps')
    print('  episodes / steps per source:')
    for src in per_source:
        print(f'    {src:22s}: {per_source[src]:4d} eps, '
              f'{steps_source[src]:7d} steps')
    print()
    print(f'  wrote {args.out}')
    print(f'  wrote {args.out.replace(".npz", "_manifest.json")}')
    print('=' * 74)


if __name__ == '__main__':
    main()
