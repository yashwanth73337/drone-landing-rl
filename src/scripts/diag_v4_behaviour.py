"""
diag_v4_behaviour.py
====================

Answers one question about a V4 run that never landed:

    Did training select DESCENT out of the policy, or did the policy never
    descend in the first place?

The two have opposite fixes, and the success_evaluations.csv cannot separate
them -- near_miss is 0 from the very first evaluation, so there is no "before"
to compare against.

The comparison that does separate them is TRAINED vs RANDOM on identical
episodes and seeds:

    random descends more than trained  ->  the reward is selecting against
                                           descent. The active-perception term
                                           (or the shaping) is the suspect.

    neither descends                   ->  exploration never reached the
                                           terminal reward. On-policy PPO has
                                           no mechanism to retain a rare
                                           success; this is not a perception
                                           problem at all.

Reports, per policy: minimum height reached, final lateral distance, mean and
maximum commanded action magnitude, and how far down the episode actually got.

No training, no writes to the run directory. ~2 minutes.

USAGE
    python diag_v4_behaviour.py \\
        --model results_shin_v4/run_20260919_153640/final_model.zip
"""

import argparse
import os
import sys

import numpy as np
import torch as th

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.ShinLanderAviary import ShinLanderAviary          # noqa: E402
import train_aruco_v3a as v3a                               # noqa: E402
from train_shin_v4 import ShinRecurrentPPO, make_kwargs     # noqa: E402


def run_episode(env, seed, policy=None, rng=None):
    """One episode. policy=None means uniform random actions."""
    obs, info = env.reset(seed=seed)

    if policy is not None:
        device = policy.device
        n_l = policy.lstm_actor.num_layers
        n_h = policy.lstm_actor.hidden_size
        states = (th.zeros(n_l, 1, n_h, device=device),
                  th.zeros(n_l, 1, n_h, device=device))
        ep_start = th.ones(1, dtype=th.float32, device=device)

    heights, lat, acts = [], [], []
    while True:
        if policy is None:
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape)
        else:
            with th.no_grad():
                obs_t, _ = policy.obs_to_tensor(obs)
                feats = policy.extract_features(obs_t)
                pi_feat, _ = policy._split_features(feats)
                lstm_out, states = policy._process_sequence(
                    pi_feat, states, ep_start, policy.lstm_actor)
                latent_pi = policy.mlp_extractor.forward_actor(lstm_out)
                dist = policy._get_action_dist_from_latent(latent_pi)
                action = dist.get_actions(
                    deterministic=True)[0].cpu().numpy()
            ep_start = th.zeros(1, dtype=th.float32, device=policy.device)

        acts.append(float(np.abs(action).max()))   # pre-clip magnitude
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        heights.append(float(info['height_above_pad']))
        lat.append(float(info['d_target']))

        if terminated or truncated:
            break

    return {
        'steps': len(heights),
        'h_start': heights[0],
        'h_min': float(np.min(heights)),
        'h_end': heights[-1],
        'lat_min': float(np.min(lat)),
        'act_mean': float(np.mean(acts)),
        'act_max': float(np.max(acts)),
        'success': bool(info.get('success', False)),
        'reason': info.get('truncation_reason') or 'none',
    }


def summarise(tag, rows):
    print(f'\n  {tag}')
    print(f'  {"":<6}{"steps":>7}{"h_start":>9}{"h_min":>8}{"h_end":>8}'
          f'{"lat_min":>9}{"|a|mean":>9}{"|a|max":>8}  reason')
    for i, r in enumerate(rows):
        print(f'  ep{i:<4}{r["steps"]:>7}{r["h_start"]:>9.3f}'
              f'{r["h_min"]:>8.3f}{r["h_end"]:>8.3f}{r["lat_min"]:>9.3f}'
              f'{r["act_mean"]:>9.3f}{r["act_max"]:>8.3f}  {r["reason"]}')
    print(f'  {"mean":<6}{"":>7}{np.mean([r["h_start"] for r in rows]):>9.3f}'
          f'{np.mean([r["h_min"] for r in rows]):>8.3f}'
          f'{np.mean([r["h_end"] for r in rows]):>8.3f}'
          f'{np.mean([r["lat_min"] for r in rows]):>9.3f}'
          f'{np.mean([r["act_mean"] for r in rows]):>9.3f}'
          f'{np.mean([r["act_max"] for r in rows]):>8.3f}')
    return float(np.mean([r['h_min'] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--seed-base', type=int, default=9000)
    ap.add_argument('--radius-min', type=float, default=0.11)
    ap.add_argument('--radius-max', type=float, default=0.14)
    ap.add_argument('--speed', type=float, default=0.0)
    ap.add_argument('--frame', default='world')
    args = ap.parse_args()

    kw = make_kwargs(args.radius_min, args.radius_max, args.speed, args.frame)

    print('=' * 78)
    print('  V4 BEHAVIOUR DIAGNOSTIC -- trained vs random, identical seeds')
    print('=' * 78)
    print(f'  model : {args.model}')
    print(f'  task  : radius {args.radius_min}-{args.radius_max} m, '
          f'speed {args.speed} m/s')
    print(f'  NOTE  : touchdown is at 0.0125 m (Crazyflie collision cylinder '
          f'half-height).')
    print(f'          near_miss counts steps below 0.03 m.')

    model = ShinRecurrentPPO.load(args.model)

    env = ShinLanderAviary(**kw)
    try:
        trained = [run_episode(env, args.seed_base + i, policy=model.policy)
                   for i in range(args.episodes)]
    finally:
        env.close()

    rng = np.random.default_rng(0)
    env = ShinLanderAviary(**kw)
    try:
        random_rows = [run_episode(env, args.seed_base + i, rng=rng)
                       for i in range(args.episodes)]
    finally:
        env.close()

    h_trained = summarise('TRAINED (deterministic)', trained)
    h_random = summarise('RANDOM  (uniform in action space)', random_rows)

    print()
    print('=' * 78)
    print(f'  mean minimum height   trained {h_trained:.3f} m   '
          f'random {h_random:.3f} m')
    print()
    if h_trained > h_random + 0.02:
        print('  TRAINED STAYS HIGHER THAN RANDOM.')
        print('  Training actively selected descent OUT of the policy. The')
        print('  reward is the suspect: re-run with --ap-alpha 0 to test the')
        print('  active-perception term, since with a nadir camera descending')
        print('  is exactly what costs visibility.')
    elif h_random > h_trained + 0.02:
        print('  TRAINED DESCENDS FURTHER THAN RANDOM.')
        print('  The policy did learn to approach; it simply never closed the')
        print('  last gap. This is a credit-assignment / budget problem, not a')
        print('  reward-sign problem.')
    else:
        print('  NEITHER DESCENDS.')
        print('  Exploration never reached the terminal reward, so there was')
        print('  no success for the policy to learn from. On-policy PPO sees a')
        print('  rare success once and discards it, where TD3 replays it from')
        print('  the buffer -- which is how V3a/V3b got off level 1.')
        print('  The active-perception term is NOT implicated. Fixes to')
        print('  consider: denser terminal shaping, a longer budget, or')
        print('  seeding with scripted descents.')
    print('=' * 78)


if __name__ == '__main__':
    main()
