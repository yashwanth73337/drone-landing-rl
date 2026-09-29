"""
diag_v4_reachability.py
=======================

V4 seed 1 never descended in 600k steps. The tensorboard series ruled out the
active-perception reward (r_active was exactly 0 from 150k onward, because
L_est fell below tau=0.01 by 50k), so the remaining hypothesis is that PPO's
exploration cannot produce a sustained descent.

That hypothesis is testable in three minutes without training anything, by
running four open-loop policies on identical episodes and seeds:

    trained      the V4 policy, deterministic, actions clipped
    white        uniform random, resampled EVERY step
    held         uniform random, resampled every --hold steps
    oracle       proportional control on the TRUE relative position

What each outcome means:

    oracle lands, held lands, white does not
        -> the task is reachable and the action space is fine; white noise
           averages to zero net displacement over an episode, so PPO's
           exploration structurally cannot sample a landing. use_sde=True
           (temporally correlated, state-dependent noise) is the indicated
           fix, not a reward change.

    oracle lands, held does not
        -> descent needs closed-loop correction, not just a sustained
           command. Exploration alone will not find it; the fix is on the
           learning side (denser shaping, or warm-starting from the oracle).

    oracle does NOT land
        -> the problem is not exploration at all. Either the action mapping
           is not what this script assumes, or the episode budget cannot
           cover the distance. Stop and check the env before changing
           anything about the algorithm.

The oracle is the load-bearing control here. It is the difference between
"the agent cannot find the solution" and "there is no solution to find".

USAGE
    python diag_v4_reachability.py \\
        --model results_shin_v4/run_20260919_153640/final_model.zip
"""

import argparse
import os
import sys

import numpy as np
import torch as th

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.ShinLanderAviary import ShinLanderAviary          # noqa: E402
from train_shin_v4 import ShinRecurrentPPO, make_kwargs     # noqa: E402


def run_episode(env, seed, mode, policy=None, rng=None, hold=20, gain=10.0):
    obs, info = env.reset(seed=seed)
    lo, hi = env.action_space.low, env.action_space.high

    if mode == 'trained':
        device = policy.device
        n_l, n_h = policy.lstm_actor.num_layers, policy.lstm_actor.hidden_size
        states = (th.zeros(n_l, 1, n_h, device=device),
                  th.zeros(n_l, 1, n_h, device=device))
        ep_start = th.ones(1, dtype=th.float32, device=device)

    held_action = None
    heights, dists, acts = [], [], []
    t = 0

    while True:
        if mode == 'trained':
            with th.no_grad():
                obs_t, _ = policy.obs_to_tensor(obs)
                feats = policy.extract_features(obs_t)
                pi_feat, _ = policy._split_features(feats)
                lstm_out, states = policy._process_sequence(
                    pi_feat, states, ep_start, policy.lstm_actor)
                latent_pi = policy.mlp_extractor.forward_actor(lstm_out)
                a = policy._get_action_dist_from_latent(latent_pi).get_actions(
                    deterministic=True)[0].cpu().numpy()
            ep_start = th.zeros(1, dtype=th.float32, device=policy.device)

        elif mode == 'white':
            a = rng.uniform(lo, hi)

        elif mode == 'held':
            if t % hold == 0:
                held_action = rng.uniform(lo, hi)
            a = held_action

        elif mode == 'oracle':
            # obs['target'] is [d_true, dv_true] in physical units, d_true =
            # pad_top_centre - drone_pos. Drive the commanded offset straight
            # at it. Ground truth is legitimate here: this is a reachability
            # probe, not a policy.
            d = np.asarray(obs['target'][:3], dtype=np.float64)
            a = np.clip(d * gain, lo, hi)

        else:
            raise ValueError(mode)

        a = np.clip(np.asarray(a, dtype=np.float64), lo, hi)
        acts.append(float(np.abs(a).max()))

        obs, reward, terminated, truncated, info = env.step(a)
        heights.append(float(info['height_above_pad']))
        dists.append(float(info['d_target']))
        t += 1

        if terminated or truncated:
            break

    return {
        'steps': len(heights),
        'h_min': float(np.min(heights)),
        'd_min': float(np.min(dists)),
        'act_mean': float(np.mean(acts)),
        'success': bool(info.get('success', False)),
        'reason': info.get('truncation_reason') or 'none',
    }


def summarise(tag, rows):
    n_succ = sum(r['success'] for r in rows)
    print(f'\n  {tag}')
    print(f'  {"":<6}{"steps":>7}{"h_min":>9}{"d_min":>9}{"|a|mean":>9}'
          f'  {"ok":>3}  reason')
    for i, r in enumerate(rows):
        print(f'  ep{i:<4}{r["steps"]:>7}{r["h_min"]:>9.4f}{r["d_min"]:>9.4f}'
              f'{r["act_mean"]:>9.3f}  {"YES" if r["success"] else " - ":>3}'
              f'  {r["reason"]}')
    print(f'  {"mean":<6}{np.mean([r["steps"] for r in rows]):>7.0f}'
          f'{np.mean([r["h_min"] for r in rows]):>9.4f}'
          f'{np.mean([r["d_min"] for r in rows]):>9.4f}'
          f'{np.mean([r["act_mean"] for r in rows]):>9.3f}'
          f'  {n_succ}/{len(rows)}')
    return float(np.mean([r['h_min'] for r in rows])), n_succ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--seed-base', type=int, default=9000)
    ap.add_argument('--radius-min', type=float, default=0.11)
    ap.add_argument('--radius-max', type=float, default=0.14)
    ap.add_argument('--speed', type=float, default=0.0)
    ap.add_argument('--frame', default='world')
    ap.add_argument('--hold', type=int, default=20,
                    help='steps to hold each sampled action in "held" mode')
    ap.add_argument('--gain', type=float, default=10.0,
                    help='oracle proportional gain; 10 saturates at 0.1 m '
                         'error, matching dp = 0.1 * c_t')
    args = ap.parse_args()

    kw = make_kwargs(args.radius_min, args.radius_max, args.speed, args.frame)

    print('=' * 78)
    print('  V4 REACHABILITY PROBE')
    print('=' * 78)
    print(f'  task      : radius {args.radius_min}-{args.radius_max} m, '
          f'speed {args.speed} m/s')
    print(f'  touchdown : 0.0125 m   near_miss threshold: 0.03 m')
    print(f'  hold      : {args.hold} steps   oracle gain: {args.gain}')

    model = ShinRecurrentPPO.load(args.model)

    results = {}
    for mode in ('trained', 'white', 'held', 'oracle'):
        rng = np.random.default_rng(0)
        env = ShinLanderAviary(**kw)
        try:
            rows = [run_episode(env, args.seed_base + i, mode,
                                policy=model.policy, rng=rng,
                                hold=args.hold, gain=args.gain)
                    for i in range(args.episodes)]
        finally:
            env.close()
        results[mode] = rows

    labels = {
        'trained': 'TRAINED  (V4 deterministic, clipped)',
        'white':   'WHITE    (uniform, resampled every step)',
        'held':    f'HELD     (uniform, resampled every {args.hold} steps)',
        'oracle':  'ORACLE   (proportional on true relative position)',
    }
    stats = {m: summarise(labels[m], results[m]) for m in results}

    print()
    print('=' * 78)
    for m in ('trained', 'white', 'held', 'oracle'):
        h, n = stats[m]
        print(f'  {m:<9} mean h_min {h:7.4f} m   landings {n}/{args.episodes}')
    print()

    o_h, o_n = stats['oracle']
    w_h, w_n = stats['white']
    d_h, d_n = stats['held']

    if o_n == 0:
        print('  ORACLE DID NOT LAND.')
        print('  This is not an exploration problem. Either the action')
        print('  mapping is not dp = 0.1 * c_t toward obs[target][:3], or the')
        print('  300-step budget cannot cover the distance. Check the env')
        print('  before changing anything about the algorithm.')
        print(f'  (oracle reached {o_h:.4f} m; touchdown is 0.0125 m)')
    elif d_n > 0 and w_n == 0:
        print('  ORACLE AND HELD LAND; WHITE DOES NOT.')
        print('  Confirmed: a sustained command reaches the pad, and white')
        print('  noise averages to nothing over an episode. PPO cannot')
        print('  sample a landing, so the +20 is never experienced and')
        print('  hovering is correctly optimal in the landscape it sees.')
        print('  Indicated fix: use_sde=True (temporally correlated,')
        print('  state-dependent exploration). NOT a reward change.')
    elif d_n == 0 and o_n > 0:
        print('  ORACLE LANDS; HELD DOES NOT.')
        print('  Descent needs closed-loop correction, not just a sustained')
        print('  command, so correlated exploration alone will not find it.')
        print('  The fix is on the learning side: denser terminal shaping, or')
        print('  warm-starting from the oracle.')
    else:
        print('  MIXED RESULT -- read the table above rather than this line.')
        print(f'  white {w_n}/{args.episodes}, held {d_n}/{args.episodes}, '
              f'oracle {o_n}/{args.episodes}')
    print('=' * 78)


if __name__ == '__main__':
    main()
