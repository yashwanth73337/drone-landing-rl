"""
diag_v4_reward_profile.py
=========================

Maps the per-step reward against height, to answer whether the reward has a
usable gradient in the region the drone actually starts in.

WHY THIS IS THE RIGHT QUESTION NOW
------------------------------------------------------------------------------
The reachability probe established:

    oracle (ground-truth proportional)   lands 5/5 in 66 steps
    held random (correlated, 20 steps)   reaches 0.264 m, never lands
    white random (fresh every step)      reaches 0.399 m, never lands
    V4 trained, deterministic            reaches 0.431 m, never lands

So the task is reachable, and the trained policy is the WORST of the four at
descending while also being the most consistent (SD 0.014 m against white's
0.065 m). It is not failing to descend; it is deliberately holding 0.43 m.

A policy converges on that only if holding altitude is what the reward
actually pays for. V3a's report already records, as one of four documented
deviations from the published Lander.AI reward, that the far-field branch
carries no gradient. The drone starts at 0.5 m. If the reward is flat up
there, then:

    - TD3 still learned, because one lucky landing enters the replay buffer
      and its value propagates backward over thousands of replays
    - PPO cannot, because it is on-policy: a rare success is seen once, in
      one batch, and discarded

That would explain V4 exactly, and it would mean the fix is the reward or the
starting distribution, NOT the exploration scheme.

WHAT TO READ IN THE OUTPUT
------------------------------------------------------------------------------
The oracle descends smoothly from ~0.5 m to touchdown, so its reward trace is
a sweep of the reward function over height. Compare against HOVER, which
holds position and collects only the per-step cost.

    oracle reward >> hover reward at 0.4-0.5 m
        -> there IS a descent gradient in the far field. The reward is not
           the problem; the problem is that PPO never followed it. Look at
           exploration and at the value function.

    oracle reward ~= hover reward until the drone is already close
        -> CONFIRMED: no far-field gradient. Nothing pulls the policy down
           from 0.5 m, and hovering is genuinely optimal until a success is
           sampled. Off-policy replay is what carried V3a/V3b past this.
           Fixing exploration alone will not help.

USAGE
    python diag_v4_reward_profile.py
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.ShinLanderAviary import ShinLanderAviary          # noqa: E402
from train_shin_v4 import make_kwargs                       # noqa: E402


def run(env, seed, mode, gain=10.0, max_steps=300):
    obs, info = env.reset(seed=seed)
    lo, hi = env.action_space.low, env.action_space.high
    rows = []
    for t in range(max_steps):
        if mode == 'oracle':
            d = np.asarray(obs['target'][:3], dtype=np.float64)
            a = np.clip(d * gain, lo, hi)
        elif mode == 'hover':
            a = np.zeros_like(lo)
        else:
            raise ValueError(mode)

        obs, reward, terminated, truncated, info = env.step(a)
        rows.append({
            't': t,
            'h': float(info['height_above_pad']),
            'd': float(info['d_target']),
            'r': float(reward),
        })
        if terminated or truncated:
            break
    return rows, bool(info.get('success', False))


def band_means(rows, edges):
    """Mean per-step reward within each height band."""
    out = []
    for lo_h, hi_h in zip(edges[:-1], edges[1:]):
        sel = [r['r'] for r in rows if lo_h <= r['h'] < hi_h]
        out.append((lo_h, hi_h, len(sel),
                    float(np.mean(sel)) if sel else float('nan')))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=9000)
    ap.add_argument('--radius-min', type=float, default=0.11)
    ap.add_argument('--radius-max', type=float, default=0.14)
    ap.add_argument('--speed', type=float, default=0.0)
    ap.add_argument('--frame', default='world')
    ap.add_argument('--gain', type=float, default=10.0)
    args = ap.parse_args()

    kw = make_kwargs(args.radius_min, args.radius_max, args.speed, args.frame)

    print('=' * 78)
    print('  REWARD PROFILE vs HEIGHT')
    print('=' * 78)

    env = ShinLanderAviary(**kw)
    try:
        oracle, ok_o = run(env, args.seed, 'oracle', gain=args.gain)
        hover, ok_h = run(env, args.seed, 'hover')
    finally:
        env.close()

    print(f'\n  oracle : {len(oracle)} steps, success={ok_o}, '
          f'total reward {sum(r["r"] for r in oracle):.3f}')
    print(f'  hover  : {len(hover)} steps, success={ok_h}, '
          f'total reward {sum(r["r"] for r in hover):.3f}')

    print('\n  ORACLE TRACE (every 5th step)')
    print(f'  {"t":>5}{"height":>9}{"d_target":>10}{"reward":>10}'
          f'{"cum":>10}')
    cum = 0.0
    for r in oracle:
        cum += r['r']
        if r['t'] % 5 == 0 or r is oracle[-1]:
            print(f'  {r["t"]:>5}{r["h"]:>9.4f}{r["d"]:>10.4f}'
                  f'{r["r"]:>10.4f}{cum:>10.3f}')

    print('\n  HOVER TRACE (every 25th step)')
    print(f'  {"t":>5}{"height":>9}{"d_target":>10}{"reward":>10}'
          f'{"cum":>10}')
    cum = 0.0
    for r in hover:
        cum += r['r']
        if r['t'] % 25 == 0 or r is hover[-1]:
            print(f'  {r["t"]:>5}{r["h"]:>9.4f}{r["d"]:>10.4f}'
                  f'{r["r"]:>10.4f}{cum:>10.3f}')

    edges = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.60]
    print('\n  MEAN PER-STEP REWARD BY HEIGHT BAND')
    print(f'  {"band (m)":>16}{"oracle n":>10}{"oracle r":>11}'
          f'{"hover n":>10}{"hover r":>11}{"delta":>10}')
    ob = band_means(oracle, edges)
    hb = band_means(hover, edges)
    for (lo_h, hi_h, no, ro), (_, _, nh, rh) in zip(ob, hb):
        delta = (ro - rh) if (np.isfinite(ro) and np.isfinite(rh)) \
            else float('nan')
        print(f'  {f"{lo_h:.2f}-{hi_h:.2f}":>16}{no:>10}'
              f'{ro:>11.4f}{nh:>10}{rh:>11.4f}{delta:>10.4f}')

    # The band the drone actually starts in and the one the trained policy
    # parked in are the ones that decide this.
    far = [b for b in ob if b[0] >= 0.40 and b[2] > 0]
    far_h = [b for b in hb if b[0] >= 0.40 and b[2] > 0]
    print()
    print('=' * 78)
    if far and far_h:
        ro = float(np.nanmean([b[3] for b in far]))
        rh = float(np.nanmean([b[3] for b in far_h]))
        print(f'  Above 0.40 m:  oracle {ro:+.4f} / step   '
              f'hover {rh:+.4f} / step   delta {ro - rh:+.4f}')
        print()
        if abs(ro - rh) < 0.005:
            print('  NO FAR-FIELD GRADIENT.')
            print('  Descending pays essentially the same as hovering at the')
            print('  altitude the drone starts from, so the policy gradient')
            print('  has nothing to follow until a landing is sampled.')
            print('  On-policy PPO cannot retain a rare success; TD3 replayed')
            print('  one thousands of times. Fixing exploration will not help.')
            print('  Options: shape the far field, lower the spawn altitude,')
            print('  or warm-start from the oracle.')
        else:
            print('  THERE IS A FAR-FIELD GRADIENT.')
            print('  Descending pays more than hovering up there, so the')
            print('  reward is not the blocker. PPO had a signal and did not')
            print('  follow it -- look at exploration and at whether the')
            print('  value function is drowning the advantage.')
    else:
        print('  Not enough samples above 0.40 m to judge; rerun with a')
        print('  higher spawn or a slower oracle gain.')
    print('=' * 78)


if __name__ == '__main__':
    main()
