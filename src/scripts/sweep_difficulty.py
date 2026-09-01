"""
Difficulty sweep for a trained landing policy.

Loads one trained policy and evaluates it, WITHOUT RETRAINING, across a range
of platform speeds. Produces a success-versus-difficulty curve.

Why this matters
----------------
Every configuration in the curriculum saturated at 100% success. That is the
same problem identified in the literature survey: Goldschmid & Ahmad (2024),
TornadoDrone (2024), and Shin et al. (2026) all report simulation success rates
between 97% and 100%, so the metric cannot discriminate between methods.

A saturated metric is not a result. Finding where the policy BREAKS is.

This sweep also measures generalisation. The policy was trained at exactly one
platform speed; testing it at unseen speeds shows whether it learned to land on
a moving target or merely memorised one trajectory.

Platform motion is x(t) = A · sin(ω · t), so peak speed = A · ω.

Usage:
    python sweep_difficulty.py                    # sweep frequency
    python sweep_difficulty.py --mode amplitude   # sweep amplitude
    python sweep_difficulty.py --episodes 50      # fewer episodes, faster
"""

import os
import sys
import csv
import argparse
import numpy as np

from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.LandingAviary import LandingAviary


def evaluate_config(model, amplitude, omega, n_episodes, seed_offset=0):
    """Run n_episodes at one platform setting. Returns a dict of metrics."""
    env = LandingAviary(platform_amplitude=amplitude, platform_omega=omega)

    successes = 0
    touchdowns = 0
    vz, lat, tilt, offset = [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        if info.get("touchdown"):
            touchdowns += 1
            if info.get("success"):
                successes += 1
                vz.append(info["touchdown_vz"])
                lat.append(info["touchdown_lateral"])
                tilt.append(info["touchdown_tilt"])
                offset.append(info["touchdown_offset"])

    env.close()

    return {
        "amplitude": amplitude,
        "omega": omega,
        "peak_speed": amplitude * omega,
        "success_rate": 100.0 * successes / n_episodes,
        "touchdown_rate": 100.0 * touchdowns / n_episodes,
        "n_success": successes,
        "vz_mean": float(np.mean(vz)) if vz else float('nan'),
        "lat_mean": float(np.mean(lat)) if lat else float('nan'),
        "tilt_deg_mean": float(np.degrees(np.mean(tilt))) if tilt else float('nan'),
        "offset_mean": float(np.mean(offset)) if offset else float('nan'),
        "viol_vz": 100.0 * np.mean(np.array(vz) > 0.5) if vz else float('nan'),
        "viol_lat": 100.0 * np.mean(np.array(lat) > 0.3) if lat else float('nan'),
        "viol_tilt": 100.0 * np.mean(np.degrees(tilt) > 10) if tilt else float('nan'),
    }


def run_sweep(model_path, mode, n_episodes, out_csv):
    model = PPO.load(model_path)

    if mode == "frequency":
        # Hold amplitude at the trained value, raise how fast it oscillates.
        # Peak speed ranges 0.00 -> 1.00 m/s. Trained at 0.25 m/s.
        amplitude = 0.5
        omegas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        configs = [(amplitude, w) for w in omegas]
        varied = "omega"
    else:
        # Hold frequency, widen the sweep. Also tests a larger arena.
        omega = 0.5
        amplitudes = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
        configs = [(a, omega) for a in amplitudes]
        varied = "amplitude"

    print(f"\nSweeping {varied}, {n_episodes} episodes per point.")
    print(f"Policy was trained at amplitude 0.5, omega 0.5 (peak speed 0.25 m/s).\n")

    header = (f"{'A':>5} {'omega':>6} {'peak m/s':>9} {'success':>8} "
              f"{'vz':>7} {'lat':>7} {'tilt°':>7} {'offset':>7} {'tilt>10°':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for amplitude, omega in configs:
        r = evaluate_config(model, amplitude, omega, n_episodes)
        rows.append(r)
        print(f"{r['amplitude']:5.2f} {r['omega']:6.2f} {r['peak_speed']:9.3f} "
              f"{r['success_rate']:7.1f}% "
              f"{r['vz_mean']:7.3f} {r['lat_mean']:7.3f} "
              f"{r['tilt_deg_mean']:7.2f} {r['offset_mean']:7.3f} "
              f"{r['viol_tilt']:8.1f}%")

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {out_csv}")

    # Where does it break?
    failures = [r for r in rows if r['success_rate'] < 100.0]
    if failures:
        first = failures[0]
        print(f"\nFirst degradation at peak speed {first['peak_speed']:.3f} m/s "
              f"({first['success_rate']:.0f}% success).")
    else:
        print("\nNo degradation across the tested range. Widen the sweep.")


def _latest_model():
    runs = sorted(os.listdir('results_landing'))
    if not runs:
        print("No trained models found in results_landing/.")
        sys.exit(1)
    return os.path.join('results_landing', runs[-1], 'best_model.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['frequency', 'amplitude'],
                        default='frequency',
                        help='which parameter to sweep')
    parser.add_argument('--episodes', type=int, default=50,
                        help='episodes per configuration')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    out = args.out or f'sweep_{args.mode}.csv'
    run_sweep(args.model or _latest_model(), args.mode, args.episodes, out)
