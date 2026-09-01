"""
Acceleration-limit test.

Two earlier sweeps produced contradictory results at the same peak platform
speed:

    peak speed    frequency sweep (A=0.5, w varies)    amplitude sweep (w=0.5, A varies)
    0.250 m/s     100%                                  100%
    0.375 m/s      26%                                  100%
    0.500 m/s       0%                                   92%
    0.750 m/s       0%                                     0%

Same speed, opposite outcomes. So peak speed is NOT what breaks the policy.

HYPOTHESIS
----------
The limiting quantity is peak ACCELERATION, not peak velocity.

Platform motion is  x(t) = A sin(wt), so:

    peak speed         = A * w
    peak acceleration  = A * w^2

Raising w increases acceleration QUADRATICALLY; raising A increases it only
linearly. That asymmetry predicts exactly the pattern above.

Physically this is plausible. The Crazyflie has thrust-to-weight 2.25 and the
environment truncates beyond 0.4 rad (~23 deg) of tilt, which caps achievable
horizontal acceleration at roughly

    a_max = g * tan(0.4) = 9.81 * 0.423 = 4.15 m/s^2

before accounting for the thrust needed to hold altitude. When the platform
reverses direction, the drone must decelerate and re-accelerate the other way.
Above some reversal rate it cannot turn around fast enough and overshoots past
the pad edge — consistent with the observed failure mode, where `offset` blows
up while `vz` and `tilt` stay healthy.

TEST DESIGN
-----------
Two families of configurations:

  A. CONSTANT ACCELERATION, VARYING SPEED
     Hold A*w^2 fixed, vary A*w over a wide range.
     If the hypothesis holds, success should stay roughly FLAT.

  B. CONSTANT SPEED, VARYING ACCELERATION
     Hold A*w fixed, vary A*w^2 over a wide range.
     If the hypothesis holds, success should FALL as acceleration rises.

Together these decouple the two quantities. Speed-limited and
acceleration-limited policies give opposite signatures.

Usage:
    python sweep_acceleration.py
    python sweep_acceleration.py --episodes 30
"""

import os
import sys
import csv
import argparse
import numpy as np

from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.LandingAviary import LandingAviary


# Achievable lateral acceleration at the environment's tilt limit, ignoring
# the extra thrust needed to hold altitude. An upper bound.
TILT_LIMIT_RAD = 0.4
A_MAX_THEORY = 9.81 * np.tan(TILT_LIMIT_RAD)


def evaluate_config(model, amplitude, omega, n_episodes):
    """Run n_episodes at one platform setting."""
    env = LandingAviary(platform_amplitude=amplitude, platform_omega=omega)

    successes = 0
    vz, lat, tilt, offset = [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        if info.get("touchdown") and info.get("success"):
            successes += 1
            vz.append(info["touchdown_vz"])
            lat.append(info["touchdown_lateral"])
            tilt.append(info["touchdown_tilt"])
            offset.append(info["touchdown_offset"])

    env.close()

    return {
        "amplitude": round(amplitude, 4),
        "omega": round(omega, 4),
        "peak_speed": round(amplitude * omega, 4),
        "peak_accel": round(amplitude * omega ** 2, 4),
        "success_rate": 100.0 * successes / n_episodes,
        "vz_mean": float(np.mean(vz)) if vz else float('nan'),
        "lat_mean": float(np.mean(lat)) if lat else float('nan'),
        "tilt_deg_mean": float(np.degrees(np.mean(tilt))) if tilt else float('nan'),
        "offset_mean": float(np.mean(offset)) if offset else float('nan'),
    }


def make_configs_constant_accel(target_accel, speeds):
    """Given peak acceleration a and desired peak speed v:
           v = A*w,  a = A*w^2   =>   w = a/v,  A = v^2/a
    """
    out = []
    for v in speeds:
        omega = target_accel / v
        amplitude = v ** 2 / target_accel
        out.append((amplitude, omega))
    return out


def make_configs_constant_speed(target_speed, accels):
    """Given peak speed v and desired peak acceleration a:
           w = a/v,  A = v^2/a
    """
    out = []
    for a in accels:
        omega = a / target_speed
        amplitude = target_speed ** 2 / a
        out.append((amplitude, omega))
    return out


def print_table(title, rows):
    print(f"\n{title}")
    header = (f"{'A':>6} {'omega':>6} {'speed':>7} {'accel':>7} "
              f"{'success':>8} {'offset':>7} {'lat':>7} {'tilt°':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['amplitude']:6.3f} {r['omega']:6.3f} "
              f"{r['peak_speed']:7.3f} {r['peak_accel']:7.3f} "
              f"{r['success_rate']:7.1f}% "
              f"{r['offset_mean']:7.3f} {r['lat_mean']:7.3f} "
              f"{r['tilt_deg_mean']:7.2f}")


def run(model_path, n_episodes, out_csv):
    model = PPO.load(model_path)

    print("=" * 74)
    print("  ACCELERATION-LIMIT TEST")
    print("=" * 74)
    print(f"  Policy trained at A=0.5, omega=0.5")
    print(f"    -> peak speed 0.250 m/s, peak acceleration 0.125 m/s^2")
    print(f"  Theoretical max lateral accel at {TILT_LIMIT_RAD} rad tilt: "
          f"{A_MAX_THEORY:.2f} m/s^2")
    print(f"  {n_episodes} episodes per configuration")
    print("=" * 74)

    all_rows = []

    # --- A. constant acceleration, varying speed ----------------------
    # Fixed at the value the policy was trained on.
    target_accel = 0.125
    speeds = [0.15, 0.25, 0.40, 0.60, 0.80, 1.00]
    rows_a = []
    for amplitude, omega in make_configs_constant_accel(target_accel, speeds):
        r = evaluate_config(model, amplitude, omega, n_episodes)
        r["family"] = "const_accel"
        rows_a.append(r)
        all_rows.append(r)

    print_table(f"A. CONSTANT ACCELERATION ({target_accel} m/s^2), VARYING SPEED\n"
                f"   Hypothesis predicts: success stays FLAT", rows_a)

    # --- B. constant speed, varying acceleration ----------------------
    target_speed = 0.25
    accels = [0.05, 0.125, 0.20, 0.30, 0.45, 0.70]
    rows_b = []
    for amplitude, omega in make_configs_constant_speed(target_speed, accels):
        r = evaluate_config(model, amplitude, omega, n_episodes)
        r["family"] = "const_speed"
        rows_b.append(r)
        all_rows.append(r)

    print_table(f"B. CONSTANT SPEED ({target_speed} m/s), VARYING ACCELERATION\n"
                f"   Hypothesis predicts: success FALLS as acceleration rises", rows_b)

    # --- verdict ------------------------------------------------------
    print("\n" + "=" * 74)
    print("  VERDICT")
    print("=" * 74)

    a_success = [r['success_rate'] for r in rows_a]
    b_success = [r['success_rate'] for r in rows_b]

    a_spread = max(a_success) - min(a_success)
    b_spread = max(b_success) - min(b_success)

    print(f"  Family A (speed varies, accel fixed) : "
          f"success ranges {min(a_success):.0f}% - {max(a_success):.0f}%  "
          f"(spread {a_spread:.0f})")
    print(f"  Family B (accel varies, speed fixed) : "
          f"success ranges {min(b_success):.0f}% - {max(b_success):.0f}%  "
          f"(spread {b_spread:.0f})")
    print()

    if b_spread > a_spread + 20:
        print("  -> ACCELERATION-LIMITED. Success is insensitive to platform")
        print("     speed but degrades sharply with platform acceleration.")
        print("     The binding constraint is the drone's ability to reverse")
        print("     direction, not its ability to keep up.")
    elif a_spread > b_spread + 20:
        print("  -> SPEED-LIMITED. Hypothesis rejected; the drone simply")
        print("     cannot fly fast enough.")
    else:
        print("  -> INCONCLUSIVE. Both quantities matter, or the tested")
        print("     ranges are too narrow to separate them.")

    # find the acceleration at which family B first drops
    degraded = [r for r in rows_b if r['success_rate'] < 100.0]
    if degraded:
        print(f"\n  First degradation in family B at "
              f"{degraded[0]['peak_accel']:.3f} m/s^2 "
              f"({degraded[0]['success_rate']:.0f}% success), which is "
              f"{100*degraded[0]['peak_accel']/A_MAX_THEORY:.1f}% of the "
              f"theoretical airframe limit.")
    print("=" * 74)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved to {out_csv}\n")


def _latest_model():
    runs = sorted(os.listdir('results_landing'))
    if not runs:
        print("No trained models found in results_landing/.")
        sys.exit(1)
    return os.path.join('results_landing', runs[-1], 'best_model.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--out', type=str, default='sweep_acceleration.csv')
    args = parser.parse_args()

    run(args.model or _latest_model(), args.episodes, args.out)
