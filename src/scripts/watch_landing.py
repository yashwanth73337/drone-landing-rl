"""
Watch a landing, with the simulation continuing past touchdown.

WHY THIS SCRIPT EXISTS
----------------------
LandingAviary ends the episode when the drone reaches 4 cm above the pad
surface:

    if drone_pos[2] - plat_pos[2] > 0.04:
        return False        # not yet
    self.touchdown_recorded = True

That threshold is a deliberate modelling choice — contact reporting in a
physics engine depends on collision-mesh detail and is not reproducible across
runs, whereas a height condition is. All four touchdown metrics are recorded at
that instant and are unaffected by anything that happens afterwards.

But it makes for a poor visualisation. The episode terminates and everything
freezes, so on screen the drone appears to stop 4 cm in the air and hang there.
It never visibly settles onto the platform.

This script records the metrics at the normal touchdown instant, then keeps
stepping physics for a further second so the drone actually comes to rest on
the pad. Nothing measured changes; only what you see.

Usage:
    python watch_landing.py                    # one episode, random motion
    python watch_landing.py --slow 3           # 3x slower than real time
    python watch_landing.py --seed 42          # a specific episode
    python watch_landing.py --best 10          # try 10, show the softest one
"""

import os
import sys
import time
import argparse
import numpy as np
import pybullet as p

from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.LandingAviary import LandingAviary


def _latest_model(directory='results_landing'):
    runs = sorted(os.listdir(directory))
    if not runs:
        print(f"No trained models in {directory}/.")
        sys.exit(1)
    return os.path.join(directory, runs[-1], 'best_model.zip')


def find_good_episode(model_path, n_tries, sensing):
    """Run several episodes headless and return the seed of the best landing.

    'Best' = successful, with the lowest touchdown vertical speed. Useful for
    picking a clip to record: at ~46% success roughly half of what you watch
    will be a failure, and the successes vary widely in quality.
    """
    env = LandingAviary(sensing=sensing)
    model = PPO.load(model_path)

    candidates = []
    print(f"\nScanning {n_tries} episodes for a clean landing...")

    for ep in range(n_tries):
        seed = np.random.randint(0, 100000)
        obs, info = env.reset(seed=seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            if term or trunc:
                break

        if info.get("success"):
            candidates.append((info["touchdown_vz"], seed, info))
            print(f"  seed {seed:6d}  vz {info['touchdown_vz']:.3f} m/s  "
                  f"offset {info['touchdown_offset']:.3f} m  "
                  f"platform {info['plat_peak_speed']:.2f} m/s")

    env.close()

    if not candidates:
        print("  No successful landing found. Try more episodes.")
        return None

    candidates.sort()
    best_vz, best_seed, best_info = candidates[0]
    print(f"\nBest: seed {best_seed}, vz {best_vz:.3f} m/s\n")
    return best_seed


def watch(model_path, seed=None, slow=2.0, sensing='abstract',
          settle_seconds=1.0):
    """Fly one episode with the GUI, continuing physics past touchdown.

    Parameters
    ----------
    slow : float
        Playback slowdown. 1.0 is real time; 2.0 is half speed. Landings take
        1.5-2 seconds, which is too fast to follow at real time.
    settle_seconds : float
        How long to keep stepping physics after touchdown, so the drone
        visibly comes to rest instead of freezing in mid-air.
    """
    if seed is None:
        seed = np.random.randint(0, 100000)

    env = LandingAviary(gui=True, sensing=sensing)
    model = PPO.load(model_path)

    obs, info = env.reset(seed=seed)

    # Slightly closer camera than the default, and angled to show the descent.
    p.resetDebugVisualizerCamera(
        cameraDistance=1.8,
        cameraYaw=-40,
        cameraPitch=-25,
        cameraTargetPosition=[0, 0, 0.3],
        physicsClientId=env.CLIENT)

    dt = slow / env.CTRL_FREQ
    touchdown_info = None

    for _ in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(dt)

        if terminated or truncated:
            touchdown_info = info
            break

    # --- keep the physics running so the drone settles ----------------
    #
    # The episode is over and every metric is already recorded. From here we
    # only step PyBullet directly, so nothing that follows can affect the
    # reported numbers.
    if touchdown_info is not None and touchdown_info.get("touchdown"):
        n_settle = int(settle_seconds * env.PYB_FREQ)
        for i in range(n_settle):
            env._updatePlatform()          # keep the pad moving
            p.stepSimulation(physicsClientId=env.CLIENT)
            if i % 8 == 0:
                time.sleep(dt / 8)

    _report(touchdown_info, seed)
    time.sleep(3)
    env.close()


def _report(info, seed):
    print("\n" + "=" * 52)
    print(f"  EPISODE  (seed {seed})")
    print("=" * 52)

    if info is None:
        print("  Episode did not end normally.")
        print("=" * 52 + "\n")
        return

    print(f"  Platform amplitude : {info.get('plat_amplitude', 0):.3f} m")
    print(f"  Platform frequency : {info.get('plat_omega', 0):.3f} rad/s")
    print(f"  Platform peak speed: {info.get('plat_peak_speed', 0):.3f} m/s")

    if info.get('sensing') == 'camera':
        print(f"  Blind fraction     : {100*info.get('blind_fraction', 0):.1f}%")

    print("-" * 52)

    if not info.get("touchdown"):
        print("  No touchdown — the drone never committed to a descent.")
        print("=" * 52 + "\n")
        return

    vz = info["touchdown_vz"]
    lat = info["touchdown_lateral"]
    tilt = np.degrees(info["touchdown_tilt"])
    off = info["touchdown_offset"]

    print(f"  Landed on pad      : {info.get('success')}")
    print(f"  Vertical speed     : {vz:.3f} m/s   "
          f"{'OK' if vz <= 0.5 else 'VIOLATION (> 0.5)'}")
    print(f"  Lateral speed      : {lat:.3f} m/s   "
          f"{'OK' if lat <= 0.3 else 'VIOLATION (> 0.3)'}")
    print(f"  Tilt at contact    : {tilt:.2f} deg   "
          f"{'OK' if tilt <= 10 else 'VIOLATION (> 10)'}")
    print(f"  Offset from centre : {off:.3f} m")
    print("=" * 52 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                        help='specific episode to replay')
    parser.add_argument('--slow', type=float, default=2.0,
                        help='playback slowdown; 2.0 = half speed')
    parser.add_argument('--best', type=int, default=0,
                        help='scan N episodes first, then show the softest landing')
    parser.add_argument('--sensing', type=str, default='abstract',
                        choices=['privileged', 'abstract', 'camera'],
                        help="'camera' deadlocks with the GUI on some systems")
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dir', type=str, default='results_landing')
    args = parser.parse_args()

    model_path = args.model or _latest_model(args.dir)
    print(f"Model: {model_path}")

    seed = args.seed
    if args.best > 0 and seed is None:
        seed = find_good_episode(model_path, args.best, args.sensing)
        if seed is None:
            sys.exit(1)

    watch(model_path, seed=seed, slow=args.slow, sensing=args.sensing)
