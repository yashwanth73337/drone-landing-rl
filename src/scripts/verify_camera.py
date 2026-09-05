"""
Verify the camera and ArUco detection pipeline BEFORE training on it.

Why this exists
---------------
Earlier in this project, eight reward variants were tuned against an
environment that contained a bug making the landing bonus unpayable. Weeks were
spent optimising something that could never work. The lesson was: verify the
environment produces correct values before spending compute on it.

A perception pipeline is far easier to get silently wrong than a reward. Wrong
axis conventions, wrong corner ordering, a wrong focal length — all of these
produce plausible-looking numbers that are simply incorrect, and a policy
trained on them looks like an architecture failure rather than a bug.

What this checks
----------------
1. Is the marker detected at all, and from what heights?
2. Is the RECOVERED relative position close to ground truth?
3. At what height does detection actually stop?
4. How much does tilt cost?
5. How slow is rendering?

Usage:
    python verify_camera.py
    python verify_camera.py --gui        # watch it
    python verify_camera.py --save       # dump sample images to /tmp
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.LandingAviary import LandingAviary

try:
    import cv2
except ImportError:
    print("Needs opencv:  pip install opencv-contrib-python")
    sys.exit(1)


def check_static(gui=False, save=False):
    """Hold the drone at a series of heights and check detection at each.

    The platform is held stationary and the drone is teleported, so the only
    variable is height. This isolates the field-of-view geometry.
    """
    import pybullet as p

    env = LandingAviary(gui=gui, sensing='camera',
                        randomize_platform=False,
                        platform_amplitude=0.0)
    env.reset(seed=0)

    print("\n" + "=" * 72)
    print("  STATIC HEIGHT SWEEP")
    print(f"  marker {env.MARKER_SIZE} m,  camera {env.CAM_W}x{env.CAM_H}, "
          f"fov {env.CAM_FOV} deg")
    print("=" * 72)
    print(f"{'height':>8} {'detected':>9} {'true rel pos':>26} "
          f"{'measured rel pos':>26} {'error':>8}")
    print("-" * 72)

    heights = [1.20, 1.00, 0.80, 0.60, 0.50, 0.40, 0.35,
               0.30, 0.25, 0.20, 0.15, 0.10]

    first_fail = None

    for h in heights:
        # Place the drone directly above the pad centre, level.
        p.resetBasePositionAndOrientation(
            env.DRONE_IDS[0], [0.0, 0.0, h], [0, 0, 0, 1],
            physicsClientId=env.CLIENT)
        p.resetBaseVelocity(env.DRONE_IDS[0], [0, 0, 0], [0, 0, 0],
                            physicsClientId=env.CLIENT)
        env._updatePlatform()

        img = env._renderCamera()
        det_pos, ok = env._detectMarker(img)

        state = env._getDroneStateVector(0)
        true_rel = env._getPlatformPos() - state[0:3]

        if save:
            cv2.imwrite(f"/tmp/aruco_view_h{h:.2f}.png", img)

        if ok:
            err = np.linalg.norm(det_pos - true_rel)
            print(f"{h:8.2f} {'YES':>9} "
                  f"[{true_rel[0]:+6.3f} {true_rel[1]:+6.3f} {true_rel[2]:+6.3f}] "
                  f"[{det_pos[0]:+6.3f} {det_pos[1]:+6.3f} {det_pos[2]:+6.3f}] "
                  f"{err:8.4f}")
        else:
            if first_fail is None:
                first_fail = h
            print(f"{h:8.2f} {'no':>9} "
                  f"[{true_rel[0]:+6.3f} {true_rel[1]:+6.3f} {true_rel[2]:+6.3f}] "
                  f"{'-':>26} {'-':>8}")

    print("-" * 72)
    if first_fail is not None:
        predicted = env.MARKER_SIZE
        print(f"  Detection first failed at h = {first_fail:.2f} m")
        print(f"  Geometric prediction (h ~ marker side) = {predicted:.2f} m")
    else:
        print("  Detected at every height tested. Extend the sweep lower.")
    print("=" * 72)

    if save:
        print(f"\n  Sample images written to /tmp/aruco_view_h*.png")

    env.close()
    return first_fail


def check_offset(gui=False):
    """Move the drone sideways at fixed height. Checks that the recovered
    x and y are correct, not just the range."""
    import pybullet as p

    env = LandingAviary(gui=gui, sensing='camera',
                        randomize_platform=False,
                        platform_amplitude=0.0)
    env.reset(seed=0)

    print("\n" + "=" * 72)
    print("  LATERAL OFFSET SWEEP  (height fixed at 0.8 m)")
    print("=" * 72)
    print(f"{'offset x':>9} {'detected':>9} {'true rel':>20} "
          f"{'measured rel':>20} {'error':>8}")
    print("-" * 72)

    for dx in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        p.resetBasePositionAndOrientation(
            env.DRONE_IDS[0], [dx, 0.0, 0.8], [0, 0, 0, 1],
            physicsClientId=env.CLIENT)
        env._updatePlatform()

        img = env._renderCamera()
        det_pos, ok = env._detectMarker(img)

        state = env._getDroneStateVector(0)
        true_rel = env._getPlatformPos() - state[0:3]

        if ok:
            err = np.linalg.norm(det_pos - true_rel)
            print(f"{dx:9.2f} {'YES':>9} "
                  f"[{true_rel[0]:+6.3f} {true_rel[1]:+6.3f}] "
                  f"     [{det_pos[0]:+6.3f} {det_pos[1]:+6.3f}] "
                  f"     {err:8.4f}")
        else:
            print(f"{dx:9.2f} {'no':>9} "
                  f"[{true_rel[0]:+6.3f} {true_rel[1]:+6.3f}]"
                  f"{'-':>21} {'-':>8}")

    print("=" * 72)
    env.close()


def check_speed():
    """How much does rendering cost? Decides whether training is feasible."""
    env_cam = LandingAviary(sensing='camera')
    env_cam.reset(seed=0)
    action = np.array([[0.0, 0.0, 0.0, 0.0]])

    n = 200
    t0 = time.time()
    for _ in range(n):
        env_cam.step(action)
    t_cam = time.time() - t0
    env_cam.close()

    env_abs = LandingAviary(sensing='abstract')
    env_abs.reset(seed=0)
    t0 = time.time()
    for _ in range(n):
        env_abs.step(action)
    t_abs = time.time() - t0
    env_abs.close()

    print("\n" + "=" * 72)
    print("  THROUGHPUT")
    print("=" * 72)
    print(f"  abstract sensing : {n/t_abs:8.1f} steps/sec")
    print(f"  camera sensing   : {n/t_cam:8.1f} steps/sec"
          f"   ({t_cam/t_abs:.1f}x slower)")
    print()
    for steps in (500_000, 1_500_000):
        print(f"  {steps:,} steps would take "
              f"{steps/(n/t_cam)/3600:5.1f} hours with the camera, "
              f"{steps/(n/t_abs)/3600:5.1f} hours without")
    print("=" * 72)


def check_episode():
    """Run one full episode with fixed actions and report detection stats."""
    env = LandingAviary(sensing='camera')
    obs, info = env.reset(seed=0)

    action = np.array([[-0.05, -0.05, -0.05, -0.05]])   # gentle descent

    print("\n" + "=" * 72)
    print("  ONE EPISODE, SCRIPTED DESCENT")
    print("=" * 72)
    print(f"{'step':>5} {'height':>8} {'aruco':>7} {'meas x':>8} "
          f"{'true x':>8} {'err':>8}")
    print("-" * 72)

    for i in range(300):
        obs, r, term, trunc, info = env.step(action)

        if i % 15 == 0:
            state = env._getDroneStateVector(0)
            true_rel = env._getPlatformPos() - state[0:3]
            meas = obs[0, -8:-5]
            aruco = obs[0, -1]
            err = np.linalg.norm(meas - true_rel)
            print(f"{i:5d} {info['height_above_pad']:8.3f} "
                  f"{'YES' if aruco > 0.5 else 'no':>7} "
                  f"{meas[0]:8.3f} {true_rel[0]:8.3f} {err:8.4f}")

        if term or trunc:
            print(f"\n  Episode ended at step {i}: "
                  f"terminated={term} truncated={trunc}")
            break

    print("-" * 72)
    print(f"  Blind for {100*info['blind_fraction']:.1f}% of steps")
    print(f"  Mean detection error while visible: "
          f"{info['detect_error_mean']:.4f} m")
    print("=" * 72)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--save', action='store_true',
                        help='write sample camera images to /tmp')
    parser.add_argument('--skip-speed', action='store_true')
    args = parser.parse_args()

    check_static(gui=args.gui, save=args.save)
    check_offset(gui=args.gui)
    check_episode()
    if not args.skip_speed:
        check_speed()

    print("\nWhat to look for:")
    print("  - detection error should be a few centimetres, not tens")
    print("  - measured x should TRACK true x, with the same sign")
    print("  - detection should fail somewhere near h = marker size")
    print("  - if error is large or the sign is wrong, the pose recovery is")
    print("    broken and training on it would be meaningless\n")
