"""
Training and evaluation for LandingAviary.

Usage:
    python train_landing.py                    # train
    python train_landing.py --gui              # watch one landing
    python train_landing.py --eval             # 100 episodes, metrics table
    python train_landing.py --eval --note "GRU policy"   # tag the CSV row

Every evaluation is appended to results_log.csv so results are never lost to a
terminal scrollback buffer.
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.LandingAviary import LandingAviary


RESULTS_LOG = 'results_log.csv'


def train(total_timesteps=500_000, output_dir='results_landing', continue_from=None):
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_env = make_vec_env(LandingAviary, n_envs=4, seed=0)
    eval_env = LandingAviary()

    if continue_from:
        print(f"\nContinuing training from: {continue_from}\n")
        model = PPO.load(continue_from, env=train_env,
                         tensorboard_log=os.path.join(run_dir, 'tb'))
    else:
        model = PPO('MlpPolicy', train_env, verbose=1,
                    tensorboard_log=os.path.join(run_dir, 'tb'))

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=run_dir,
                                 log_path=run_dir,
                                 eval_freq=5000,
                                 deterministic=True,
                                 render=False)

    print(f"\nTraining for {total_timesteps} steps. Output: {run_dir}\n")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(run_dir, 'final_model'))
    print(f"\nDone. Saved to {run_dir}\n")
    return run_dir


def _append_to_log(row):
    """Append one evaluation to the CSV log, writing a header if new.

    Results are cheap to compute but expensive to lose. Everything printed to
    the terminal disappears with the scrollback buffer; this does not.
    """
    fields = list(row.keys())
    new_file = not os.path.exists(RESULTS_LOG)

    with open(RESULTS_LOG, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    print(f"  Appended to {RESULTS_LOG}")


def evaluate(model_path, n_episodes=100, note=""):
    """Run many episodes and report the metrics table.

    Success rate is what the literature reports. The touchdown columns are what
    it does not, and are the contribution of this work.
    """
    env = LandingAviary()
    model = PPO.load(model_path)

    successes = 0
    touchdowns = 0
    vz_list, lat_list, tilt_list, offset_list = [], [], [], []

    # Platform parameters of the episodes that FAILED, so failures can be
    # attributed to specific motion regimes rather than treated as noise.
    fail_speed, fail_accel = [], []

    reason_counts = Counter()
    aborted_episodes = 0
    below_platform_horiz = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        if info.get("touchdown"):
            touchdowns += 1

        if info.get("success"):
            successes += 1
            vz_list.append(info["touchdown_vz"])
            lat_list.append(info["touchdown_lateral"])
            tilt_list.append(info["touchdown_tilt"])
            offset_list.append(info["touchdown_offset"])
        else:
            fail_speed.append(info.get("plat_peak_speed", float('nan')))
            fail_accel.append(info.get("plat_peak_accel", float('nan')))
            reason = info.get("truncation_reason") or "terminated_no_reason"
            reason_counts[reason] += 1
            if reason == "below_platform" and info.get("truncation_horiz_dist") is not None:
                below_platform_horiz.append(info["truncation_horiz_dist"])
            if info.get("failed_attempts", 0) > 0:
                aborted_episodes += 1

    env.close()

    print("\n" + "=" * 58)
    print(f"  EVALUATION OVER {n_episodes} EPISODES")
    if note:
        print(f"  {note}")
    print("=" * 58)
    print(f"  Reached the ground        : {touchdowns}/{n_episodes}")
    print(f"  Landed ON the pad         : {successes}/{n_episodes}"
          f"  ({100*successes/n_episodes:.1f}%)")

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_path,
        "note": note,
        "n_episodes": n_episodes,
        "success_rate": round(100.0 * successes / n_episodes, 2),
        "touchdown_rate": round(100.0 * touchdowns / n_episodes, 2),
    }

    if successes > 0:
        print("-" * 58)
        print("  Touchdown quality (successful landings only)")
        print(f"    vertical speed   : {np.mean(vz_list):.3f} +/- {np.std(vz_list):.3f} m/s"
              f"   (p95 {np.percentile(vz_list, 95):.3f})")
        print(f"    lateral speed    : {np.mean(lat_list):.3f} +/- {np.std(lat_list):.3f} m/s")
        print(f"    tilt at contact  : {np.degrees(np.mean(tilt_list)):.2f} +/- "
              f"{np.degrees(np.std(tilt_list)):.2f} deg")
        print(f"    offset from centre: {np.mean(offset_list):.3f} +/- "
              f"{np.std(offset_list):.3f} m")
        print("-" * 58)
        print("  Constraint violations (proposed safety thresholds)")
        print(f"    vz    > 0.5 m/s  : {100*np.mean(np.array(vz_list) > 0.5):.1f}%")
        print(f"    lat   > 0.3 m/s  : {100*np.mean(np.array(lat_list) > 0.3):.1f}%")
        print(f"    tilt  > 10 deg   : "
              f"{100*np.mean(np.degrees(tilt_list) > 10):.1f}%")

        row.update({
            "vz_mean": round(float(np.mean(vz_list)), 4),
            "vz_std": round(float(np.std(vz_list)), 4),
            "vz_p95": round(float(np.percentile(vz_list, 95)), 4),
            "lat_mean": round(float(np.mean(lat_list)), 4),
            "lat_std": round(float(np.std(lat_list)), 4),
            "tilt_deg_mean": round(float(np.degrees(np.mean(tilt_list))), 3),
            "tilt_deg_std": round(float(np.degrees(np.std(tilt_list))), 3),
            "offset_mean": round(float(np.mean(offset_list)), 4),
            "offset_std": round(float(np.std(offset_list)), 4),
            "viol_vz": round(100 * float(np.mean(np.array(vz_list) > 0.5)), 2),
            "viol_lat": round(100 * float(np.mean(np.array(lat_list) > 0.3)), 2),
            "viol_tilt": round(100 * float(np.mean(np.degrees(tilt_list) > 10)), 2),
        })
    else:
        for k in ["vz_mean", "vz_std", "vz_p95", "lat_mean", "lat_std",
                  "tilt_deg_mean", "tilt_deg_std", "offset_mean", "offset_std",
                  "viol_vz", "viol_lat", "viol_tilt"]:
            row[k] = ""

    # Which motion regimes did it fail on?
    if fail_speed and not all(np.isnan(fail_speed)):
        print("-" * 58)
        print("  Failed episodes — platform motion")
        print(f"    peak speed  : {np.nanmean(fail_speed):.3f} m/s "
              f"(range {np.nanmin(fail_speed):.3f} - {np.nanmax(fail_speed):.3f})")
        print(f"    peak accel  : {np.nanmean(fail_accel):.3f} m/s^2 "
              f"(range {np.nanmin(fail_accel):.3f} - {np.nanmax(fail_accel):.3f})")
        row["fail_speed_mean"] = round(float(np.nanmean(fail_speed)), 4)
        row["fail_accel_mean"] = round(float(np.nanmean(fail_accel)), 4)
    else:
        row["fail_speed_mean"] = ""
        row["fail_accel_mean"] = ""

    print("-" * 58)
    print("  Why failed episodes ended")
    for reason, count in reason_counts.most_common():
        print(f"    {reason:20s}: {count}")
    print(f"    episodes with >=1 aborted attempt: {aborted_episodes}")

    if below_platform_horiz:
        print(f"    below_platform horiz_dist: mean {np.mean(below_platform_horiz):.3f} m, "
              f"range {np.min(below_platform_horiz):.3f}-{np.max(below_platform_horiz):.3f} m "
              f"(pad half-size {env.PLAT_SIZE:.3f} m)")
    print("=" * 58)
    _append_to_log(row)
    print()


def play(model_path):
    """Watch a single landing."""
    env = LandingAviary(gui=True)
    model = PPO.load(model_path)

    obs, info = env.reset(seed=np.random.randint(0, 100000))
    for _ in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1 / env.CTRL_FREQ)
        if terminated or truncated:
            break
    env.close()

    print(f"\n  Platform: amplitude {info.get('plat_amplitude', 0):.3f} m, "
          f"omega {info.get('plat_omega', 0):.3f} rad/s")
    if info.get("touchdown"):
        print(f"  Landed on pad : {info.get('success')}")
        print(f"  vertical speed: {info.get('touchdown_vz'):.3f} m/s")
        print(f"  lateral speed : {info.get('touchdown_lateral'):.3f} m/s")
        print(f"  tilt          : {np.degrees(info.get('touchdown_tilt')):.2f} deg")
        print(f"  offset        : {info.get('touchdown_offset'):.3f} m\n")
    else:
        print("  No touchdown — episode was truncated.\n")


def _latest_model():
    runs = sorted(os.listdir('results_landing'))
    if not runs:
        print("No trained models found. Run without flags first.")
        sys.exit(1)
    return os.path.join('results_landing', runs[-1], 'best_model.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='watch one landing')
    parser.add_argument('--eval', action='store_true', help='run the metrics table')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=500_000)
    parser.add_argument('--continue-from', type=str, default=None,
                        help='path to an existing best_model.zip to continue training from')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--note', type=str, default="",
                        help='label for this run in results_log.csv')
    args = parser.parse_args()

    if args.gui:
        play(args.model or _latest_model())
    elif args.eval:
        evaluate(args.model or _latest_model(), args.episodes, args.note)
    else:
        train(total_timesteps=args.steps, continue_from=args.continue_from)
