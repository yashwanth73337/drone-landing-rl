"""
Training and evaluation for LandingAviary.

Usage:
    python train_landing.py                    # train
    python train_landing.py --gui              # watch one landing
    python train_landing.py --eval             # run 100 episodes, print metrics table
"""

import os
import sys
import time
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.LandingAviary import LandingAviary


def train(total_timesteps=500_000, output_dir='results_landing'):
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_env = make_vec_env(LandingAviary, n_envs=4, seed=0)
    eval_env = LandingAviary()

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


def evaluate(model_path, n_episodes=100):
    """Run many episodes and report the metrics that go in your results table.

    Success rate is what the literature reports. The touchdown columns are
    what it does NOT report, and are the contribution of this work.
    """
    env = LandingAviary()
    model = PPO.load(model_path)

    successes = 0
    touchdowns = 0
    vz_list, lat_list, tilt_list, offset_list = [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info.get("touchdown"):
            touchdowns += 1
            if info.get("success"):
                successes += 1
                vz_list.append(info["touchdown_vz"])
                lat_list.append(info["touchdown_lateral"])
                tilt_list.append(info["touchdown_tilt"])
                offset_list.append(info["touchdown_offset"])

    env.close()

    print("\n" + "=" * 58)
    print(f"  EVALUATION OVER {n_episodes} EPISODES")
    print("=" * 58)
    print(f"  Reached the ground        : {touchdowns}/{n_episodes}")
    print(f"  Landed ON the pad         : {successes}/{n_episodes}"
          f"  ({100*successes/n_episodes:.1f}%)")

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
    print("=" * 58 + "\n")


def play(model_path):
    """Watch a single landing."""
    env = LandingAviary(gui=True)
    model = PPO.load(model_path)

    obs, info = env.reset(seed=42)
    for _ in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1 / env.CTRL_FREQ)
        if terminated or truncated:
            break
    env.close()

    if info.get("touchdown"):
        print(f"\n  Landed on pad : {info.get('success')}")
        print(f"  vertical speed: {info.get('touchdown_vz'):.3f} m/s")
        print(f"  lateral speed : {info.get('touchdown_lateral'):.3f} m/s")
        print(f"  tilt          : {np.degrees(info.get('touchdown_tilt')):.2f} deg")
        print(f"  offset        : {info.get('touchdown_offset'):.3f} m\n")
    else:
        print("\n  No touchdown — episode was truncated.\n")


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
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    if args.gui:
        play(args.model or _latest_model())
    elif args.eval:
        evaluate(args.model or _latest_model(), args.episodes)
    else:
        train(total_timesteps=args.steps)
