"""
Training script for TrackingAviary.

Usage:
    python train_tracking.py                # train
    python train_tracking.py --gui          # watch the trained policy fly
"""

import os
import sys
import time
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.TrackingAviary import TrackingAviary


def train(total_timesteps=200_000, output_dir='results'):
    """Train a PPO policy to follow the moving target."""

    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # Four environments running in parallel — more experience per wall-clock second.
    train_env = make_vec_env(TrackingAviary, n_envs=4, seed=0)
    eval_env = TrackingAviary()

    model = PPO('MlpPolicy',
                train_env,
                verbose=1,
                tensorboard_log=os.path.join(run_dir, 'tb'))

    # Periodically test the policy and save whichever version scores best.
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=run_dir,
                                 log_path=run_dir,
                                 eval_freq=2000,
                                 deterministic=True,
                                 render=False)

    print(f"\nTraining for {total_timesteps} steps. Output: {run_dir}\n")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save(os.path.join(run_dir, 'final_model'))
    print(f"\nDone. Saved to {run_dir}\n")
    return run_dir


def play(model_path):
    """Load a trained policy and watch it fly."""

    env = TrackingAviary(gui=True)
    model = PPO.load(model_path)

    obs, info = env.reset(seed=42)
    errors = []

    for i in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        errors.append(info["tracking_error"])
        env.render()
        time.sleep(1 / env.CTRL_FREQ)   # slow to real time so it's watchable

        if terminated or truncated:
            break

    env.close()
    print(f"\nMean tracking error: {np.mean(errors):.3f} m")
    print(f"Max tracking error:  {np.max(errors):.3f} m\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true',
                        help='watch the most recently trained policy')
    parser.add_argument('--steps', type=int, default=200_000,
                        help='training timesteps')
    parser.add_argument('--model', type=str, default=None,
                        help='path to a specific model to play')
    args = parser.parse_args()

    if args.gui:
        if args.model:
            play(args.model)
        else:
            runs = sorted(os.listdir('results'))
            if not runs:
                print("No trained models found. Run without --gui first.")
                sys.exit(1)
            play(os.path.join('results', runs[-1], 'best_model.zip'))
    else:
        train(total_timesteps=args.steps)
