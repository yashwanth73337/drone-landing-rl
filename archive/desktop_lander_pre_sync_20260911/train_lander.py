"""
train_lander.py
================

Training and evaluation for LanderAIAviary, using TD3 -- matching the
algorithm and hyperparameters reported in Peter et al., "Lander.AI:
DRL-based Autonomous Drone Landing on Moving 3D Surface in the Presence of
Aerodynamic Disturbances" (ICUAS 2024), Table I.

Usage:
    python train_lander.py                      # train
    python train_lander.py --gui                # watch one landing
    python train_lander.py --eval                # 100 episodes, metrics
    python train_lander.py --eval --note "..."   # tag the CSV row


HYPERPARAMETERS -- PAPER FACT (Table I) vs. THIS FILE'S CHOICES
------------------------------------------------------------------------------
    Algorithm            TD3                          PAPER FACT
    Policy                MlpPolicy                    PAPER FACT
    Learning rate         0.0001                       PAPER FACT
    Buffer size            1,000,000                    PAPER FACT
    Batch size             100                          PAPER FACT
    Activation             ReLU                         PAPER FACT
    Optimizer              Adam                         PAPER FACT (SB3 TD3
                                                          default -- nothing
                                                          to configure)
    Network architecture   FC512 x2 -> FC256 -> FC128    PAPER FACT for the
                                                          ACTOR (Eq. 4). The
                                                          paper does not
                                                          describe a separate
                                                          critic architecture
                                                          -- the SAME
                                                          architecture is
                                                          applied to the
                                                          critic here too,
                                                          as a documented
                                                          assumption, not a
                                                          paper-stated fact.
    learning_starts=100    "Training begins after       PAPER FACT
                            100 steps"
    Episode duration        20 s                         PAPER FACT (also
                                                          baked into
                                                          LanderAIAviary's
                                                          EPISODE_LEN_SEC)
    Total training steps   5,000,000 initial, up to     PAPER FACT, but NOT
                            35,000,000 extended          USED HERE -- not
                                                          feasible in a single
                                                          session's remaining
                                                          time. --steps
                                                          defaults to a much
                                                          smaller number (see
                                                          CLI default below);
                                                          THIS IS A DOCUMENTED
                                                          DEVIATION driven by
                                                          time, not a claim
                                                          that this matches
                                                          the paper's own
                                                          training budget.

train_freq, gradient_steps, policy_delay, target_policy_noise, and other TD3
internals not mentioned in the paper are left at SB3's own defaults --
NOT paper-specified, not independently chosen here either.
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
from collections import Counter

import torch
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.LanderAIAviary import LanderAIAviary


RESULTS_LOG = 'results_lander_log.csv'

# Eq. 4 of the paper (PAPER FACT for the actor; applied to the critic too
# as a documented assumption -- see module docstring).
NET_ARCH = [512, 512, 256, 128]


def train(total_timesteps=200_000, output_dir='results_lander', continue_from=None):
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_env = make_vec_env(LanderAIAviary, n_envs=4, seed=0)
    eval_env = LanderAIAviary()

    # TD3 needs exploration noise on top of its deterministic policy (unlike
    # PPO's built-in stochastic action sampling). Not specified in the
    # paper's Table I -- standard TD3 default (N(0, 0.1) per action dim) is
    # used here, documented as such.
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                     sigma=0.1 * np.ones(n_actions))

    policy_kwargs = dict(
        net_arch=NET_ARCH,
        activation_fn=torch.nn.ReLU,
    )

    if continue_from:
        print(f"\nContinuing training from: {continue_from}\n")
        model = TD3.load(continue_from, env=train_env,
                         tensorboard_log=os.path.join(run_dir, 'tb'))
    else:
        model = TD3('MlpPolicy', train_env, verbose=1,
                    learning_rate=0.0001,       # PAPER FACT
                    buffer_size=1_000_000,      # PAPER FACT
                    batch_size=100,             # PAPER FACT
                    learning_starts=100,        # PAPER FACT
                    action_noise=action_noise,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=os.path.join(run_dir, 'tb'))

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=run_dir,
                                 log_path=run_dir,
                                 eval_freq=5000,
                                 deterministic=True,
                                 render=False)

    print(f"\nTraining for {total_timesteps} steps. Output: {run_dir}\n")
    print("NOTE: paper's own initial training budget is 5,000,000 steps "
          "(Table I) -- this run uses far fewer, a documented deviation "
          "driven by today's time constraints, not a claim of matching "
          "the paper's training regime.\n")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(run_dir, 'final_model'))
    print(f"\nDone. Saved to {run_dir}\n")
    return run_dir


def _append_to_log(row):
    fields = list(row.keys())
    new_file = not os.path.exists(RESULTS_LOG)
    with open(RESULTS_LOG, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Appended to {RESULTS_LOG}")


def evaluate(model_path, n_episodes=100, note=""):
    """Run many episodes and report metrics comparable to the paper's own
    reported figures (Table I: success rate: Table II: mean distance to
    target at landing, in cm)."""
    env = LanderAIAviary()
    model = TD3.load(model_path)

    successes = 0
    final_distances = []       # cm, successful landings only -- matches
                               # the paper's Table II metric
    reason_counts = Counter()
    plat_speeds_fail = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        if info.get("success"):
            successes += 1
            final_distances.append(info["d_target"] * 100.0)   # m -> cm
        else:
            reason = info.get("truncation_reason") or "terminated_no_reason"
            reason_counts[reason] += 1
            plat_speeds_fail.append(info.get("plat_speed", float('nan')))

    env.close()

    print("\n" + "=" * 58)
    print(f"  EVALUATION OVER {n_episodes} EPISODES")
    if note:
        print(f"  {note}")
    print("=" * 58)
    print(f"  Landed successfully       : {successes}/{n_episodes}"
          f"  ({100*successes/n_episodes:.1f}%)")
    print(f"  (Paper's own LMPL result was 93.33% -- compare against that,"
          f" not 100%, since this is the linear-motion scenario)")

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_path,
        "note": note,
        "n_episodes": n_episodes,
        "success_rate": round(100.0 * successes / n_episodes, 2),
    }

    if final_distances:
        print("-" * 58)
        print("  Landing precision (successful landings only, cm)")
        print(f"    mean {np.mean(final_distances):.2f} +/- "
              f"{np.std(final_distances):.2f}   "
              f"(paper's LMPL: 6.50 +/- 2.14 cm)")
        row["dist_mean_cm"] = round(float(np.mean(final_distances)), 3)
        row["dist_std_cm"] = round(float(np.std(final_distances)), 3)
    else:
        row["dist_mean_cm"] = ""
        row["dist_std_cm"] = ""

    print("-" * 58)
    print("  Why failed episodes ended")
    for reason, count in reason_counts.most_common():
        print(f"    {reason:20s}: {count}")
    if plat_speeds_fail and not all(np.isnan(plat_speeds_fail)):
        print(f"    failed episodes' platform speed: "
              f"mean {np.nanmean(plat_speeds_fail):.3f} m/s")

    print("=" * 58)
    _append_to_log(row)
    print()


def play(model_path):
    """Watch a single landing. NOTE: GUI rendering has previously crashed
    on Intel/Mesa driver setups in this project -- try
    LIBGL_ALWAYS_SOFTWARE=1 python train_lander.py --gui if it crashes."""
    env = LanderAIAviary(gui=True)
    model = TD3.load(model_path)

    obs, info = env.reset(seed=np.random.randint(0, 100000))
    for _ in range(int(env.EPISODE_LEN_SEC * env.CTRL_FREQ)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1 / env.CTRL_FREQ)
        if terminated or truncated:
            break
    env.close()

    print(f"\n  Platform speed: {info.get('plat_speed', 0):.3f} m/s")
    print(f"  Success: {info.get('success')}")
    print(f"  Final distance to target: {info.get('d_target', float('nan'))*100:.2f} cm\n")


def _latest_model():
    runs = sorted(os.listdir('results_lander'))
    if not runs:
        print("No trained models found. Run without flags first.")
        sys.exit(1)
    return os.path.join('results_lander', runs[-1], 'best_model.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='watch one landing')
    parser.add_argument('--eval', action='store_true', help='run the metrics table')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=200_000,
                        help='paper uses 5,000,000 initial -- default here is '
                             'much smaller due to today\'s time constraints')
    parser.add_argument('--continue-from', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--note', type=str, default="")
    args = parser.parse_args()

    if args.gui:
        play(args.model or _latest_model())
    elif args.eval:
        evaluate(args.model or _latest_model(), args.episodes, args.note)
    else:
        train(total_timesteps=args.steps, continue_from=args.continue_from)
