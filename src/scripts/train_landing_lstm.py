"""
Training and evaluation for LandingAviary with an LSTM (recurrent) policy.

This is the ablation against train_landing.py. Everything is held fixed —
environment, reward, sensor model, training budget — and only the policy
network changes:

    train_landing.py       PPO           + MlpPolicy       (memoryless)
    train_landing_lstm.py  RecurrentPPO  + MlpLstmPolicy   (memory)

RecurrentPPO is NOT a different algorithm. It is PPO with a recurrent network.
It exists as a separate class for one bookkeeping reason: standard PPO shuffles
collected transitions before computing gradients, which is harmless for a
memoryless network but destroys a recurrent one, since step 50 only means
anything after steps 1-49. RecurrentPPO keeps sequences intact and carries
hidden states correctly through the update.

NOTE ON NAMING: sb3-contrib provides LSTM only, not GRU. Earlier notes in this
project said "GRU"; the implementation is an LSTM. Both are gated recurrent
networks serving the same purpose here, but the reports should say LSTM.


WHY MEMORY SHOULD HELP
----------------------
The sensor model makes the ArUco marker disappear below ~30 cm, because the
marker leaves the camera's field of view. The drone is blind for the final
stretch of every descent.

A memoryless policy sees only the present instant, so when the marker vanishes
it commits blind. Measured over three seeds:

    condition            success      lateral speed    lat > 0.3 m/s
    privileged state     83.3%        0.068 m/s         0.0%
    sensor model, MLP    96.7%        0.270 m/s        34.9%

Success went UP while landings became substantially more dangerous. With poor
information, waiting does not help, so committing early maximises reward. One
landing in three arrives sliding sideways fast enough to tumble a real
airframe.


TRAINING HISTORY OF THIS SCRIPT
-------------------------------
Run 1 (defaults, 1.5M steps)
    Failed completely. ep_len_mean pinned at exactly 36.0 across all 75
    evaluations — the drone free-fell every episode and never learned.
    Cause: RecurrentPPO defaults to n_steps=128, against PPO's 2048. With 4
    environments that is 512 steps per update, rarely containing a complete
    landing, so the terminal bonus almost never appeared in a rollout and
    there was no learning signal.

Run 2 (n_steps=1024, 1.5M steps)
    Learned, but did not converge. Reward peaked near 328 at 440k steps and
    drifted to 304 by 1.5M. Evaluation: 68% success, 26.5% lateral violations.
    Better than the MLP on lateral violations (34.9%), much worse on success
    (96.7%).

Run 3 (n_steps=1024, 3M steps)
    COLLAPSED. Reward fell to 4.75 and episodes to 6.6 steps. Diagnostics:

        approx_kl     1.047    (healthy range ~0.01-0.03)
        clip_fraction 0.777    (healthy range ~0.1-0.3)
        std           0.077    (collapsed from 0.482)

    approx_kl at 50x the healthy range means each update was violently
    rewriting the policy. std collapsing to 0.077 means exploration stopped
    entirely and the policy locked onto one bad behaviour. This is policy
    collapse, not under-training — run 2 was likely already mid-decline rather
    than plateaued.

Run 4 (this version) — STABILISED
    Three changes, all aimed at the collapse:

        learning_rate  3e-4 -> 1e-4      smaller steps per update
        target_kl      None -> 0.02      abort an update that moves the
                                         policy too far. This is the direct
                                         guard against what killed run 3.
        max_grad_norm  0.5  -> 0.5       (unchanged, but stated explicitly)
        ent_coef       0.0  -> 0.005     small entropy bonus, so the policy
                                         does not become deterministic and
                                         stop exploring

    Recurrent policies are known to be unstable in RL — this is why GTrXL
    exists for transformers. The fix is smaller, more constrained updates.


Usage:
    python train_landing_lstm.py --steps 2000000
    python train_landing_lstm.py --eval --note "LSTM stabilised, seed 1"
    python train_landing_lstm.py --gui
"""

import os
import sys
import csv
import time
import argparse
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.LandingAviary import LandingAviary


RESULTS_LOG = 'results_log.csv'
OUTPUT_DIR = 'results_recurrent'


def train(total_timesteps=2_000_000):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_dir = os.path.join(OUTPUT_DIR, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_env = make_vec_env(LandingAviary, n_envs=4, seed=0)
    eval_env = LandingAviary()

    model = RecurrentPPO(
        'MlpLstmPolicy',
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(run_dir, 'tb'),

        # --- rollout size -----------------------------------------------
        # RecurrentPPO defaults to n_steps=128, against PPO's 2048. Episodes
        # here run 36-300 steps, so 128 x 4 envs rarely contains a complete
        # landing and the terminal bonus almost never appears in a rollout.
        # 1024 x 4 = 4096 steps per update, enough for many full episodes.
        n_steps=1024,
        batch_size=256,

        # --- stability --------------------------------------------------
        # Run 3 collapsed with approx_kl at 1.047, roughly 50x healthy.
        # Smaller steps, and a hard stop on any update that moves the policy
        # too far.
        learning_rate=1e-4,
        target_kl=0.02,
        max_grad_norm=0.5,

        # Small entropy bonus. Run 3's action std collapsed to 0.077, meaning
        # exploration stopped and the policy locked onto one bad behaviour.
        ent_coef=0.005,

        # --- unchanged from PPO defaults --------------------------------
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        n_epochs=10,

        policy_kwargs=dict(
            lstm_hidden_size=256,
            n_lstm_layers=1,
            shared_lstm=False,          # separate recurrent state for actor and critic
            enable_critic_lstm=True,
        ),
    )

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=run_dir,
                                 log_path=run_dir,
                                 eval_freq=5000,
                                 deterministic=True,
                                 render=False)

    print(f"\nTraining stabilised LSTM policy for {total_timesteps} steps.")
    print(f"  learning_rate 1e-4, target_kl 0.02, ent_coef 0.005")
    print(f"  Output: {run_dir}")
    print(f"\n  WATCH approx_kl. It should stay near 0.02. If it climbs past")
    print(f"  0.1 the policy is being rewritten too fast and will collapse.\n")

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(run_dir, 'final_model'))
    print(f"\nDone. Saved to {run_dir}\n")
    return run_dir


def _append_to_log(row):
    """Append one evaluation to the CSV log, writing a header if new."""
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

    A recurrent policy needs its hidden state threaded through the episode and
    RESET at every episode boundary. Forgetting the reset leaks memory from one
    episode into the next and quietly corrupts the results.
    """
    env = LandingAviary()
    model = RecurrentPPO.load(model_path)

    successes = 0
    touchdowns = 0
    vz_list, lat_list, tilt_list, offset_list = [], [], [], []
    fail_speed, fail_accel = [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)

        lstm_states = None          # hidden state, cleared each episode
        episode_start = True

        while True:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=np.array([episode_start]),
                deterministic=True,
            )
            episode_start = False

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

    env.close()

    print("\n" + "=" * 58)
    print(f"  EVALUATION OVER {n_episodes} EPISODES  [LSTM]")
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

    print("=" * 58)
    _append_to_log(row)
    print()


def play(model_path):
    """Watch a single landing."""
    env = LandingAviary(gui=True)
    model = RecurrentPPO.load(model_path)

    obs, info = env.reset(seed=np.random.randint(0, 100000))
    lstm_states = None
    episode_start = True

    for _ in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
        action, lstm_states = model.predict(
            obs, state=lstm_states,
            episode_start=np.array([episode_start]),
            deterministic=True)
        episode_start = False

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1 / env.CTRL_FREQ)
        if terminated or truncated:
            break
    env.close()

    print(f"\n  Platform: amplitude {info.get('plat_amplitude', 0):.3f} m, "
          f"omega {info.get('plat_omega', 0):.3f} rad/s")
    print(f"  Blind fraction: {100*info.get('blind_fraction', 0):.1f}% of steps")
    if info.get("touchdown"):
        print(f"  Landed on pad : {info.get('success')}")
        print(f"  vertical speed: {info.get('touchdown_vz'):.3f} m/s")
        print(f"  lateral speed : {info.get('touchdown_lateral'):.3f} m/s")
        print(f"  tilt          : {np.degrees(info.get('touchdown_tilt')):.2f} deg")
        print(f"  offset        : {info.get('touchdown_offset'):.3f} m\n")
    else:
        print("  No touchdown — episode was truncated.\n")


def _latest_model():
    if not os.path.exists(OUTPUT_DIR):
        print(f"No {OUTPUT_DIR}/ directory. Train first.")
        sys.exit(1)
    runs = sorted(os.listdir(OUTPUT_DIR))
    if not runs:
        print("No trained models found. Run without flags first.")
        sys.exit(1)
    return os.path.join(OUTPUT_DIR, runs[-1], 'best_model.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='watch one landing')
    parser.add_argument('--eval', action='store_true', help='run the metrics table')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=2_000_000)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--note', type=str, default="",
                        help='label for this run in results_log.csv')
    args = parser.parse_args()

    if args.gui:
        play(args.model or _latest_model())
    elif args.eval:
        evaluate(args.model or _latest_model(), args.episodes, args.note)
    else:
        train(total_timesteps=args.steps)
