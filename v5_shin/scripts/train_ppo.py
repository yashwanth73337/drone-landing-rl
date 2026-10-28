"""V5 training entry point (Block 12): recurrent PPO + L_est, curriculum, logging, checkpoints.

Usage (from the repo root). DO NOT start a run without the supervisor/owner's go-ahead:
  python -m v5_shin.scripts.train_ppo --mode privileged --name P_smoke --total-steps 200000
  python -m v5_shin.scripts.train_ppo --mode vision --name A_s1 --seed 1 --total-steps 20000000
  --motor {asym,sym_slow,sym_fast}: motor time constants (default asym = AerialGym LMF2, the
  paper's setting). sym_* are DIAGNOSTIC ablations of the actuator sink (SPEC §15, 1 Oct).
  --reward {paper,v1,...,v6}: paper = Table III (default). v1-v6 = touchdown-aware reward
  (SPEC D21, D22, D27, D30) and they switch on the post-touchdown hold (SPEC D20); success = safe landing.
  --hold-practice P (D22, needs v1-v4): share of training episodes that start resting on the pad
  in the hold phase. These are logged (episodes.csv column 'practice', updates.csv n_practice*)
  but never counted for the curriculum or the success / outcome columns. Default 0 = off.
  --hold-vz-down V (D23, needs v1-v4): in the hold, a negative vertical action is scaled to V m/s
  (instead of --vz-max), so the drone can cut thrust and sit on the pad. Default: off.
  --pad-smooth (D23): pad speed changes spread over each 0.1 s step (same random walk). Default off.
  --action-lowpass ALPHA (D25): low-pass filter on the vertical action (0 < ALPHA <= 1). Default off.
  --log-std-max-xy L (D26): cap on the vx, vy exploration log-std (applied at start, after loading a
  checkpoint and after every optimizer step). Default off.
  --lr-restart (D28, with --resume): the linear LR schedule starts at --lr at the resumed update and
  falls to 0 at --total-steps (default: one schedule over the whole run history, as before).
  --resume from a distill.py checkpoint (D32): it has no curriculum state, so the run starts at
  --start-level, and the update counter is set to global_step // (n_envs * n_steps).
  --lr-warmup N (D29): the LR is also multiplied by (k + 1) / N for the first N updates k of the
  schedule (k = 0 at the schedule start). Default 0 = off.

Outputs in v5_shin/runs/<name>/:
  config.json            every argument + versions (tracked)
  updates.csv            one row per PPO update (tracked)
  episodes.csv           one row per finished episode (tracked)
  curriculum.csv         one row per closed 512-episode window (tracked)
  latest.pt, u<N>.pt, level_<L>.pt   checkpoints (git-ignored)
"""
import argparse
import csv
import json
import os
import platform as pyplatform
import time

import gymnasium as gym
import numpy as np
import torch

from v5_shin.envs.curriculum import Curriculum
from v5_shin.envs.dr import DRConfig
from v5_shin.envs import landing_sim as LS
from v5_shin.envs.reward import ShinReward, TouchdownReward
from v5_shin.envs.shin_env import VZ_MAX_DEFAULT, ShinLandingEnv
from v5_shin.policies.active_perception import ActivePerceptionReward
from v5_shin.policies.ppo import PPOConfig, PPOTrainer

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")


def make_env_fn(mode, seed, renderer, vz_penalty, c, dr_off=False, vz_max=VZ_MAX_DEFAULT,
                motor_mode="asym", reward="paper", hold_practice=0.0, hold_vz_down=None,
                pad_smooth=False, action_lowpass=None):
    hold = reward != "paper"                     # D20 hold goes with the D21 reward
    def f():
        rf = TouchdownReward(reward, vz_penalty) if hold else ShinReward(vz_penalty)
        return ShinLandingEnv(mode=mode, renderer=renderer, egl=(renderer == "egl"), seed=seed, c=c,
                              reward_fn=rf, vz_max=vz_max,
                              dr=DRConfig.off() if dr_off else DRConfig(), motor_mode=motor_mode,
                              hold_steps=(LS.HOLD_STEPS if hold else 0),
                              hold_practice=(hold_practice if hold else 0.0),
                              hold_vz_down=(hold_vz_down if hold else None),
                              pad_smooth=pad_smooth, action_lowpass=action_lowpass)
    return f


def make_venv(mode, n_envs, seed, renderer, vz_penalty, c, sync=False, dr_off=False,
              vz_max=VZ_MAX_DEFAULT, motor_mode="asym", reward="paper", hold_practice=0.0,
              hold_vz_down=None, pad_smooth=False, action_lowpass=None):
    fns = [make_env_fn(mode, seed * 1000 + i, renderer, vz_penalty, c, dr_off, vz_max, motor_mode,
                       reward, hold_practice, hold_vz_down, pad_smooth, action_lowpass)
           for i in range(n_envs)]
    kw = dict(autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    if sync:
        return gym.vector.SyncVectorEnv(fns, **kw)
    return gym.vector.AsyncVectorEnv(fns, context="spawn", **kw)


class CSVLog:
    def __init__(self, path):
        self.path, self.f, self.w = path, None, None

    def write(self, row):
        if self.w is None:
            new = not os.path.exists(self.path)
            self.f = open(self.path, "a", newline="")
            self.w = csv.DictWriter(self.f, fieldnames=list(row))
            if new:
                self.w.writeheader()
        self.w.writerow({k: row.get(k) for k in self.w.fieldnames})
        self.f.flush()


def episode_row(e, update, level, hold=False):
    rv = e.get("rel_vel")
    row = dict(update=update, level=level, outcome=e["outcome"], steps=e["t"],
               c_episode=e["c_episode"], com_over_pad=e.get("com_over_pad"),
               impact_vz=(float(rv[2]) if rv is not None else None),
               impact_speed=(float(np.linalg.norm(rv)) if rv is not None else None),
               tilt_deg=e.get("tilt_deg"))
    if hold:                                     # extra columns only for D20 runs
        row["touchdown"] = bool(e.get("touchdown"))
        row["practice"] = bool(e.get("practice"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vision", "privileged"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--total-steps", type=int, required=True)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--micro-chunks", type=int, default=None,
                    help="chunks per micro-batch (grad accumulation); vision on 4 GB: 8")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-warmup", type=int, default=0,
                    help="D29: linear LR warm-up over this many updates (default 0 = off)")
    ap.add_argument("--lr-restart", action="store_true",
                    help="D28: with --resume, LR decays from --lr to 0 over this run only")
    ap.add_argument("--log-std-max-xy", type=float, default=None,
                    help="D26: cap on the vx, vy exploration log-std (default off)")
    ap.add_argument("--lambda-est", type=float, default=1.0, help="0 = w/o state estimation")
    ap.add_argument("--vz-penalty", choices=["literal", "prose"], default="literal")
    ap.add_argument("--r-active", choices=["auto", "off"], default="auto",
                    help="active-perception reward (Block 13): auto = on for vision with "
                         "lambda_est > 0 (proposed method); off = A-noAP. Never used by variant P")
    ap.add_argument("--vz-max", type=float, default=VZ_MAX_DEFAULT,
                    help="vertical-speed action limit (m/s); D16 default 1.0 (P_smoke_s1 used 3.0)")
    ap.add_argument("--motor", choices=["asym", "sym_slow", "sym_fast"], default="asym",
                    help="motor time constants; asym = AerialGym LMF2 (default); sym_* = diagnostic")
    ap.add_argument("--reward", choices=["paper", "v1", "v2", "v3", "v4", "v5", "v6"], default="paper",
                    help="paper = Table III; v1-v4 = touchdown-aware (D21, D22) with the hold (D20)")
    ap.add_argument("--hold-practice", type=float, default=0.0,
                    help="D22: share of training episodes starting on the pad (needs v1-v4)")
    ap.add_argument("--hold-vz-down", type=float, default=None,
                    help="D23: downward vertical limit (m/s) in the hold (needs v1-v4)")
    ap.add_argument("--pad-smooth", action="store_true",
                    help="D23: spread pad speed changes over each 0.1 s step")
    ap.add_argument("--action-lowpass", type=float, default=None,
                    help="D25: vertical action low-pass alpha in (0, 1] (default off)")
    ap.add_argument("--renderer", choices=["egl", "tiny"], default="egl")
    ap.add_argument("--start-level", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=50, help="updates")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/latest.pt")
    a = ap.parse_args()
    hold = a.reward != "paper"
    if not 0.0 <= a.hold_practice <= 1.0 or (a.hold_practice > 0 and not hold):
        ap.error("--hold-practice must be in [0, 1] and needs --reward v1-v6")
    if a.hold_vz_down is not None and (not hold or a.hold_vz_down < a.vz_max):
        ap.error("--hold-vz-down needs --reward v1-v6 and must be >= --vz-max")
    if a.action_lowpass is not None and not 0.0 < a.action_lowpass <= 1.0:
        ap.error("--action-lowpass must be in (0, 1]")

    run = os.path.join(RUNS, a.name)
    os.makedirs(run, exist_ok=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    per_update = a.n_envs * a.n_steps
    total_updates = int(np.ceil(a.total_steps / per_update))
    micro = a.micro_chunks if a.micro_chunks else (8 if (a.mode == "vision" and a.device != "cpu") else None)
    cfg = PPOConfig(mode=a.mode, n_steps=a.n_steps, seq_len=a.seq_len, epochs=a.epochs,
                    minibatches=a.minibatches, micro_chunks=micro, lr=a.lr,
                    lambda_est=(a.lambda_est if a.mode == "vision" else 0.0),
                    log_std_max_xy=a.log_std_max_xy, lr_warmup=a.lr_warmup)
    cur = Curriculum(start_level=a.start_level)
    venv = make_venv(a.mode, a.n_envs, a.seed, a.renderer, a.vz_penalty, cur.c, vz_max=a.vz_max,
                     motor_mode=a.motor, reward=a.reward, hold_practice=a.hold_practice,
                     hold_vz_down=a.hold_vz_down, pad_smooth=a.pad_smooth,
                     action_lowpass=a.action_lowpass)
    use_ra = a.r_active == "auto" and a.mode == "vision" and cfg.lambda_est > 0
    tr = PPOTrainer(venv, cfg, device=a.device, total_updates=total_updates,
                    r_active_fn=ActivePerceptionReward() if use_ra else None)
    if a.resume:
        d = torch.load(os.path.join(run, "latest.pt"), map_location=a.device, weights_only=False)
        extra = tr.load_state_dict(d)
        tr.clamp_log_std()                       # D26 (no-op unless --log-std-max-xy)
        if "curriculum" in extra:
            cur = Curriculum.from_state_dict(extra["curriculum"])
        else:                                    # D32: checkpoint from distill.py
            cur = Curriculum(start_level=a.start_level)
            tr.update = tr.global_step // per_update   # align the PPO update counter with the steps
            print(f"[D32] checkpoint without curriculum (distill.py): level {cur.level}, "
                  f"update set to {tr.update} from global_step {tr.global_step}", flush=True)
        if a.lr_restart:                         # D28
            tr.cfg.lr_schedule_start = tr.update
        venv.call("set_c", cur.c)
    if not a.resume or not os.path.exists(os.path.join(run, "config.json")):
        with open(os.path.join(run, "config.json"), "w") as f:
            json.dump(dict(vars(a), micro_chunks_used=micro, total_updates=total_updates,
                           r_active_used=use_ra, hold_steps=(LS.HOLD_STEPS if hold else 0),
                           ppo=vars(cfg), host=pyplatform.node(), torch=torch.__version__,
                           gymnasium=gym.__version__,
                           gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)),
                      f, indent=2)
    upd_log, ep_log, cur_log = (CSVLog(os.path.join(run, n)) for n in
                                ("updates.csv", "episodes.csv", "curriculum.csv"))
    tr.reset(seeds=[a.seed * 100000 + tr.update * 1000 + i for i in range(a.n_envs)])
    n_hist = len(cur.history)
    t0 = time.time()
    try:
        while tr.update < total_updates:
            t_roll = time.time()
            buf, eps_all = tr.collect()
            t_roll = time.time() - t_roll
            level_before = cur.level
            for e in eps_all:
                ep_log.write(episode_row(e, tr.update, cur.level, hold))
                if e.get("practice"):                # D22: never counted for the curriculum
                    continue
                if cur.record(e["outcome"] == "success", e["c_episode"]):
                    venv.call("set_c", cur.c)
            prac = [e for e in eps_all if e.get("practice")]
            eps = [e for e in eps_all if not e.get("practice")]
            for h in cur.history[n_hist:]:
                cur_log.write(h)
            n_hist = len(cur.history)
            t_learn = time.time()
            stats = tr.learn(buf)
            t_learn = time.time() - t_learn
            outc = [e["outcome"] for e in eps]
            succ = [e for e in eps if e["outcome"] == "success"]
            hold_cols = {}
            if hold:
                td = [e for e in eps if e.get("touchdown")]
                hold_cols = {f"n_{k}": outc.count(k) for k in LS.HOLD_OUTCOMES}
                hold_cols.update(n_touchdown=len(td), td_impact_vz_mean=(
                    float(np.mean([e["rel_vel"][2] for e in td])) if td else None),
                    n_practice=len(prac),
                    n_practice_success=sum(e["outcome"] == "success" for e in prac))
            row = dict(update=tr.update, global_step=tr.global_step, wall_s=round(time.time() - t0, 1),
                       sps=round(per_update / max(t_roll + t_learn, 1e-9), 1),
                       rollout_s=round(t_roll, 2), learn_s=round(t_learn, 2),
                       level=cur.level, c=cur.c, window_n=cur.n,
                       window_success=(cur.k / cur.n if cur.n else None), episodes=len(eps),
                       **{f"n_{k}": outc.count(k) for k in ("success", "crash_ground", "crash_platform",
                                                             "tilt", "drift", "timeout")},
                       strict_success=sum(bool(e.get("com_over_pad")) for e in succ),
                       impact_vz_mean=(float(np.mean([e["rel_vel"][2] for e in succ])) if succ else None),
                       ep_len_mean=(float(np.mean([e["t"] for e in eps])) if eps else None),
                       reward_mean=float(buf["rewards"].mean()),
                       r_active_mean=float(buf["r_active"].mean()),
                       r_active_frac=buf.get("r_active_frac"),
                       **hold_cols, **stats)
            if a.mode == "vision":
                row.update(tr.estimate_error_by_blind_age(buf))
            upd_log.write(row)
            print(f"u{tr.update:5d} step {tr.global_step:>10d} L{cur.level} win {cur.k}/{cur.n} "
                  f"eps {len(eps):3d} succ {outc.count('success'):3d} "
                  f"crash {outc.count('crash_ground') + outc.count('crash_platform'):3d} "
                  f"tilt {outc.count('tilt'):2d} drift {outc.count('drift'):3d} "
                  f"to {outc.count('timeout'):3d} | kl {stats['kl']:.4f} ev {stats['explained_var']:.2f} "
                  f"est {stats['est']:.3f}"
                  + (f" ra {row['r_active_frac']:.2f}/{row['r_active_mean']:+.4f}"
                     if row["r_active_frac"] is not None else "")
                  + (f" | td {row['n_touchdown']:3d} hard {row['n_hard_landing']:3d} "
                     f"hfail {sum(row['n_' + k] for k in ('bounce', 'slide_off', 'tip_over', 'not_still')):3d}"
                     if hold else "")
                  + (f" | pr {row['n_practice_success']:3d}/{row['n_practice']:3d}"
                     if a.hold_practice > 0 else "")
                  + f" | {row['sps']:.0f} sps", flush=True)
            extra = dict(curriculum=cur.state_dict())
            if tr.update % a.ckpt_every == 0 or tr.update == total_updates:
                torch.save(tr.state_dict(extra), os.path.join(run, f"u{tr.update:05d}.pt"))
            torch.save(tr.state_dict(extra), os.path.join(run, "latest.pt"))
            if cur.level != level_before:            # first checkpoint at each new level
                torch.save(tr.state_dict(extra), os.path.join(run, f"level_{cur.level:02d}.pt"))
    finally:
        venv.close()


if __name__ == "__main__":
    main()
