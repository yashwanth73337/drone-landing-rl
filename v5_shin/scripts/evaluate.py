"""Pinned evaluation of a trained checkpoint (V5 Block 14; SPEC §13).

Episode i uses seed seed_base + i (spawn, platform walk, DR, sensor noise AND controller
gains are re-drawn per episode), so results do not depend on the number of workers.
The env configuration (mode, vz_max, reward variant) is read from the run's config.json.
Actions are the policy MEAN by default (--stochastic to sample).

Reports (summary.json + episodes.csv):
  success rate with Wilson 95% CI, strict success (CoM over pad), outcome counts,
  impact v_z / speed (pre-contact), steps to land,
  safe success: fraction of ALL episodes that succeed with |impact v_z| <= 0.5/1.0/1.5/2.0 m/s
               (and the strict version),
  commanded v_z: last step and mean of the last 5 steps (per episode; median over successes),
  vision: estimate RMSE (pos m, vel m/s) over ALL frames (Table IV style) and the mean
          position error per blind-age bin (visible, 1-15, 16-60, 61-120, >120 steps).

Motor time constants (--motor): asym (AerialGym LMF2) | sym_slow (spin-down tau := spin-up tau) |
  sym_fast (both 5 ms). Default: the motor the run was TRAINED with (config.json "motor"; runs
  without the key were trained with asym). Only the time constants change; every RNG stream and
  pinned episode is the same. The output folder gets a _<motor> suffix unless motor is asym.

True-state injection (--inject-true-state, vision only, evaluation diagnostic): the decision
  layer gets the TRUE relative state in place of the estimate y[0:6]; everything else (image,
  LSTM, the other 250 latent dims) is unchanged. Estimate metrics still report the estimator.
  Output folder suffix: _injtrue.

Usage:
  python -m v5_shin.scripts.evaluate --run P_smoke_s1 --ckpt latest.pt --episodes 1000 --c 1.0
  python -m v5_shin.scripts.evaluate --run P_smoke_vz1_s1 --c 1.0 --motor sym_slow
Output: v5_shin/runs/<run>/eval_<ckpt>_c<c>_s<seed_base>_<det|sto>[_<motor>]/
        (the suffix is omitted for 'asym', so existing result folders keep their names)
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
BINS = ((0, 0), (1, 15), (16, 60), (61, 120), (121, 10 ** 9))
SAFE_VZ = (0.5, 1.0, 1.5, 2.0)          # m/s, |impact v_z| thresholds for safe success
MOTOR_MODES = ("asym", "sym_slow", "sym_fast")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _worker(args):
    ckpt_path, cfg, seeds, c, stochastic, renderer, motor, inject = args
    from v5_shin.envs.reward import ShinReward
    from v5_shin.envs.shin_env import ShinLandingEnv
    from v5_shin.policies.shin_policy import ShinPolicy
    torch.set_num_threads(1)
    mode = cfg["mode"]
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pol = ShinPolicy(mode)
    pol.load_state_dict(d["policy"])
    pol.eval()
    env = ShinLandingEnv(mode=mode, renderer=renderer, egl=(renderer == "egl"), seed=0, c=c,
                         reward_fn=ShinReward(cfg.get("vz_penalty", "literal")),
                         vz_max=cfg.get("vz_max", 3.0), motor_mode=motor)
    keys = ("image", "u", "critic", "target", "s_rel") if mode == "vision" else ("u", "critic", "target", "s_rel")
    rows, frames = [], []
    try:
        for seed in seeds:
            env.sim.reseed(seed)                          # gains + all streams from this seed
            obs, info = env.reset(options={"c": c})
            state = pol.initial_state(1)
            start = torch.ones(1)
            age = 0 if info["pad_centre_in_view"] else 1
            ret, k = 0.0, 0
            cmd_vz = []
            g = torch.Generator().manual_seed(seed)
            while True:
                tob = {kk: torch.as_tensor(np.asarray(obs[kk]))[None] for kk in keys}
                with torch.no_grad():
                    mu, s_est, state = pol.actor_seq({kk: v[None] for kk, v in tob.items()}, state,
                                                     start[None], inject_true_state=inject)
                    mu = mu[0, 0]
                    if stochastic:
                        a = mu + pol.log_std.exp() * torch.randn(mu.shape, generator=g)
                    else:
                        a = mu
                if s_est is not None:
                    e = s_est[0, 0].numpy() - obs["target"]
                    frames.append((age, float(np.linalg.norm(e[:3])), e[:3] ** 2, e[3:] ** 2))
                obs, r, te, tr, info = env.step(a.numpy())
                cmd_vz.append(float(info["action_cmd"][2]))
                ret += r
                k += 1
                start = torch.zeros(1)
                age = 0 if info["pad_centre_in_view"] else age + 1
                if te or tr:
                    break
            sp = env.sim.spawn
            rv = info.get("rel_vel", np.full(3, np.nan))
            rp = info.get("rel_pos_pad", np.full(2, np.nan))
            rows.append(dict(seed=seed, outcome=info["outcome"], steps=k, ret=ret,
                             com_over_pad=bool(info.get("com_over_pad", False)),
                             impact_vz=float(rv[2]), impact_speed=float(np.linalg.norm(rv)),
                             touch_x_pad=float(rp[0]), touch_y_pad=float(rp[1]),
                             tilt_deg=float(info.get("tilt_deg", np.nan)),
                             cmd_vz_last=cmd_vz[-1], cmd_vz_last5=float(np.mean(cmd_vz[-5:])),
                             dz0=float(sp["dxyz0"][2]), d0=float(np.hypot(*sp["dxyz0"][:2])),
                             v_plat0=float(env.sim.plat.v)))
    finally:
        env.close()
    return rows, frames


def evaluate(run, ckpt, episodes=1000, c=1.0, seed_base=9000, workers=8, stochastic=False,
             renderer="tiny", out=None, motor=None, inject_true_state=False):
    run_dir = os.path.join(RUNS, run)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    trained_motor = cfg.get("motor", "asym")
    motor = motor or trained_motor
    if motor not in MOTOR_MODES:
        raise ValueError(f"motor {motor!r} not in {MOTOR_MODES}")
    if inject_true_state and cfg["mode"] != "vision":
        raise ValueError("--inject-true-state needs a vision run (variant P already sees the true state)")
    seeds = [seed_base + i for i in range(episodes)]
    chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
    args = [(os.path.join(run_dir, ckpt), cfg, ch, c, stochastic, renderer, motor, inject_true_state)
            for ch in chunks]
    t0 = time.time()
    if len(args) == 1:
        res = [_worker(args[0])]
    else:
        with mp.get_context("spawn").Pool(len(args)) as pool:
            res = pool.map(_worker, args)
    rows = sorted([r for rr, _ in res for r in rr], key=lambda r: r["seed"])
    frames = [f for _, ff in res for f in ff]
    n = len(rows)
    oc = {k: sum(r["outcome"] == k for r in rows) for k in
          ("success", "crash_ground", "crash_platform", "tilt", "drift", "timeout")}
    succ = [r for r in rows if r["outcome"] == "success"]
    strict = sum(r["com_over_pad"] for r in succ)
    vz = np.array([r["impact_vz"] for r in succ]) if succ else np.array([np.nan])
    summ = dict(run=run, ckpt=ckpt, timestep=None, c=c, episodes=n, seed_base=seed_base,
                policy="stochastic" if stochastic else "deterministic(mean)", renderer=renderer,
                vz_max=cfg.get("vz_max", 3.0), mode=cfg["mode"], motor=motor,
                trained_motor=trained_motor, inject_true_state=bool(inject_true_state),
                success_rate=oc["success"] / n, success_ci95=wilson(oc["success"], n),
                strict_success_rate=strict / n, strict_ci95=wilson(strict, n), outcomes=oc,
                impact_vz_median=float(np.median(vz)), impact_vz_p10=float(np.percentile(vz, 10)),
                impact_vz_p90=float(np.percentile(vz, 90)),
                impact_speed_median=float(np.median([r["impact_speed"] for r in succ])) if succ else None,
                steps_to_land_median=float(np.median([r["steps"] for r in succ])) if succ else None,
                safe_success={f"<={t}": sum(abs(r["impact_vz"]) <= t for r in succ) / n for t in SAFE_VZ},
                safe_strict_success={f"<={t}": sum(abs(r["impact_vz"]) <= t and r["com_over_pad"]
                                                   for r in succ) / n for t in SAFE_VZ},
                cmd_vz_last5_median=float(np.median([r["cmd_vz_last5"] for r in succ])) if succ else None,
                wall_s=round(time.time() - t0, 1))
    try:
        d = torch.load(os.path.join(run_dir, ckpt), map_location="cpu", weights_only=False)
        summ["timestep"] = int(d.get("global_step", -1))
        summ["update"] = int(d.get("update", -1))
    except Exception:
        pass
    if frames:
        ages = np.array([f[0] for f in frames])
        perr = np.array([f[1] for f in frames])
        summ["est_pos_rmse_m"] = float(np.sqrt(np.mean(np.sum([f[2] for f in frames], axis=1))))
        summ["est_vel_rmse_ms"] = float(np.sqrt(np.mean(np.sum([f[3] for f in frames], axis=1))))
        summ["est_pos_err_by_blind_age"] = {
            f"{lo}-{hi if hi < 10 ** 9 else 'inf'}": dict(
                n=int(((ages >= lo) & (ages <= hi)).sum()),
                mean_m=float(perr[(ages >= lo) & (ages <= hi)].mean()) if ((ages >= lo) & (ages <= hi)).any() else None)
            for lo, hi in BINS}
    tag = f"eval_{os.path.splitext(ckpt)[0]}_c{c}_s{seed_base}_{'sto' if stochastic else 'det'}"
    if motor != "asym":
        tag += f"_{motor}"
    if inject_true_state:
        tag += "_injtrue"
    out = out or os.path.join(run_dir, tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    return summ, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--c", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--renderer", choices=["tiny", "egl"], default="tiny")
    ap.add_argument("--inject-true-state", action="store_true",
                    help="vision only: feed the decision layer the true s_rel instead of the estimate")
    ap.add_argument("--motor", choices=MOTOR_MODES, default=None,
                    help="motor time constants (default: as trained, from config.json)")
    a = ap.parse_args()
    for c in a.c:
        s, _ = evaluate(a.run, a.ckpt, a.episodes, c, a.seed_base, a.workers, a.stochastic, a.renderer,
                        motor=a.motor, inject_true_state=a.inject_true_state)
        show = {k: s[k] for k in ("run", "ckpt", "timestep", "c", "episodes", "policy", "motor", "trained_motor", "inject_true_state",
                                  "success_rate", "success_ci95", "strict_success_rate", "outcomes",
                                  "impact_vz_median", "impact_vz_p10", "impact_vz_p90",
                                  "cmd_vz_last5_median", "safe_success", "safe_strict_success",
                                  "steps_to_land_median", "wall_s")}
        for k in ("est_pos_rmse_m", "est_vel_rmse_ms", "est_pos_err_by_blind_age"):
            if k in s:
                show[k] = s[k]
        print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
