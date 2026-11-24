"""v7 S4p: PPO for V7 (24 Nov 2026, after step R "yes R").

The Stage 1 main method (v6_drop/scripts/train_ppo.py, run ppo2a: 1497/1500 on fresh seeds)
with the V7 world. The PPO pieces are the V6 ones, unchanged (v6_drop/learn/ppo.py:
ActorCritic with the asymmetric critic V([priv, h detached]), act, gae, chunks, clipped
loss; train_ppo.learn / save / gather): gamma 0.99, lambda 0.95, 4 epochs over 32-step
chunks, clip 0.2, KL stop 0.02, critic warm-up (>= 5 rounds, until explained variance
>= 0.5, at most 20), actor lr rising 0 -> 1e-5 over 10 rounds, noise 0.2 / 0.2 / 0.05 m/s
and 3 deg/s.
What is V7 here (nothing else changes):
  env      v7.config.make_env(student_rules=True, air_drop_bonus=True): 60 deg gimbal camera,
           S500, pad up to 9 m/s with random speed changes, waves and corners, 60 s flights
           (600 decisions; a timeout bootstraps, P3), vx vy +-10 m/s, 3 m/s^2, drop shield
           0.8 m/s, the drone's own touch detector, drop bonus only for drops from the air
           (= the Stage 1 main method's rules: ppo2a used --shield 0.8 --touch sensor
           --air-drop-bonus).
  reward   v7/envs/reward_v7.py with the step R end marks (unsafe landing -8, hard landing
           -8 - 2 (impact - 1) floor -10; all else as V6).
  noise    the same PHYSICAL widths; in policy units divided by the V7 action scale
           (+-10 m/s instead of +-6), so vx/vy noise is 0.02 instead of 0.033.
  seeds    training flights 3,000,000 + 100,000 x worker + k (plan section 5); pad mode
           "random" (the random driver). Dev exams: seeds 30000+ on the four named paths,
           --exam-n flights per path, clean (no dice: move = mean, drop if chance > 0.5),
           same rules as eval_student_v7.
  start    --init: a V7 student checkpoint (e.g. dagger_p60v_r9aux/last.pt). Only its
           "model" (the flying Student) is used; a helper head ("aux") is not trained here.
  guard    --guard G: stop if a dev exam is below G % (default 10: only a collapse stops it;
           the starting student is ~31 %).
Checkpoints: v7/results/runs/<name>/last.pt every round, r<round>.pt at each exam ("model" =
the student, so eval_student_v7 reads them as they are), metrics.csv, exam_r<round>.csv.

Run (repo root; charger in; performance mode):
  nohup python -m v7.scripts.train_ppo_v7 --init v7/results/runs/dagger_p60v_r9aux/last.pt \\
        --name ppo_p60v_s0 --workers 12 --rounds 300 > v7/results/p60v/ppo_p60v_s0.log 2>&1 &
Watch:  tail -f v7/results/p60v/ppo_p60v_s0.log | grep --line-buffered -E "^\\[(ppo|exam|guard|done)"
Stop:   pkill -f "v7.scripts.train_ppo_v7"     (last.pt is kept; --resume continues)
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import time

import numpy as np

from v6_drop.scripts import train_ppo as T6

RUNS = os.path.join("v7", "results", "runs")
ROLL_SEED0 = 3_000_000
SEED_PER_WORKER = 100_000
DEV_SEED0 = 30000
PATHS = ("random", "straight", "wave", "corners")
SIGMA = (0.2, 0.2, 0.05, 3.0)          # vx, vy, vz [m/s], yaw rate [deg/s] (V6 P4, physical)


def sigma_norm(sigma=SIGMA):
    """Noise width in the policy's units (action / V7 action scale)."""
    from v7.envs.landing_env_v7 import VXY_MAX, VZ_MAX, YAW_RATE_MAX
    scale = np.array([VXY_MAX, VXY_MAX, VZ_MAX, YAW_RATE_MAX], float)
    s = np.array([sigma[0], sigma[1], sigma[2], np.radians(sigma[3])], float)
    return s / scale


def make_env_ppo(renderer, seed=0):
    from v7.config import make_env
    return make_env(student_rules=True, air_drop_bonus=True, renderer=renderer, seed=seed)


def train_seed(w, k):
    return ROLL_SEED0 + SEED_PER_WORKER * w + k


def exam_tasks(n_per_path):
    return [(m, DEV_SEED0 + i) for m in PATHS for i in range(n_per_path)]


def fly_clean(env, student, mode, seed):
    """One clean dev flight (no dice), as eval_student_v7. Returns a row."""
    import torch
    obs, info0 = env.reset(seed=seed, options=dict(mode=mode))
    hc, n, ret, presses, blocks = None, 0, 0.0, 0, 0
    with torch.no_grad():
        while True:
            act, hc, _ = student.act(obs, hc, "cpu")
            obs, r, term, trunc, info = env.step(act)
            ret += r
            n += 1
            presses += int(info.get("drop_requested", False))
            blocks += int(bool(info.get("shield_veto")))
            if term or trunc:
                break
    td = info.get("touchdown") or {}
    return dict(mode=mode, seed=int(seed), outcome=info["outcome"], reason=td.get("reason"),
                impact=None if td.get("impact") is None else round(float(td["impact"]), 3),
                touched_before_cut=td.get("touched_before_cut"), t=n, ret=round(ret, 3),
                presses=presses, blocks=blocks, pad_v_max=round(float(info["pad"]["v_max"]), 3))


# ------------------------------------------------------------------ worker process
def worker(w, cmd_q, res_q, renderer, sigma, run_seed):
    import torch
    torch.set_num_threads(1)
    from v6_drop.learn import ppo as PPO
    model = PPO.ActorCritic(sigma_n=sigma_norm(sigma)).eval()
    env = make_env_ppo(renderer)
    rng = np.random.default_rng([int(run_seed), w, 11])
    ep_count, need_reset, obs, hc = 0, True, None, None
    ep = {}
    try:
        while True:
            msg = cmd_q.get()
            if msg[0] == "stop":
                break
            if msg[0] == "exam":
                _, sd, tasks = msg
                model.load_state_dict(sd)
                rows = [fly_clean(env, model.student, m, s) for m, s in tasks]
                need_reset = True                       # the training flight was interrupted
                res_q.put(("exam", w, rows))
                continue
            _, sd, T = msg                              # "collect"
            model.load_state_dict(sd)
            keys = ("image", "state", "priv", "move_raw", "drop", "logp", "value", "start",
                    "h", "c", "reward", "terminated", "truncated", "boot", "p_drop")
            buf = {k: [] for k in keys}
            flights, n_veto = [], 0
            for _ in range(T):
                start = 0
                if need_reset:
                    seed = train_seed(w, ep_count)
                    ep_count += 1
                    obs, info0 = env.reset(seed=seed, options=dict(mode="random"))
                    hc = model.zero_hc()
                    ep = dict(seed=seed, pad_speed=round(float(info0["pad_speed"]), 3), ret=0.0, steps=0)
                    need_reset, start = False, 1
                buf["h"].append(hc[0][0, 0].numpy().copy())
                buf["c"].append(hc[1][0, 0].numpy().copy())
                action, rec, hc = PPO.act(model, obs, hc, rng)
                for k in ("move_raw", "drop", "logp", "value", "p_drop"):
                    buf[k].append(rec[k])
                for k in ("image", "state", "priv"):
                    buf[k].append(obs[k])
                buf["start"].append(start)
                obs, r, term, trunc, info = env.step(action)
                n_veto += int(bool(info.get("shield_veto")))
                trunc = bool(trunc and not term)
                buf["reward"].append(r)
                buf["terminated"].append(bool(term))
                buf["truncated"].append(trunc)
                buf["boot"].append(PPO.value_of(model, obs, hc) if trunc else 0.0)
                ep["ret"] += r
                ep["steps"] += 1
                if term or trunc:
                    td = info.get("touchdown") or {}
                    flights.append(dict(ep, ret=round(ep["ret"], 3), outcome=info["outcome"],
                                        reason=td.get("reason"),
                                        impact=None if td.get("impact") is None else round(td["impact"], 3),
                                        touched_before_cut=td.get("touched_before_cut")))
                    need_reset = True
            last_value = 0.0 if need_reset else PPO.value_of(model, obs, hc)
            arrays = {k: np.asarray(v) for k, v in buf.items()}
            arrays["image"] = arrays["image"].astype(np.uint8)
            res_q.put(("roll", w, arrays, float(last_value), flights, n_veto))
    finally:
        env.close()


# ------------------------------------------------------------------ learner
def run_exam(cmd_qs, res_q, model, n_per_path):
    sd = T6.cpu_state(model)
    tasks = exam_tasks(n_per_path)
    W = len(cmd_qs)
    for w, q in enumerate(cmd_qs):
        q.put(("exam", sd, tasks[w::W]))
    rows = [r for (rs,) in T6.gather(res_q, W, "exam") for r in rs]
    return sorted(rows, key=lambda r: (PATHS.index(r["mode"]), r["seed"]))


def exam_line(rows, rnd):
    ok = [r for r in rows if r["outcome"] == "success"]
    outs = {}
    for r in rows:
        outs[r["outcome"]] = outs.get(r["outcome"], 0) + 1
    per = " ".join(f"{m} {sum(r['outcome'] == 'success' for r in rows if r['mode'] == m)}/"
                   f"{sum(r['mode'] == m for r in rows)}" for m in PATHS)
    line = (f"[exam] round {rnd}: safe {len(ok)}/{len(rows)} ({per}); by drop "
            f"{sum(r['reason'] == 'drop' for r in ok)}, detector {sum(r['reason'] == 'touch_detect' for r in ok)}; "
            f"outcomes {outs}; return {np.mean([r['ret'] for r in rows]):.2f}")
    return line, len(ok)


def main(argv=None):
    import torch
    from v6_drop.learn import ppo as PPO
    from v7.envs.landing_env_v7 import VXY_MAX, VZ_MAX, YAW_RATE_MAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", help="V7 student checkpoint to start the actor from")
    ap.add_argument("--resume", help="a PPO last.pt to continue")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default=RUNS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--steps", type=int, default=256, help="steps per worker per round")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--mb-chunks", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl-stop", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr-actor", type=float, default=1e-5)
    ap.add_argument("--lr-critic", type=float, default=3e-4)
    ap.add_argument("--actor-warmup", type=int, default=10)
    ap.add_argument("--critic-min", type=int, default=5)
    ap.add_argument("--critic-max", type=int, default=20)
    ap.add_argument("--ev-target", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=300, help="total rounds incl. the critic warm-up")
    ap.add_argument("--exam-every", type=int, default=10)
    ap.add_argument("--exam-n", type=int, default=25, help="dev exam flights PER PATH (4 paths)")
    ap.add_argument("--guard", type=float, default=10.0, help="stop if a dev exam is below this %")
    ap.add_argument("--sigma", type=float, nargs=4, default=list(SIGMA))
    ap.add_argument("--renderer", choices=["egl", "tiny"], default="egl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    assert (a.init is not None) + (a.resume is not None) == 1, "give exactly one of --init / --resume"
    out = os.path.join(a.out_root, a.name)
    if a.init and os.path.exists(os.path.join(out, "last.pt")):
        raise SystemExit(f"{out} already has last.pt: choose another --name (or --resume)")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    act_scale = np.array([VXY_MAX, VXY_MAX, VZ_MAX, YAW_RATE_MAX])

    model = PPO.ActorCritic(sigma_n=sigma_norm(a.sigma)).to(device)
    opt_a = torch.optim.Adam(model.actor_parameters(), lr=0.0)
    opt_c = torch.optim.Adam(model.critic.parameters(), lr=a.lr_critic)
    start, steps, ppo_start, ev_last = 0, 0, None, float("nan")
    if a.init:
        ck = torch.load(a.init, map_location=device, weights_only=False)
        model.student.load_state_dict(ck["model"])
        print(f"[ppo] actor from {os.path.abspath(a.init)}"
              + (" (its helper head is not used)" if "aux" in ck else "") + "; critic NEW", flush=True)
    else:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.student.load_state_dict(ck["model"])
        model.log_std.data.copy_(ck["log_std"].to(device))
        model.critic.load_state_dict(ck["critic"])
        opt_a.load_state_dict(ck["opt_actor"])
        opt_c.load_state_dict(ck["opt_critic"])
        start, steps, ppo_start, ev_last = ck["round"] + 1, ck["steps"], ck["ppo_start"], ck["ev_last"]
        print(f"[ppo] resumed {os.path.abspath(a.resume)} at round {start}", flush=True)
    print(f"[ppo] V7: 60 deg camera, shield 0.8, touch sensor, air-drop bonus, step R end marks; "
          f"gamma {a.gamma}; noise {a.sigma} (policy units {np.round(sigma_norm(a.sigma), 4).tolist()}); "
          f"{a.workers} workers x {a.steps} steps; exam {4 * a.exam_n} dev flights every {a.exam_every} rounds; "
          f"guard {a.guard:g} %", flush=True)
    cfg = dict(vars(a), version="v7", training_seed0=ROLL_SEED0, dev_seed0=DEV_SEED0)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)

    ctx = mp.get_context("spawn")
    cmd_qs = [ctx.Queue() for _ in range(a.workers)]
    res_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(w, cmd_qs[w], res_q, a.renderer, a.sigma, a.seed), daemon=True)
             for w in range(a.workers)]
    for p in procs:
        p.start()
    metrics_path = os.path.join(out, "metrics.csv")
    new_csv = not os.path.exists(metrics_path)
    mf = open(metrics_path, "a", newline="")
    mw = None
    try:
        for rnd in range(start, a.rounds):
            in_warmup = ppo_start is None and (rnd < a.critic_min or
                                              (not ev_last >= a.ev_target and rnd < a.critic_max))
            if ppo_start is None and not in_warmup:
                ppo_start = rnd
                print(f"[ppo] critic warm-up done (explained variance {ev_last:.2f}); PPO from round {rnd}",
                      flush=True)
            k = 0 if ppo_start is None else rnd - ppo_start
            lr_a = 0.0 if ppo_start is None else a.lr_actor * min(1.0, (k + 1) / a.actor_warmup)
            t0 = time.time()
            sd = T6.cpu_state(model)
            for q in cmd_qs:
                q.put(("collect", sd, a.steps))
            res = T6.gather(res_q, a.workers, "roll")
            t1 = time.time()
            roll = {key: np.stack([r[0][key] for r in res]) for key in res[0][0]}
            flights = [f for r in res for f in r[2]]
            adv, tgt = zip(*[PPO.gae(r[0]["reward"], r[0]["value"], r[0]["terminated"], r[0]["truncated"],
                                     r[0]["boot"], r[1], a.gamma, a.lam) for r in res])
            adv, tgt = np.stack(adv), np.stack(tgt)
            ev = PPO.explained_variance(roll["value"].ravel(), tgt.ravel())
            roll["adv"] = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)
            roll["target"] = tgt.astype(np.float32)
            chunks = PPO.make_chunks(roll, a.chunk)
            model.train()
            m = T6.learn(model, opt_a, opt_c, chunks, a, device, train_actor=ppo_start is not None, lr_a=lr_a)
            model.eval()
            ev_last = ev
            n_dec = a.workers * a.steps
            steps += n_dec
            t2 = time.time()
            nf = len(flights)
            safe = sum(f["outcome"] == "success" for f in flights)
            drops = sum(f["outcome"] == "success" and f["reason"] == "drop" for f in flights)
            std = np.exp(model.clamped_log_std().detach().cpu().numpy()) * act_scale
            outs = {}
            for f in flights:
                outs[f["outcome"]] = outs.get(f["outcome"], 0) + 1
            row = dict(round=rnd, phase="ppo" if ppo_start is not None else "critic", steps=steps,
                       flights=nf, safe_pct=round(100 * safe / nf, 1) if nf else None,
                       drop_pct=round(100 * drops / nf, 1) if nf else None,
                       hard=outs.get("hard_landing", 0),
                       unsafe=outs.get("slide_off", 0) + outs.get("tip_over", 0) + outs.get("not_still", 0),
                       lost=outs.get("lost", 0), timeout=outs.get("timeout", 0),
                       vetoes=sum(r[3] for r in res),
                       ret=round(float(np.mean([f["ret"] for f in flights])), 2) if nf else None,
                       ev=round(ev, 3), v_loss=round(m["v_loss"], 4),
                       pg_loss=round(m.get("pg_loss", 0.0), 4), kl=round(m.get("kl", 0.0), 5),
                       clipfrac=round(m.get("clipfrac", 0.0), 3), epochs=m["epochs"], kl_stop=m["kl_stop"],
                       std_vx=round(float(std[0]), 3), std_vz=round(float(std[2]), 3),
                       std_yaw_deg=round(float(np.degrees(std[3])), 2), lr_actor=lr_a,
                       t_collect=round(t1 - t0, 1), t_learn=round(t2 - t1, 1),
                       dps=round(n_dec / (t1 - t0), 0))
            if mw is None:
                mw = csv.DictWriter(mf, fieldnames=list(row))
                if new_csv:
                    mw.writeheader()
            mw.writerow(row)
            mf.flush()
            print(f"[ppo] r {rnd} {row['phase']} | steps {steps:,} | flights {nf} safe {row['safe_pct']}% "
                  f"drop {row['drop_pct']}% unsafe {row['unsafe']} hard {row['hard']} lost {row['lost']} "
                  f"timeout {row['timeout']} veto {row['vetoes']} | ret {row['ret']} | EV {row['ev']:.2f} "
                  f"Vloss {row['v_loss']:.3f} | KL {row['kl']:.4f} clip {row['clipfrac']:.2f} ep {row['epochs']} | "
                  f"std vx {row['std_vx']:.3f} vz {row['std_vz']:.3f} | lr {lr_a:.1e} | "
                  f"{row['t_collect']:.0f}+{row['t_learn']:.0f} s, {row['dps']:.0f} dec/s", flush=True)
            T6.save(os.path.join(out, "last.pt"), model, opt_a, opt_c, rnd, steps, ppo_start, ev_last, cfg)
            last_round = rnd == a.rounds - 1
            if ppo_start is not None and ((k + 1) % a.exam_every == 0 or last_round):
                rows = run_exam(cmd_qs, res_q, model, a.exam_n)
                line, k_safe = exam_line(rows, rnd)
                print(line, flush=True)
                tag = f"r{rnd:04d}"
                T6.save(os.path.join(out, tag + ".pt"), model, opt_a, opt_c, rnd, steps, ppo_start, ev_last, cfg)
                with open(os.path.join(out, f"exam_{tag}.csv"), "w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                    wr.writeheader()
                    wr.writerows(rows)
                if 100.0 * k_safe / len(rows) < a.guard:
                    print(f"[guard] safe {k_safe}/{len(rows)} < {a.guard:g}%: STOP", flush=True)
                    break
        print(f"[done] {steps:,} steps; checkpoints in {os.path.abspath(out)}", flush=True)
    finally:
        mf.close()
        for q in cmd_qs:
            q.put(("stop",))
        for p in procs:
            p.join(timeout=30)


if __name__ == "__main__":
    main()
