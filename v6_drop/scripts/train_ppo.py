"""v6_drop Step 7d: PPO training (decided 30 Oct, P1-P7; taught in 7.1-7.5).

One ROUND (7.4, picture s7_4_ppo.png):
  1. COLLECT  W worker processes (one EGL simulator each, B5 rule; the actor on the CPU,
              one thread each) fly STEPS steps each with the current policy and dice.
              Flights continue across rounds; each new flight gets its own training seed
              1,000,000 + 100,000 x worker + k (never the dev seeds 9000..).
  2. GAE      gamma 0.99, lambda 0.95 (P2); landing/crash = end, timeout = bootstrap (P3).
  3. LEARN    4 epochs over 32-step chunks (memory restored from the rollout), minibatches
              of MB chunks; clipped loss (0.2), value loss, grad clip 0.5; stop the epochs
              early when approx KL > 0.02 (P6). Advantages normalised per round.
Phases (P5):
  critic warm-up  the actor is FROZEN (its eyes/LSTM run without gradients); only the
                  critic learns. At least --critic-min rounds, then until explained
                  variance >= --ev-target (at most --critic-max rounds).
  ppo             actor and critic learn. Actor: fresh Adam, lr rises linearly from 0 to
                  --lr-actor over --actor-warmup rounds (V5 lesson). Critic lr --lr-critic.
Guard (P5): every --exam-every PPO rounds a CLEAN exam (no dice) on the dev seeds 9000..
(--exam-n flights) runs in the workers; the checkpoint is saved as r<round>.pt; if the
safe count falls below --guard % the run STOPS (then: add a copy/KL anchor).

Shield (Step 7e): --shield L uses the camera-based drop shield in the workers' simulators
(training flights AND exams): a drop is carried out only if the onboard predicted impact
<= L m/s. A blocked press is part of the environment (nothing happens), so the stored
action stays the policy's own choice. 'vetoes' = blocked presses per round.

Checkpoints: results/runs/<name>/last.pt every round (+ r<round>.pt at each exam). Key
'model' = the student, so eval_student / dice_exam read them as they are.

Run (from the repo root; performance mode first):
  nohup python -m v6_drop.scripts.train_ppo --init v6_drop/results/runs/dagger_r4/last.pt \\
        --name ppo1 --workers 8 --rounds 300 > v6_drop/results/runs/ppo1.log 2>&1 &
Watch:  tail -f v6_drop/results/runs/ppo1.log | grep --line-buffered -E "^\\[(ppo|exam|guard|done)"
Stop:   pkill -f "v6_drop.scripts.train_ppo"     (last.pt is kept; --resume continues)
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import time

import numpy as np

RUNS = os.path.join(os.path.dirname(__file__), "..", "results", "runs")
ROLL_SEED0 = 1_000_000
SEED_PER_WORKER = 100_000
DEV_SEED0 = 9000


# ------------------------------------------------------------------ worker process
def worker(w, cmd_q, res_q, renderer, sigma, run_seed, shield=None):
    import torch
    torch.set_num_threads(1)
    from v6_drop.envs import landing_env as E
    from v6_drop.learn import ppo as PPO
    from v6_drop.scripts.dice_exam import fly_dice, sigma_norm
    sig_n = sigma_norm(sigma)
    model = PPO.ActorCritic(sigma_n=sig_n).eval()
    env = E.LandingEnv(renderer=renderer, seed=0, drop_shield=shield)
    rng = np.random.default_rng([int(run_seed), w, 11])
    ep_count, need_reset, obs, hc = 0, True, None, None
    ep = {}
    try:
        while True:
            msg = cmd_q.get()
            if msg[0] == "stop":
                break
            if msg[0] == "exam":
                _, sd, seeds = msg
                model.load_state_dict(sd)
                rows = [fly_dice(env, model.student, s, "clean", sig_n) for s in seeds]
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
                    seed = ROLL_SEED0 + SEED_PER_WORKER * w + ep_count
                    ep_count += 1
                    obs, info0 = env.reset(seed=seed)
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
                                        impact=None if td.get("impact") is None else round(td["impact"], 3)))
                    need_reset = True
            last_value = 0.0 if need_reset else PPO.value_of(model, obs, hc)
            arrays = {k: np.asarray(v) for k, v in buf.items()}
            arrays["image"] = arrays["image"].astype(np.uint8)
            res_q.put(("roll", w, arrays, float(last_value), flights, n_veto))
    finally:
        env.close()


# ------------------------------------------------------------------ learner
def cpu_state(model):
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


def gather(res_q, n, kind):
    out = {}
    while len(out) < n:
        msg = res_q.get()
        assert msg[0] == kind, msg[0]
        out[msg[1]] = msg[2:]
    return [out[w] for w in sorted(out)]


def run_exam(cmd_qs, res_q, model, n):
    sd = cpu_state(model)
    seeds = list(range(DEV_SEED0, DEV_SEED0 + n))
    W = len(cmd_qs)
    for w, q in enumerate(cmd_qs):
        q.put(("exam", sd, seeds[w::W]))
    rows = sorted([r for (rs,) in gather(res_q, W, "exam") for r in rs], key=lambda r: r["seed"])
    return rows


def exam_line(rows, rnd):
    n = len(rows)
    ok = [r for r in rows if r["outcome"] == "success"]
    outs = {}
    for r in rows:
        outs[r["outcome"]] = outs.get(r["outcome"], 0) + 1
    return (f"[exam] round {rnd}: safe {len(ok)}/{n}, by drop {sum(r['reason'] == 'drop' for r in ok)}, "
            f"auto-cut {sum(r['reason'] == 'auto_cut' for r in ok)}; outcomes {outs}; "
            f"return {np.mean([r['ret'] for r in rows]):.2f}"), len(ok)


def save(path, model, opt_a, opt_c, rnd, steps, ppo_start, ev_last, cfg):
    import torch
    torch.save(dict(model=model.student.state_dict(), size=model.student.size,
                    log_std=model.log_std.detach().cpu(), critic=model.critic.state_dict(),
                    opt_actor=opt_a.state_dict(), opt_critic=opt_c.state_dict(), round=rnd,
                    steps=steps, ppo_start=ppo_start, ev_last=ev_last, config=cfg), path + ".tmp")
    os.replace(path + ".tmp", path)


def learn(model, opt_a, opt_c, chunks, a, device, train_actor, lr_a):
    import torch
    from v6_drop.learn import ppo as PPO
    for g in opt_a.param_groups:
        g["lr"] = lr_a
    n = len(chunks["reward"])
    rng = np.random.default_rng()
    stats, stop, epochs = [], False, 0
    for _ in range(a.epochs):
        perm = rng.permutation(n)
        for i in range(0, n, a.mb_chunks):
            b = PPO.to_batch(chunks, perm[i:i + a.mb_chunks], device)
            loss, st = PPO.ppo_losses(model, b, a.clip, train_actor, a.ent_coef)
            opt_a.zero_grad(set_to_none=True)
            opt_c.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), a.grad_clip)
            opt_c.step()
            if train_actor:
                torch.nn.utils.clip_grad_norm_(model.actor_parameters(), a.grad_clip)
                opt_a.step()
            stats.append(st)
            if train_actor and st["kl"] > a.kl_stop:
                stop = True
                break
        epochs += 1
        if stop:
            break
    m = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    m["epochs"] = epochs
    m["kl_stop"] = stop
    return m


def main(argv=None):
    import torch
    from v6_drop.learn import ppo as PPO
    from v6_drop.envs.landing_env import ACT_SCALE
    from v6_drop.scripts.dice_exam import SIGMA, sigma_norm
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", help="student checkpoint to start the actor from (dagger_r4/last.pt)")
    ap.add_argument("--resume", help="a PPO last.pt to continue")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default=RUNS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=256, help="steps per worker per round")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--mb-chunks", type=int, default=8, help="chunks per minibatch (8 x 32 = 256 steps, ~2 GB activations in fp32)")
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
    ap.add_argument("--exam-n", type=int, default=100)
    ap.add_argument("--guard", type=float, default=90.0)
    ap.add_argument("--sigma", type=float, nargs=4, default=list(SIGMA))
    ap.add_argument("--shield", type=float, default=None, help="drop shield limit, m/s (Step 7e: 0.8)")
    ap.add_argument("--renderer", choices=["egl", "tiny"], default="egl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    assert (a.init is None) != (a.resume is None), "give exactly one of --init / --resume"
    out = os.path.join(a.out_root, a.name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)

    model = PPO.ActorCritic(sigma_n=sigma_norm(a.sigma)).to(device)
    opt_a = torch.optim.Adam(model.actor_parameters(), lr=0.0)
    opt_c = torch.optim.Adam(model.critic.parameters(), lr=a.lr_critic)
    start, steps, ppo_start, ev_last = 0, 0, None, float("nan")
    if a.init:
        ck = torch.load(a.init, map_location=device, weights_only=False)
        model.student.load_state_dict(ck["model"])
        print(f"[ppo] actor from {os.path.abspath(a.init)}; critic NEW", flush=True)
    else:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.student.load_state_dict(ck["model"])
        model.log_std.data.copy_(ck["log_std"].to(device))
        model.critic.load_state_dict(ck["critic"])
        opt_a.load_state_dict(ck["opt_actor"])
        opt_c.load_state_dict(ck["opt_critic"])
        start, steps, ppo_start, ev_last = ck["round"] + 1, ck["steps"], ck["ppo_start"], ck["ev_last"]
        print(f"[ppo] resumed {os.path.abspath(a.resume)} at round {start}", flush=True)
    cfg = dict(vars(a))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)

    ctx = mp.get_context("spawn")
    cmd_qs = [ctx.Queue() for _ in range(a.workers)]
    res_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(w, cmd_qs[w], res_q, a.renderer, a.sigma, a.seed, a.shield),
                         daemon=True)
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
            sd = cpu_state(model)
            for q in cmd_qs:
                q.put(("collect", sd, a.steps))
            res = gather(res_q, a.workers, "roll")
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
            m = learn(model, opt_a, opt_c, chunks, a, device, train_actor=ppo_start is not None, lr_a=lr_a)
            model.eval()
            ev_last = ev
            n_dec = a.workers * a.steps
            steps += n_dec
            t2 = time.time()
            nf = len(flights)
            safe = sum(f["outcome"] == "success" for f in flights)
            drops = sum(f["outcome"] == "success" and f["reason"] == "drop" for f in flights)
            std = np.exp(model.clamped_log_std().detach().cpu().numpy()) * ACT_SCALE
            row = dict(round=rnd, phase="ppo" if ppo_start is not None else "critic", steps=steps,
                       flights=nf, safe_pct=round(100 * safe / nf, 1) if nf else None,
                       drop_pct=round(100 * drops / nf, 1) if nf else None,
                       hard=sum(f["outcome"] == "hard_landing" for f in flights),
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
                  f"drop {row['drop_pct']}% hard {row['hard']} veto {row['vetoes']} | ret {row['ret']} | EV {row['ev']:.2f} "
                  f"Vloss {row['v_loss']:.3f} | KL {row['kl']:.4f} clip {row['clipfrac']:.2f} ep {row['epochs']} | "
                  f"std vx {row['std_vx']:.3f} vz {row['std_vz']:.3f} | lr {lr_a:.1e} | "
                  f"{row['t_collect']:.0f}+{row['t_learn']:.0f} s, {row['dps']:.0f} dec/s", flush=True)
            save(os.path.join(out, "last.pt"), model, opt_a, opt_c, rnd, steps, ppo_start, ev_last, cfg)
            last_round = rnd == a.rounds - 1
            if ppo_start is not None and ((k + 1) % a.exam_every == 0 or last_round):
                rows = run_exam(cmd_qs, res_q, model, a.exam_n)
                line, k_safe = exam_line(rows, rnd)
                print(line, flush=True)
                tag = f"r{rnd:04d}"
                save(os.path.join(out, tag + ".pt"), model, opt_a, opt_c, rnd, steps, ppo_start, ev_last, cfg)
                with open(os.path.join(out, f"exam_{tag}.csv"), "w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                    wr.writeheader()
                    wr.writerows(rows)
                if 100.0 * k_safe / len(rows) < a.guard:
                    print(f"[guard] safe {k_safe}/{len(rows)} < {a.guard}%: STOP (next: add a copy/KL anchor)",
                          flush=True)
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
