"""v7 S4a: DAgger data collection, one round (20 Nov 2026).

DAgger (Ross et al. 2011): the STUDENT flies; the TEACHER rides along and, at every step,
says what it would do there (from the simulator's true state). We save the student's
picture + sensor numbers with the TEACHER's answer as the label, so the student learns to
fix its OWN mistakes. S3 diagnostic (20 Nov): the BC student overshoots the pad, loses it
behind the 45 deg camera and flies on at ~9 m/s; the teacher would say "go back" there.

The V6 method (v6_drop/scripts/dagger_collect.py, decided 29 Oct) with the V7 parts:
  D1 pure DAgger: the student flies the WHOLE flight; the teacher never takes the controls
     (S4b, 20 Nov: --beta adds beta MIXING. At every step the TEACHER's action is flown with
     chance beta, else the student's (Ross et al. 2011, beta_1 = 1 decaying). Round 1 with
     pure DAgger gave 1/400 on dev: the student never reached the pad, so the data held no
     approach, descent or drop states. Plan: round 2 beta 0.5, round 3 0.25, round 4 0.
     The coin per step comes from the flight's own random stream (seed), so a flight is
     reproducible. The label is always the teacher's; flag = 5 on teacher-flown steps.)
  D2 after each round: continue training from the last student (last.pt), 8 epochs, all data
  D3 no troubles: the student's own mistakes are the troubles now
  ALL flights are kept (failures too: their states are exactly what is needed)
  The student acts deterministically (as in the dev test): mean move, drop if chance > 0.5.
  env       v7.config.make_env() (45 deg, 3 m/s^2, 9 m/s, 60 s; simulator touch, no shield,
            as the demos), the RANDOM pad driver (as the demos)
  teacher   TeacherV7 (its own safe30m drop rule; labels from the true state)
  ids       round r, flight i: episode id r x 10000 + i (demos are 0..1199), seed 600000 + id
            (round 1: seeds 610000..; V7 DAgger seeds, never used elsewhere)
Saved per decision (same keys as the demos, so the V6 dataset builder reads it unchanged):
  image, state, priv (BEFORE the decision), move + drop = LABEL (teacher), flown = the
  student's move, flag 0, phase (teacher), t, ep.
Parallel: W worker processes, one EGL simulator each; the student runs on the CPU in each
worker (one thread), so the GPU is left to the renderer.

Run (repo root; charger in; performance mode):
  nohup python -m v7.scripts.dagger_collect_v7 --ckpt v7/results/runs/bc_v7/last.pt \\
        --round 1 --flights 300 --workers 6 > v7/results/dagger_r1_collect.log 2>&1 &
  round 2 with beta mixing: ... --ckpt v7/results/runs/dagger_r1/last.pt --round 2 --beta 0.5
Watch:  tail -f v7/results/dagger_r1_collect.log | grep --line-buffered -E "^\\[(w|dag)"
Stop:   pkill -f "v7.scripts.dagger_collect_v7"
Output: v7/results/dagger/r<round>/ (git-ignored): dagger_w*.npz, flights.csv, summary.txt
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import time

import numpy as np

from v6_drop.scripts.record_demos import save_chunk
from v7.scripts.record_demos_v7 import PHASES

OUT = os.path.join("v7", "results", "dagger")
SEED0 = 600000
EP_PER_ROUND = 10000
FLUSH_EVERY = 20                          # flights per .npz file
KEYS = ("image", "state", "priv", "move", "drop", "flown", "flag", "phase", "t", "ep")
FLAG_TEACHER = 5                          # flag code: the teacher's action was flown (beta)


def episode_id(rnd, i):
    assert rnd >= 1 and 0 <= i < EP_PER_ROUND
    return rnd * EP_PER_ROUND + i


def fly_dagger(env, model, teacher, ep, device="cpu", beta=0.0):
    """One DAgger flight: the student flies (the teacher with chance beta per step), the
    teacher labels. Returns (rec, row)."""
    import torch
    seed = SEED0 + ep
    coin = np.random.default_rng([seed, 5])
    n_teacher = 0
    obs, info0 = env.reset(seed=seed, options=dict(mode="random"))
    teacher.reset()
    rec = {k: [] for k in KEYS}
    hc, p_max, n_seen, n_rt, err = None, 0.0, 0, 0, []
    with torch.no_grad():
        while True:
            label, ti = teacher.act(env)               # the teacher's answer HERE
            act, hc, p = model.act(obs, hc, device)    # what the student does (memory always runs)
            by_teacher = beta > 0.0 and coin.random() < beta
            flown = {"move": np.asarray(label["move"], np.float32), "drop": int(label["drop"])} \
                if by_teacher else {"move": act["move"], "drop": act["drop"]}
            n_teacher += by_teacher
            rec["image"].append(obs["image"])
            rec["state"].append(obs["state"])
            rec["priv"].append(obs["priv"])
            rec["move"].append(np.asarray(label["move"], np.float32))
            rec["drop"].append(int(label["drop"]))
            rec["flown"].append(np.asarray(flown["move"], np.float32))
            rec["flag"].append(FLAG_TEACHER if by_teacher else 0)
            rec["phase"].append(PHASES.index(ti["phase"]))
            rec["t"].append(env.world.t)
            rec["ep"].append(ep)
            err.append(np.abs(np.asarray(act["move"]) - label["move"]) * env.act_scale)
            p_max = max(p_max, p)
            obs, r, term, trunc, info = env.step(flown)
            rt = info.get("reward_terms") or {}
            if rt.get("seen") is not None:
                n_rt += 1
                n_seen += bool(rt["seen"])
            if term or trunc:
                break
    td = info.get("touchdown") or {}
    err = np.array(err)
    pad = info["pad"]
    row = dict(ep=ep, seed=seed, beta=beta, teacher_share=round(n_teacher / len(rec["t"]), 3),
               outcome=info["outcome"], steps=len(rec["t"]),
               time_s=round(len(rec["t"]) * 0.1, 1),
               pad_v_start=round(pad["v_start"], 3), pad_v_max=round(pad["v_max"], 3),
               reason=td.get("reason"),
               impact=None if td.get("impact") is None else round(td["impact"], 3),
               offset_cm=None if not td.get("land_xy") else round(100 * float(np.linalg.norm(td["land_xy"])), 1),
               teacher_drop_labels=int(sum(rec["drop"])), student_max_drop_chance=round(p_max, 3),
               seen_pct=round(100 * n_seen / max(n_rt, 1), 1),
               err_vx=round(float(err[:, 0].mean()), 3), err_vy=round(float(err[:, 1].mean()), 3),
               err_vz=round(float(err[:, 2].mean()), 3))
    return rec, row


def worker(w, n_workers, rnd, flights, ckpt, out, renderer, q, beta=0.0):
    import torch
    torch.set_num_threads(1)
    from v6_drop.scripts.eval_student import load
    from v7.config import make_env
    from v7.experts.teacher_v7 import TeacherV7
    model, _ = load(ckpt, "cpu")
    env = make_env(renderer=renderer, seed=0)
    teacher = TeacherV7()
    buf = {k: [] for k in KEYS}
    n_buf, chunk, rows, t0 = 0, 0, [], time.time()
    mine = [i for i in range(flights) if i % n_workers == w]
    try:
        for j, i in enumerate(mine):
            rec, row = fly_dagger(env, model, teacher, episode_id(rnd, i), beta=beta)
            rows.append(row)
            for k in KEYS:
                buf[k].extend(rec[k])
            n_buf += 1
            if n_buf >= FLUSH_EVERY:
                save_chunk(buf, os.path.join(out, f"dagger_w{w}_{chunk:03d}.npz"))
                chunk += 1
                buf = {k: [] for k in KEYS}
                n_buf = 0
            if (j + 1) % 10 == 0 or j + 1 == len(mine):
                safe = sum(r["outcome"] == "success" for r in rows)
                print(f"[w{w}] {j + 1}/{len(mine)} flights, safe {safe}, "
                      f"{(time.time() - t0) / (j + 1):.1f} s/flight", flush=True)
        if n_buf:
            save_chunk(buf, os.path.join(out, f"dagger_w{w}_{chunk:03d}.npz"))
    finally:
        env.close()
    q.put(rows)


def summarize(rows, out, ckpt, rnd):
    rows = sorted(rows, key=lambda r: r["ep"])
    with open(os.path.join(out, "flights.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    n = len(rows)
    outs = {}
    for r in rows:
        outs[r["outcome"]] = outs.get(r["outcome"], 0) + 1
    safe = [r for r in rows if r["outcome"] == "success"]
    steps = sum(r["steps"] for r in rows)
    beta = rows[0]["beta"]
    lines = [f"V7 DAgger round {rnd}: student {os.path.abspath(ckpt)} flew {n} flights "
             f"(seeds {rows[0]['seed']}..{rows[-1]['seed']}), all kept; beta {beta:g} (teacher flew "
             f"{100 * np.mean([r['teacher_share'] for r in rows]):.0f} % of the steps)",
             f"samples {steps:,} (mean {steps / n:.0f} per flight)",
             f"{'mixed flights' if beta > 0 else 'student'} safe {len(safe)}/{n} = {100 * len(safe) / n:.1f}% "
             f"(by drop {sum(r['reason'] == 'drop' for r in safe)}, by auto-cut "
             f"{sum(r['reason'] == 'auto_cut' for r in safe)})",
             f"outcomes: {outs}",
             f"teacher drop labels: {sum(r['teacher_drop_labels'] for r in rows)} "
             f"(in {sum(r['teacher_drop_labels'] > 0 for r in rows)} flights)",
             f"student's highest drop chance per flight: median "
             f"{np.median([r['student_max_drop_chance'] for r in rows]):.3f}",
             f"student vs teacher move gap [m/s], mean: vx {np.mean([r['err_vx'] for r in rows]):.2f}, "
             f"vy {np.mean([r['err_vy'] for r in rows]):.2f}, vz {np.mean([r['err_vz'] for r in rows]):.2f}",
             f"marker seen: {np.mean([r['seen_pct'] for r in rows]):.1f}% of steps (mean over flights)",
             f"flight time: median {np.median([r['time_s'] for r in rows]):.1f} s"]
    files = glob.glob(os.path.join(out, "dagger_*.npz"))
    lines.append(f"files: {len(files)} .npz, {sum(os.path.getsize(p) for p in files) / 1e9:.2f} GB")
    txt = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    print(txt, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the student that flies this round")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--flights", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--renderer", choices=["egl", "tiny"], default="egl")
    ap.add_argument("--out", default=None, help="default v7/results/dagger/r<round>")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="chance per step that the TEACHER's action is flown (0 = pure DAgger)")
    a = ap.parse_args(argv)
    assert 0.0 <= a.beta <= 1.0, a.beta
    out = a.out or os.path.join(OUT, f"r{a.round}")
    if glob.glob(os.path.join(out, "dagger_*.npz")):
        raise SystemExit(f"{out} already has data; delete it first (rm -rf {out})")
    os.makedirs(out, exist_ok=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(w, a.workers, a.round, a.flights, a.ckpt, out, a.renderer, q,
                                              a.beta))
             for w in range(a.workers)]
    for pr in procs:
        pr.start()
    rows = []
    for _ in procs:
        rows += q.get()
    for pr in procs:
        pr.join()
    print(f"[dag] round {a.round}: {len(rows)} flights in {(time.time() - t0) / 60:.1f} min", flush=True)
    summarize(rows, out, a.ckpt, a.round)


if __name__ == "__main__":
    main()
