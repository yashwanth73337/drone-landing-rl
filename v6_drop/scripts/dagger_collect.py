"""v6_drop Step 6d: DAgger data collection (one round).

DAgger (Ross et al. 2011) in plain words: the STUDENT flies; the TEACHER rides along and,
at every step, says what it would do there (from the simulator's true state). We save the
student's picture + sensor numbers with the TEACHER's answer as the label. Training on
these teaches the student to fix its OWN mistakes (states the teacher never visits).

Decided 29 Oct:
  D1 pure DAgger: the student flies the WHOLE flight; the teacher never takes the controls
  D2 after each round: continue training from the last student, 8 epochs, all data
  D3 no troubles: the student's own mistakes are the troubles now
  ALL flights are kept (failures too: their states are exactly what we need)
  Flights: training seeds only. Round r, flight i: episode id r x 10000 + i,
           seed 200000 + id (round 1 -> seeds 210000..; never the test seeds 9000..)
  The student acts deterministically (as in the exam): mean move, drop if chance > 0.5.

Saved per decision (same keys as the demos, so learn/dataset.py builds it unchanged):
  image, state, priv  what the student saw (BEFORE the decision)
  move, drop          LABEL: the teacher's answer in that state
  flown               the move the STUDENT actually flew
  flag                0 (no troubles); phase: the teacher's phase; ep, t

Parallel: W worker processes, one EGL simulator each (B5 rule). The student runs on the
CPU in each worker (one thread), so the GPU is left to the renderer.

Run (from the repo root; performance mode first):
  nohup python -m v6_drop.scripts.dagger_collect --ckpt v6_drop/results/runs/bc256/best.pt \\
        --round 1 --flights 300 --workers 6 > v6_drop/results/dagger_r1_collect.log 2>&1 &
Watch:  tail -f v6_drop/results/dagger_r1_collect.log | grep --line-buffered -E "^\\[(w|dag)"
Stop:   pkill -f "v6_drop.scripts.dagger_collect"
Output: v6_drop/results/dagger/r<round>/  (git-ignored): dagger_w*.npz, flights.csv, summary.txt
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import time

import numpy as np

from v6_drop.scripts.record_demos import PHASES, save_chunk

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "dagger")
SEED0 = 200000
EP_PER_ROUND = 10000
FLUSH_EVERY = 20                          # flights per .npz file
KEYS = ("image", "state", "priv", "move", "drop", "flown", "flag", "phase", "t", "ep")


def episode_id(rnd, i):
    assert 0 <= i < EP_PER_ROUND
    return rnd * EP_PER_ROUND + i


def fly_dagger(env, model, teacher, ep, device="cpu"):
    """One DAgger flight: the student flies, the teacher labels. Returns (rec, row)."""
    import torch
    from v6_drop.envs import lmf2_params as P
    from v6_drop.envs.landing_env import ACT_SCALE
    seed = SEED0 + ep
    obs, info0 = env.reset(seed=seed)
    teacher.reset()
    rec = {k: [] for k in KEYS}
    hc, p_max, n_seen, n_rt = None, 0.0, 0, 0
    err = []
    with torch.no_grad():
        while True:
            label, ti = teacher.act(env)               # the teacher's answer HERE
            act, hc, p = model.act(obs, hc, device)    # what the student does
            rec["image"].append(obs["image"])
            rec["state"].append(obs["state"])
            rec["priv"].append(obs["priv"])
            rec["move"].append(np.asarray(label["move"], np.float32))
            rec["drop"].append(int(label["drop"]))
            rec["flown"].append(np.asarray(act["move"], np.float32))
            rec["flag"].append(0)
            rec["phase"].append(PHASES.index(ti["phase"]))
            rec["t"].append(env.world.t)
            rec["ep"].append(ep)
            err.append(np.abs(np.asarray(act["move"]) - label["move"]) * ACT_SCALE)
            p_max = max(p_max, p)
            obs, r, term, trunc, info = env.step({"move": act["move"], "drop": act["drop"]})
            rt = info.get("reward_terms") or {}
            if rt.get("seen") is not None:
                n_rt += 1
                n_seen += bool(rt["seen"])
            if term or trunc:
                break
    td = info.get("touchdown") or {}
    err = np.array(err)
    row = dict(ep=ep, seed=seed, outcome=info["outcome"], steps=len(rec["t"]),
               time_s=round(len(rec["t"]) * P.POLICY_DT, 1),
               pad_speed=round(info0["pad_speed"], 3), reason=td.get("reason"),
               impact=None if td.get("impact") is None else round(td["impact"], 3),
               offset_cm=None if not td.get("land_xy") else round(100 * float(np.linalg.norm(td["land_xy"])), 1),
               teacher_drop_labels=int(sum(rec["drop"])), student_max_drop_chance=round(p_max, 3),
               seen_pct=round(100 * n_seen / max(n_rt, 1), 1),
               err_vx=round(float(err[:, 0].mean()), 3), err_vy=round(float(err[:, 1].mean()), 3),
               err_vz=round(float(err[:, 2].mean()), 3))
    return rec, row


def worker(w, n_workers, rnd, flights, ckpt, out, renderer, q):
    import torch
    torch.set_num_threads(1)
    from v6_drop.envs import landing_env as E
    from v6_drop.experts.teacher import Teacher
    from v6_drop.scripts.eval_student import load
    model, _ = load(ckpt, "cpu")
    env = E.LandingEnv(renderer=renderer, seed=0)
    teacher = Teacher()
    buf = {k: [] for k in KEYS}
    n_buf, chunk, rows, t0 = 0, 0, [], time.time()
    mine = [i for i in range(flights) if i % n_workers == w]
    try:
        for j, i in enumerate(mine):
            rec, row = fly_dagger(env, model, teacher, episode_id(rnd, i))
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
                print(f"[w{w}] {j + 1}/{len(mine)} flights, student safe {safe}, "
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
    lines = [f"DAgger round {rnd}: student {os.path.abspath(ckpt)} flew {n} flights "
             f"(seeds {rows[0]['seed']}..{rows[-1]['seed']}), all kept",
             f"samples {steps:,} (mean {steps / n:.0f} per flight)",
             f"student safe {len(safe)}/{n} = {100 * len(safe) / n:.1f}% "
             f"(by drop {sum(r['reason'] == 'drop' for r in safe)}, by auto-cut "
             f"{sum(r['reason'] == 'auto_cut' for r in safe)})",
             f"outcomes: {outs}",
             f"teacher drop labels: {sum(r['teacher_drop_labels'] for r in rows)} "
             f"(in {sum(r['teacher_drop_labels'] > 0 for r in rows)} flights)",
             f"student's highest drop chance per flight: median "
             f"{np.median([r['student_max_drop_chance'] for r in rows]):.3f}",
             f"student vs teacher move gap [m/s], mean: vx {np.mean([r['err_vx'] for r in rows]):.2f}, "
             f"vy {np.mean([r['err_vy'] for r in rows]):.2f}, vz {np.mean([r['err_vz'] for r in rows]):.2f}",
             f"marker seen: {np.mean([r['seen_pct'] for r in rows]):.1f}% of steps (mean over flights)"]
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(out, "dagger_*.npz")))
    lines.append(f"files: {len(glob.glob(os.path.join(out, 'dagger_*.npz')))} .npz, {size / 1e9:.2f} GB")
    txt = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    print(txt, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the student that flies this round")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--flights", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--renderer", choices=["egl", "tiny"], default="egl")
    ap.add_argument("--out", default=None, help="default results/dagger/r<round>")
    a = ap.parse_args()
    assert a.round >= 1
    out = a.out or os.path.join(OUT, f"r{a.round}")
    if glob.glob(os.path.join(out, "dagger_*.npz")):
        raise SystemExit(f"{out} already has data; delete it first (rm -rf {out})")
    os.makedirs(out, exist_ok=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(w, a.workers, a.round, a.flights, a.ckpt, out, a.renderer, q))
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
