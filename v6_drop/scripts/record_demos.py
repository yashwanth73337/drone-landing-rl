"""v6_drop Step 6b: record the teacher's demonstrations, with troubles (DART).

For every attempt e (seed 100000 + e, never a test seed): pick a trouble kind from the
seed (mix 25/25/20/15/15), fly the teacher, disturb the FLOWN action, and save at every
decision:
  image   uint8 (160, 256)  what the student sees (BEFORE the decision)
  state   float32 (11,)     the student's own-sensor numbers
  priv    float32 (8,)      critic-only true relative state
  move    float32 (4,)      LABEL: the teacher's move action in [-1, 1]
  drop    uint8             LABEL: the teacher's drop decision
  flown   float32 (4,)      the move action actually flown (label + trouble)
  flag    uint8             trouble active this step (troubles.CODE)
  phase   uint8             teacher phase (PHASES)
  ep, t   int32, int16      episode id (= attempt e) and decision index
Only episodes that end in a SAFE landing are kept (R3). Every attempt is listed in
episodes.csv, kept or not.

Parallel: W worker processes, one EGL simulator each (B5 rule). Worker w takes the
attempts e with e % W == w, so the data does not depend on W.

Run (from the repo root; set performance mode first):
  nohup python -m v6_drop.scripts.record_demos --attempts 1040 --workers 6 \\
        > v6_drop/results/demos.log 2>&1 &
Watch:  tail -f v6_drop/results/demos.log | grep --line-buffered -E "^\\[(w|rec)"
Stop:   pkill -f "v6_drop.scripts.record_demos"
Output: v6_drop/results/demos/  (git-ignored): demos_w*_*.npz, episodes.csv, summary.txt
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import time

import numpy as np

PHASES = ("follow", "glide", "hover", "go-around", "drop")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "demos")
SEED0 = 100000
FLUSH_EVERY = 20                         # kept episodes per .npz file


def fly_and_record(env, teacher, troubles, e):
    """One attempt. Returns (steps dict of lists, episode row)."""
    from v6_drop.experts.troubles import CODE, pick_kind
    seed = SEED0 + e
    rng = np.random.default_rng(seed)
    kind = pick_kind(rng)
    obs, info0 = env.reset(seed=seed)
    teacher.reset()
    troubles.reset(kind, rng)
    rec = {k: [] for k in ("image", "state", "priv", "move", "drop", "flown", "flag", "phase", "t")}
    seen, go_around, n_flag = True, False, 0
    while True:
        act, ti = teacher.act(env)
        flown, flag = troubles.apply(env, act, seen)
        rec["image"].append(obs["image"])
        rec["state"].append(obs["state"])
        rec["priv"].append(obs["priv"])
        rec["move"].append(np.asarray(act["move"], np.float32))
        rec["drop"].append(int(act["drop"]))
        rec["flown"].append(np.asarray(flown["move"], np.float32))
        rec["flag"].append(CODE[flag])
        rec["phase"].append(PHASES.index(ti["phase"]))
        rec["t"].append(env.world.t)
        go_around |= ti["phase"] == "go-around"
        n_flag += flag != ""
        obs, r, term, trunc, info = env.step(flown)
        rt = info.get("reward_terms") or {}
        if rt.get("seen") is not None:
            seen = bool(rt["seen"])
        if term or trunc:
            break
    td = info.get("touchdown") or {}
    row = dict(ep=e, seed=seed, kind=kind, outcome=info["outcome"],
               kept=int(info["outcome"] == "success"), steps=len(rec["t"]),
               pad_speed=round(info0["pad_speed"], 3),
               impact=None if td.get("impact") is None else round(td["impact"], 3),
               reason=td.get("reason"),
               offset_cm=None if not td.get("land_xy") else round(100 * float(np.linalg.norm(td["land_xy"])), 1),
               go_around=int(go_around), trouble_steps=n_flag)
    return rec, row


def save_chunk(buf, path):
    np.savez_compressed(
        path,
        image=np.stack(buf["image"]).astype(np.uint8),
        state=np.stack(buf["state"]).astype(np.float32),
        priv=np.stack(buf["priv"]).astype(np.float32),
        move=np.stack(buf["move"]).astype(np.float32),
        drop=np.asarray(buf["drop"], np.uint8),
        flown=np.stack(buf["flown"]).astype(np.float32),
        flag=np.asarray(buf["flag"], np.uint8),
        phase=np.asarray(buf["phase"], np.uint8),
        ep=np.asarray(buf["ep"], np.int32),
        t=np.asarray(buf["t"], np.int16))


def worker(w, n_workers, start, attempts, out, renderer, q):
    from v6_drop.envs import landing_env as E
    from v6_drop.experts.teacher import Teacher
    from v6_drop.experts.troubles import Troubles
    env = E.LandingEnv(renderer=renderer, seed=0)
    teacher, troubles = Teacher(), Troubles()
    keys = ("image", "state", "priv", "move", "drop", "flown", "flag", "phase", "t", "ep")
    buf = {k: [] for k in keys}
    n_buf, chunk, rows, t0 = 0, 0, [], time.time()
    mine = [e for e in range(start, start + attempts) if e % n_workers == w]
    try:
        for i, e in enumerate(mine):
            rec, row = fly_and_record(env, teacher, troubles, e)
            rows.append(row)
            if row["kept"]:
                for k in rec:
                    buf[k].extend(rec[k])
                buf["ep"].extend([e] * len(rec["t"]))
                n_buf += 1
                if n_buf >= FLUSH_EVERY:
                    save_chunk(buf, os.path.join(out, f"demos_w{w}_{start}_{chunk:03d}.npz"))
                    chunk += 1
                    buf = {k: [] for k in keys}
                    n_buf = 0
            if (i + 1) % 10 == 0 or i + 1 == len(mine):
                kept = sum(r["kept"] for r in rows)
                print(f"[w{w}] {i + 1}/{len(mine)} attempts, kept {kept}, "
                      f"{(time.time() - t0) / (i + 1):.1f} s/attempt", flush=True)
        if n_buf:
            save_chunk(buf, os.path.join(out, f"demos_w{w}_{start}_{chunk:03d}.npz"))
    finally:
        env.close()
    q.put(rows)


def summarize(rows, out):
    from v6_drop.experts.troubles import KINDS
    rows = sorted(rows, key=lambda r: r["ep"])
    path = os.path.join(out, "episodes.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            wr.writeheader()
        wr.writerows(rows)
    with open(path) as f:
        allrows = list(csv.DictReader(f))
    lines = [f"Demonstrations in {os.path.abspath(out)}"]
    tot_kept = sum(int(r["kept"]) for r in allrows)
    tot_steps = sum(int(r["steps"]) for r in allrows if int(r["kept"]))
    lines.append(f"attempts {len(allrows)}, kept (safe landings) {tot_kept}, samples {tot_steps}")
    for k in KINDS:
        rk = [r for r in allrows if r["kind"] == k]
        if not rk:
            continue
        kept = sum(int(r["kept"]) for r in rk)
        outs = {}
        for r in rk:
            outs[r["outcome"]] = outs.get(r["outcome"], 0) + 1
        ga = sum(int(r["go_around"]) for r in rk if int(r["kept"]))
        lines.append(f"  {k:13s} attempts {len(rk):4d}  kept {kept:4d} ({100 * kept / len(rk):5.1f}%)  "
                     f"go-arounds in kept {ga:4d}  outcomes {outs}")
    imp = [float(r["impact"]) for r in allrows if int(r["kept"])]
    if imp:
        lines.append(f"impact of kept landings: mean {np.mean(imp):.2f}, max {np.max(imp):.2f} m/s")
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(out, "demos_*.npz")))
    lines.append(f"files: {len(glob.glob(os.path.join(out, 'demos_*.npz')))} .npz, {size / 1e9:.2f} GB")
    txt = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    print(txt, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=1040)
    ap.add_argument("--start", type=int, default=0, help="first attempt index (top-ups)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--renderer", choices=["egl", "tiny"], default="egl")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(w, a.workers, a.start, a.attempts, a.out, a.renderer, q))
             for w in range(a.workers)]
    for pr in procs:
        pr.start()
    rows = []
    for _ in procs:
        rows += q.get()
    for pr in procs:
        pr.join()
    print(f"[rec] {len(rows)} attempts in {(time.time() - t0) / 60:.1f} min", flush=True)
    summarize(rows, a.out)


if __name__ == "__main__":
    main()
