"""v7 S3a: record the V7 teacher's demonstrations WITH camera images and troubles (DART).

Same as the V6 recorder (v6_drop/scripts/record_demos.py) with the V7 parts:
  env       v7.config.make_env(): camera 45 deg, 3 m/s^2, pad top speed 9 m/s, 60 s; the
            simulator's touch and no shield (as V6 demos; the student is tested with both)
  pad       the RANDOM driver only (D1, 20 Nov); the named paths stay test cases
  teacher   TeacherV7 (S2a); troubles TroublesV7 (V6 kinds, mix and numbers)
  seeds     500000 + e (V7 demos, never used elsewhere); D2: 1200 attempts
  phases    7 teacher phases (V6 5 + "wait", "abort")
For every attempt e: pick a trouble kind from the seed, fly the teacher, disturb the FLOWN
action, and save at every decision:
  image uint8 (160, 256), state float32 (11,), priv float32 (8,), move float32 (4,) LABEL,
  drop uint8 LABEL, flown float32 (4,), flag uint8, phase uint8, ep int32, t int16
Only SAFE landings are kept. Every attempt is listed in episodes.csv (kept or not), with the
pad's start / mean / max speed and the flight time.
Parallel: W worker processes, one EGL simulator each; worker w takes the attempts with
e % W == w, so the data does not depend on W. The .npz files are the V6 format, so the V6
dataset builder (v6_drop.learn.dataset) reads them unchanged.

Run (repo root; charger in; `powerprofilesctl set performance`):
  nohup python -m v7.scripts.record_demos_v7 --attempts 1200 --workers 6 \\
        > v7/results/demos.log 2>&1 &
Watch:  tail -f v7/results/demos.log | grep --line-buffered -E "^\\[(w|rec)"
Stop:   pkill -f "v7.scripts.record_demos_v7"
Output: v7/results/demos/ (git-ignored): demos_w*_*.npz, episodes.csv, summary.txt

S3v (21 Nov): --teacher view records the VIEW-KEEPING teacher (v7.config.make_teacher; default
v7 = TeacherV7, so earlier runs reproduce). One teacher per folder (teacher.txt; mixing is
refused). Run for 60 deg with the view teacher:
  nohup python -m v7.scripts.record_demos_v7 --teacher view --attempts 1200 --workers 6 \\
        --out v7/results/p60v/demos > v7/results/p60v/demos.log 2>&1 &
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import time

import numpy as np

from v6_drop.scripts.record_demos import save_chunk

PHASES = ("follow", "glide", "hover", "go-around", "drop", "wait", "abort")
OUT = os.path.join("v7", "results", "demos")
SEED0 = 500000
FLUSH_EVERY = 20                         # kept episodes per .npz file
KEYS = ("image", "state", "priv", "move", "drop", "flown", "flag", "phase", "t")


def fly_and_record(env, teacher, troubles, e):
    """One attempt. Returns (steps dict of lists, episode row)."""
    from v7.experts.troubles_v7 import CODE, pick_kind
    seed = SEED0 + e
    rng = np.random.default_rng(seed)
    kind = pick_kind(rng)
    obs, info0 = env.reset(seed=seed, options=dict(mode="random"))
    teacher.reset()
    troubles.reset(kind, rng)
    rec = {k: [] for k in KEYS}
    seen, go_around, n_flag, waits = True, False, 0, 0
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
        waits += ti["phase"] == "wait"
        n_flag += flag != ""
        obs, r, term, trunc, info = env.step(flown)
        rt = info.get("reward_terms") or {}
        if rt.get("seen") is not None:
            seen = bool(rt["seen"])
        if term or trunc:
            break
    td = info.get("touchdown") or {}
    pad = info["pad"]
    row = dict(ep=e, seed=seed, kind=kind, outcome=info["outcome"],
               kept=int(info["outcome"] == "success"), steps=len(rec["t"]),
               flight_s=round(len(rec["t"]) * 0.1, 1),
               pad_v_start=round(pad["v_start"], 3), pad_v_mean=round(pad["v_mean"], 3),
               pad_v_max=round(pad["v_max"], 3),
               impact=None if td.get("impact") is None else round(td["impact"], 3),
               reason=td.get("reason"),
               offset_cm=None if not td.get("land_xy") else round(100 * float(np.linalg.norm(td["land_xy"])), 1),
               go_around=int(go_around), wait_steps=int(waits), trouble_steps=n_flag)
    return rec, row


def worker(w, n_workers, start, attempts, out, renderer, q, teacher_kind="v7"):
    from v7.config import make_env, make_teacher
    from v7.experts.troubles_v7 import TroublesV7
    env = make_env(renderer=renderer, seed=0)
    teacher, troubles = make_teacher(teacher_kind), TroublesV7()
    buf = {k: [] for k in KEYS + ("ep",)}
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
                    buf = {k: [] for k in KEYS + ("ep",)}
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
    from v7.experts.troubles_v7 import KINDS
    tpath = os.path.join(out, "teacher.txt")
    teacher_kind = open(tpath).read().strip() if os.path.exists(tpath) else "v7"
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
    kept_rows = [r for r in allrows if int(r["kept"])]
    lines = [f"V7 demonstrations in {os.path.abspath(out)} (teacher: {teacher_kind})"]
    lines.append(f"attempts {len(allrows)}, kept (safe landings) {len(kept_rows)}, "
                 f"samples {sum(int(r['steps']) for r in kept_rows)}")
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
    if kept_rows:
        imp = [float(r["impact"]) for r in kept_rows]
        fl = [float(r["flight_s"]) for r in kept_rows]
        vm = [float(r["pad_v_max"]) for r in kept_rows]
        lines.append(f"impact of kept landings: mean {np.mean(imp):.2f}, max {np.max(imp):.2f} m/s")
        lines.append(f"flight time of kept: median {np.median(fl):.1f} s, max {np.max(fl):.1f} s")
        lines.append("pad top speed in kept flights: " + ", ".join(
            f"{a}-{b} m/s {sum(a <= v < b for v in vm)}" for a, b in ((0, 3), (3, 6), (6, 9.01))))
        lines.append(f"flights with a wait: {sum(int(r['wait_steps']) > 0 for r in kept_rows)}")
    files = glob.glob(os.path.join(out, "demos_*.npz"))
    lines.append(f"files: {len(files)} .npz, {sum(os.path.getsize(p) for p in files) / 1e9:.2f} GB")
    txt = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    print(txt, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=1200)
    ap.add_argument("--start", type=int, default=0, help="first attempt index (top-ups)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--renderer", choices=["egl", "tiny"], default="egl")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--teacher", choices=("v7", "view"), default="v7",
                    help="v7 = TeacherV7 (S2a); view = view-keeping teacher (S2', 21 Nov)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from v7.config import check_teacher_file
    check_teacher_file(a.out, a.teacher)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(w, a.workers, a.start, a.attempts, a.out, a.renderer, q,
                                              a.teacher))
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
