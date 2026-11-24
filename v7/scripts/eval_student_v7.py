"""v7 S3c: the student's DEV TEST (20 Nov 2026).

The student flies alone (camera + own sensors + its own memory), deterministic: the mean
move, and a drop when its drop chance is above 0.5 (as the V6 exam). Stage 1 rules
(v7.config.make_env(student_rules=True)): camera 45 deg, 3 m/s^2, pad top speed 9 m/s, 60 s,
drop shield 0.8 m/s, the drone's own touch detector, the five landing checks.
Each of the 4 pad paths (random, straight, wave, corners) on the DEV seeds 30000 + i
(the same seed numbers for every path and every student, so students compare flight by
flight). The final test (S7) uses other seeds (40000+); dev numbers are NOT paper numbers.

Per flight: outcome, landed by (drop / touch detector / simulator cut), impact, decisions,
pad start / mean / max speed, drop presses and shield blocks, highest drop chance, share of
steps with a marker seen. Summary: safe % (Wilson 95 %) per path, per pad top-speed band
(0-3, 3-6, 6-9 m/s) and in total, failure kinds, landed-by counts, impact.

Run (repo root; charger in; performance mode):
  python -m v7.scripts.eval_student_v7 --ckpt v7/results/runs/bc_v7/last.pt --n 100 --workers 4
Output next to the checkpoint: dev_<ckpt stem>_<n>.jsonl (one line per flight) and .txt.
--summary-only re-prints the table from the .jsonl.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

import numpy as np

PATHS = ("random", "straight", "wave", "corners")
SEED0 = 30000
CHUNK = 10
BANDS = ((0.0, 3.0), (3.0, 6.0), (6.0, 9.01))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def run_task(task):
    """task: dict(ckpt, mode, seeds, device). One env and one student per task."""
    import torch
    from v6_drop.scripts.eval_student import load
    from v7.config import make_env
    torch.set_num_threads(1)
    device = task.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load(task["ckpt"], device)
    env = make_env(student_rules=True, renderer="egl", seed=7,
                   gimbal_schedule=task.get("gimbal_schedule", "none"))
    out = []
    try:
        for seed in task["seeds"]:
            obs, info0 = env.reset(seed=seed, options=dict(mode=task["mode"]))
            hc, n, presses, blocks, p_max, seen, n_seen = None, 0, 0, 0, 0.0, 0, 0
            with torch.no_grad():
                while True:
                    act, hc, p_drop = model.act(obs, hc, device)
                    p_max = max(p_max, float(p_drop))
                    obs, _, term, trunc, info = env.step(act)
                    n += 1
                    presses += int(info.get("drop_requested", False))
                    blocks += int(bool(info.get("shield_veto")))
                    rt = info.get("reward_terms") or {}
                    if rt.get("seen") is not None:
                        seen += int(bool(rt["seen"]))
                        n_seen += 1
                    if term or trunc:
                        break
            td = info.get("touchdown") or {}
            out.append(dict(
                mode=task["mode"], seed=int(seed), outcome=info["outcome"], t=int(info["t"]),
                landed_by=td.get("reason"), impact=td.get("impact"),
                pad_v_start=round(float(info["pad"]["v_start"]), 3),
                pad_v_mean=round(float(info["pad"]["v_mean"]), 3),
                pad_v_max=round(float(info["pad"]["v_max"]), 3),
                presses=presses, blocks=blocks, p_drop_max=round(p_max, 3),
                seen=round(seen / max(n_seen, 1), 3)))
    finally:
        env.close()
    return out


def _line(name, sub):
    k = sum(r["outcome"] == "success" for r in sub)
    lo, hi = wilson(k, len(sub))
    fails = Counter(r["outcome"] for r in sub if r["outcome"] != "success")
    return (f"{name:10s} {k:4d}/{len(sub):<4d} {100 * k / max(len(sub), 1):5.1f} % "
            f"({100 * lo:5.1f} to {100 * hi:5.1f})  {dict(fails)}")


def summarize(recs, title=""):
    lines = [f"=== {title} ({len(recs)} flights) ===",
             f"{'':10s} {'safe':>9s} {'':>7s} {'95% interval':>16s}  failures"]
    for m in PATHS:
        sub = [r for r in recs if r["mode"] == m]
        if sub:
            lines.append(_line(m, sub))
    lines.append(_line("ALL", recs))
    lines.append("by pad top speed:")
    for a, b in BANDS:
        sub = [r for r in recs if a <= r["pad_v_max"] < b]
        if sub:
            lines.append(_line(f"{a:g}-{min(b, 9):g} m/s", sub))
    ok = [r for r in recs if r["outcome"] == "success"]
    lb = Counter(r["landed_by"] for r in recs if r["landed_by"])
    lines.append(f"landed by: {dict(lb)}  (drop = motors cut by the student's drop)")
    imp = [r["impact"] for r in ok if r["impact"] is not None]
    if imp:
        lines.append(f"impact of safe landings: mean {np.mean(imp):.2f}, max {np.max(imp):.2f} m/s")
    lines.append(f"drop presses {sum(r['presses'] for r in recs)}, shield blocks "
                 f"{sum(r['blocks'] for r in recs)}; flights with a press "
                 f"{sum(r['presses'] > 0 for r in recs)}")
    lines.append(f"marker seen: {100 * np.mean([r['seen'] for r in recs]):.0f} % of steps; "
                 f"flight time median {np.median([r['t'] for r in recs]) / 10:.1f} s")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100, help="flights per path")
    ap.add_argument("--paths", nargs="+", default=list(PATHS))
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="file name without extension")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--gimbal-schedule", default="none",
                    help="S4g: gimbal tilt by height (h3to6 = 60 deg below 3 m, 45 deg above 6 m)")
    a = ap.parse_args(argv)
    stem = os.path.splitext(os.path.basename(a.ckpt))[0]
    gs = "" if a.gimbal_schedule == "none" else f"_g{a.gimbal_schedule}"
    base = a.out or os.path.join(os.path.dirname(a.ckpt), f"dev_{stem}_{a.n}{gs}")
    title = f"{a.ckpt}, dev seeds {a.seed0}+, student rules" + (
        f", gimbal schedule {a.gimbal_schedule}" if gs else "")
    if a.summary_only:
        recs = [json.loads(x) for x in open(base + ".jsonl")]
        print(summarize(recs, title))
        return recs
    seeds = [a.seed0 + i for i in range(a.n)]
    tasks = [dict(ckpt=a.ckpt, mode=m, seeds=seeds[i:i + CHUNK], device=a.device,
                  gimbal_schedule=a.gimbal_schedule)
             for m in a.paths for i in range(0, len(seeds), CHUNK)]
    total = len(seeds) * len(a.paths)
    print(f"[dev] {a.ckpt}: {total} flights, {a.workers} workers -> {base}.jsonl", flush=True)
    t0, recs = time.time(), []
    with open(base + ".jsonl", "w") as f:
        if a.workers <= 1:
            results = map(run_task, tasks)
        else:
            pool = mp.get_context("spawn").Pool(a.workers)
            results = pool.imap_unordered(run_task, tasks)
        for chunk in results:
            for r in chunk:
                f.write(json.dumps(r) + "\n")
            f.flush()
            recs += chunk
            k = sum(r["outcome"] == "success" for r in recs)
            el = time.time() - t0
            print(f"[dev] {len(recs)}/{total} flights, safe so far {k}/{len(recs)} = "
                  f"{100 * k / len(recs):.1f} %, {el / 60:.1f} min, about "
                  f"{el / len(recs) * (total - len(recs)) / 60:.1f} min left", flush=True)
        if a.workers > 1:
            pool.close()
            pool.join()
    table = summarize(recs, title)
    open(base + ".txt", "w").write(table + "\n")
    print(table, flush=True)
    print(f"[dev] done in {(time.time() - t0) / 60:.1f} min -> {base}.txt", flush=True)
    return recs


if __name__ == "__main__":
    main(sys.argv[1:])
