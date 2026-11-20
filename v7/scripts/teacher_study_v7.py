"""v7 S2b: the TEACHER STUDY (20 Nov 2026). Chooses the V7 settings from numbers.

Stages (all with the S2a teacher, v7/experts/teacher_v7.py):
  grid    4 settings: speed-change limit 3 or 5 m/s^2 (rate_xy 0.3 / 0.5) x pad top speed
          9 or 8 m/s; each on the 4 paths (random, straight, wave, corners). HEADLESS (the
          teacher reads the true state). Seeds 32000 + i; the SAME seeds for every setting
          and path (paired comparison).
  check   ONE chosen setting, more flights, FRESH seeds 33000 + i (not used to choose).
  images  ONE chosen setting WITH camera images, the drone's own touch detector and the
          drop shield 0.8 m/s (the Stage 1 rules). Seeds 34000 + i.
Camera angle (60 vs 45 deg) does not change how the teacher flies, only whether the pad is
in the picture. So every flight also counts, by geometry (no images), the steps with the
pad centre in view for BOTH angles, and the longest stretch without it ("blind").

Per flight: outcome, decisions, pad start / mean / max speed, wait steps, view share and
longest blind stretch for 60 and 45 deg, impact. Summary per setting and path: safe % with
the Wilson 95 % interval, failure kinds, longest and 99th-percentile flight time.

Run (laptop, charger in, `powerprofilesctl set performance`):
  python -m v7.scripts.teacher_study_v7 --stage grid --n 200 --workers 8
  python -m v7.scripts.teacher_study_v7 --stage check --rate 0.3 --vmax 9 --n 500 --workers 8
  python -m v7.scripts.teacher_study_v7 --stage images --rate 0.3 --vmax 9 --pitch 45 --n 25 --workers 4
--pitch: the camera angle of the env (45 or 60, default 60). It matters only for the
images stage (marker reading, touch detector, shield); headless stages count both angles.
Output: v7/results/teacher_study/<stage>[_r<rate>_v<vmax>][_p<pitch>].jsonl (one line per flight) and
.txt (the summary table). --summary-only re-prints the table from the .jsonl.
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
SEED0 = {"grid": 32000, "check": 33000, "images": 34000}
OUT_DIR = os.path.join("v7", "results", "teacher_study")
CHUNK = 25                       # flights per task


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    """Wilson 95 % interval for k successes out of n -> (low, high) as fractions."""
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------- one task (in a worker)
def run_task(task):
    """task: dict(stage, rate, vmax, mode, seeds). Returns a list of flight records.
    Creates its own env (one EGL world per process at a time) and closes it."""
    from v6_drop.envs.lmf2_params import POLICY_DT
    from v7.envs.gimbal_camera import GimbalCamera
    from v7.envs.landing_env_v7 import LandingEnvV7
    from v7.experts.teacher_v7 import TeacherV7

    images = task["stage"] == "images"
    kw = dict(touch="sensor", drop_shield=0.8) if images else dict(headless=True)
    pitch = task.get("pitch", 60.0)
    env = LandingEnvV7(renderer="egl", seed=7, rate_xy=task["rate"], v_max=task["vmax"],
                       pitch_deg=pitch, **kw)
    cams = {a: GimbalCamera(env.world.cid, pitch_deg=a, renderer="egl") for a in (60, 45)}
    teacher = TeacherV7()
    out = []
    try:
        for seed in task["seeds"]:
            _, info0 = env.reset(seed=seed, options=dict(mode=task["mode"]))
            teacher.reset()
            n = waits = drops_asked = vetoes = 0
            seen = {60: 0, 45: 0}
            blind = {60: 0, 45: 0}
            blind_max = {60: 0, 45: 0}
            while True:
                s = env.world.true_state()
                for ang, cam in cams.items():
                    uv, z = cam.project(s["pad_center"], s["R"], s["pos"])
                    u, v = uv[0]
                    vis = bool(z[0] > 0 and 0 <= u <= cam.w and 0 <= v <= cam.h)
                    seen[ang] += vis
                    blind[ang] = 0 if vis else blind[ang] + 1
                    blind_max[ang] = max(blind_max[ang], blind[ang])
                a, ti = teacher.act(env)
                waits += ti["phase"] == "wait"
                _, _, term, trunc, info = env.step(a)
                n += 1
                drops_asked += int(a["drop"])
                vetoes += int(bool(info.get("shield_veto")))
                if term or trunc:
                    break
            td = info.get("touchdown") or {}
            out.append(dict(
                stage=task["stage"], rate=task["rate"], vmax=task["vmax"], pitch=pitch,
                mode=task["mode"],
                seed=int(seed), outcome=info["outcome"], t=int(info["t"]),
                pad_v_start=round(float(info["pad"]["v_start"]), 3),
                pad_v_mean=round(float(info["pad"]["v_mean"]), 3),
                pad_v_max=round(float(info["pad"]["v_max"]), 3),
                waits=int(waits), drops_asked=int(drops_asked), vetoes=int(vetoes),
                reason=td.get("reason"), impact=td.get("impact"),
                view60=round(seen[60] / n, 4), view45=round(seen[45] / n, 4),
                blind60_s=round(blind_max[60] * POLICY_DT, 1),
                blind45_s=round(blind_max[45] * POLICY_DT, 1)))
    finally:
        env.close()
    return out


# ---------------------------------------------------------------- summary
def summarize(recs):
    """Text table: one block per (rate, vmax)."""
    lines = []
    settings = sorted({(r["rate"], r["vmax"]) for r in recs}, key=lambda x: (x[0], -x[1]))
    for rate, vmax in settings:
        rs = [r for r in recs if r["rate"] == rate and r["vmax"] == vmax]
        cam = (f", camera {rs[0].get('pitch', 60.0):g} deg" if rs[0]["stage"] == "images" else "")
        lines.append(f"\n=== speed-change limit {rate * 10:.0f} m/s^2, pad top speed {vmax:g} m/s{cam} "
                     f"({len(rs)} flights) ===")
        lines.append(f"{'path':9s} {'safe':>9s} {'95% interval':>14s} {'t max':>6s} {'t p99':>6s} "
                     f"{'view60':>7s} {'view45':>7s} {'blind60':>8s} {'blind45':>8s}  failures")
        for mode in PATHS + ("ALL",):
            sub = rs if mode == "ALL" else [r for r in rs if r["mode"] == mode]
            if not sub:
                continue
            k = sum(r["outcome"] == "success" for r in sub)
            lo, hi = wilson(k, len(sub))
            t = np.array([r["t"] for r in sub]) / 10.0
            fails = Counter(r["outcome"] for r in sub if r["outcome"] != "success")
            lines.append(
                f"{mode:9s} {k:4d}/{len(sub):<4d} {100 * lo:6.1f} to {100 * hi:5.1f} "
                f"{t.max():5.1f}s {np.percentile(t, 99):5.1f}s "
                f"{100 * np.mean([r['view60'] for r in sub]):6.1f}% "
                f"{100 * np.mean([r['view45'] for r in sub]):6.1f}% "
                f"{np.mean([r['blind60_s'] for r in sub]):7.1f}s "
                f"{np.mean([r['blind45_s'] for r in sub]):7.1f}s  {dict(fails)}")
        ok_t = np.array([r["t"] for r in rs if r["outcome"] == "success"]) / 10.0
        if len(ok_t):
            lim = next((x for x in (45, 50, 60) if x >= 1.3 * ok_t.max()), None)
            lines.append(f"longest SAFE flight {ok_t.max():.1f} s -> x 1.3 = {1.3 * ok_t.max():.1f} s "
                         f"-> flight-time limit {lim if lim else 'over 60'} s")
        if any(r["stage"] == "images" for r in rs):
            lines.append(f"drop requests {sum(r['drops_asked'] for r in rs)}, shield vetoes "
                         f"{sum(r['vetoes'] for r in rs)}; landings by: "
                         f"{dict(Counter(r['reason'] for r in rs))}")
    return "\n".join(lines)


def out_name(args):
    if args.stage == "grid":
        return os.path.join(OUT_DIR, "grid")
    name = f"{args.stage}_r{args.rate[0]:g}_v{args.vmax[0]:g}"
    return os.path.join(OUT_DIR, name + (f"_p{args.pitch:g}" if args.stage == "images" else ""))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("grid", "check", "images"), required=True)
    ap.add_argument("--rate", type=float, nargs="+", default=None, help="0.3 and/or 0.5")
    ap.add_argument("--vmax", type=float, nargs="+", default=None, help="9 and/or 8")
    ap.add_argument("--pitch", type=float, default=60.0, choices=(45.0, 60.0),
                    help="camera angle of the env (images stage)")
    ap.add_argument("--n", type=int, default=200, help="flights per path per setting")
    ap.add_argument("--paths", nargs="+", default=list(PATHS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="file name without extension")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)
    if args.rate is None:
        args.rate = [0.3, 0.5] if args.stage == "grid" else [0.3]
    if args.vmax is None:
        args.vmax = [9.0, 8.0] if args.stage == "grid" else [9.0]
    if args.stage != "grid":
        assert len(args.rate) == 1 and len(args.vmax) == 1, "check/images: ONE setting"
    base = args.out or out_name(args)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    if args.summary_only:
        with open(base + ".jsonl") as f:
            recs = [json.loads(x) for x in f]
        print(summarize(recs))
        return recs

    seeds = [SEED0[args.stage] + i for i in range(args.n)]
    tasks = [dict(stage=args.stage, rate=r, vmax=v, pitch=args.pitch, mode=m, seeds=seeds[i:i + CHUNK])
             for r in args.rate for v in args.vmax for m in args.paths
             for i in range(0, len(seeds), CHUNK)]
    total = sum(len(t["seeds"]) for t in tasks)
    print(f"[study] stage {args.stage}: {len(tasks)} tasks, {total} flights, "
          f"{args.workers} workers -> {base}.jsonl", flush=True)
    t0 = time.time()
    recs, done = [], 0
    with open(base + ".jsonl", "w") as f:
        if args.workers <= 1:
            results = map(run_task, tasks)
        else:
            pool = mp.get_context("spawn").Pool(args.workers)
            results = pool.imap_unordered(run_task, tasks)
        for chunk in results:
            for r in chunk:
                f.write(json.dumps(r) + "\n")
            f.flush()
            recs += chunk
            done += len(chunk)
            k = sum(r["outcome"] == "success" for r in recs)
            el = time.time() - t0
            print(f"[study] {done}/{total} flights, safe so far {k}/{done} = {100 * k / done:.1f} %, "
                  f"{el / 60:.1f} min, about {el / done * (total - done) / 60:.1f} min left",
                  flush=True)
        if args.workers > 1:
            pool.close()
            pool.join()
    table = summarize(recs)
    with open(base + ".txt", "w") as f:
        f.write(table + "\n")
    print(table, flush=True)
    print(f"[study] done in {(time.time() - t0) / 60:.1f} min -> {base}.txt", flush=True)
    return recs


if __name__ == "__main__":
    main(sys.argv[1:])
