"""v6_drop Step 8: the FINAL EVALUATION (decided 31 Oct, E1-E4 "all ok"; taught in 8.1).

Rules (8.1):
  * FRESH seeds 9500-9999 (never used for training, dev choices or the teacher check)
  * the SAME seeds for every student (paired comparison)
  * CLEAN flights (no dice), CPU, ONE script: every flight is dice_exam.fly_dice(mode
    'clean'), exactly like all our exams
  * RUN ONCE: the output folder must not exist yet (--force only for a crashed run)

Students (E1; name, checkpoint under results/runs, shield):
  bc256             bc256/best.pt          no shield   behaviour cloning only
  dagger_r4         dagger_r4/last.pt      no shield   imitation student
  dagger_r4_shield  dagger_r4/last.pt      0.8         shield added to imitation
  ppo1              ppo1/last.pt           no shield   PPO without shield
  ppo2_r0299        ppo2/r0299.pt          0.8         MAIN METHOD (DAgger -> PPO + shield, 300 rounds)
  ppo2_r0499        ppo2/r0499.pt          0.8         longer training (press and wait)
  ppo3_r0499        ppo3/r0499.pt          0.8         press cost 0.05
The shield is switched per flight on the worker's one simulator (B5: one EGL world per
process). The press cost is 0 here, so returns are comparable between students.

Report (E3), results/final_eval/<tag>/:
  flights_<student>.csv   every flight
  summary.txt             main table; per student: safe % and drop % with Wilson 95% CI,
                          rule-of-three bound when there are 0 failures, failure types,
                          impact mean/p95/max, offset mean/p95, time, pad-speed bands;
                          paired comparisons: McNemar exact test for safe and for drop
                          (only seeds where the two students disagree count) and the
                          paired flight-time difference with a 95% CI
  compare.png             safe %, drop % (with CIs) and time per student

Run (from the repo root; AC power + performance mode first):
  nohup python -m v6_drop.scripts.final_eval --tag test9500 --n 500 --seed0 9500 --workers 12 \\
        > v6_drop/results/final_eval_test9500.log 2>&1 &
Watch:  tail -f v6_drop/results/final_eval_test9500.log | grep --line-buffered -E "^\\[(w|final)"
Stop:   pkill -f "v6_drop.scripts.final_eval"
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import time

import numpy as np

RUNS = os.path.join(os.path.dirname(__file__), "..", "results", "runs")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "final_eval")
STUDENTS = [("bc256", "bc256/best.pt", None),
            ("dagger_r4", "dagger_r4/last.pt", None),
            ("dagger_r4_shield", "dagger_r4/last.pt", 0.8),
            ("ppo1", "ppo1/last.pt", None),
            ("ppo2_r0299", "ppo2/r0299.pt", 0.8),
            ("ppo2_r0499", "ppo2/r0499.pt", 0.8),
            ("ppo3_r0499", "ppo3/r0499.pt", 0.8)]
PAIRS = [("bc256", "dagger_r4"), ("dagger_r4", "dagger_r4_shield"), ("dagger_r4_shield", "ppo2_r0299"),
         ("ppo1", "ppo2_r0299"), ("ppo2_r0299", "ppo2_r0499"), ("ppo2_r0499", "ppo3_r0499")]
BANDS = ((0, 1), (1, 2), (2, 3), (3, 4.01))


# ------------------------------------------------------------------ statistics (tested)
def wilson(k, n, z=1.96):
    """Wilson 95% interval in %."""
    if n == 0:
        return float("nan"), float("nan")
    c = (k + z * z / 2) / (n + z * z)
    hw = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return 100 * (c - hw), 100 * (c + hw)


def rule_of_three(n, alpha=0.05):
    """0 events in n trials -> one-sided 95% upper bound on the rate, in % (exact; ~3/n)."""
    return 100 * (1 - alpha ** (1 / n))


def mcnemar(a, b):
    """Exact McNemar test for paired yes/no results a, b (same seeds).
    Returns (only_a, only_b, two-sided p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    m = only_a + only_b
    if m == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return only_a, only_b, min(1.0, 2 * tail)


def paired_diff(x, y, z=1.96):
    """Mean of (y - x) over pairs with a normal 95% CI: (mean, lo, hi)."""
    d = np.asarray(y, float) - np.asarray(x, float)
    m = float(d.mean())
    if len(d) < 2:
        return m, float("nan"), float("nan")
    hw = z * float(d.std(ddof=1)) / math.sqrt(len(d))
    return m, m - hw, m + hw


def describe(rows):
    """Per-student numbers from flight rows (dice_exam.fly_dice rows)."""
    n = len(rows)
    ok = [r for r in rows if r["outcome"] == "success"]
    drops = [r for r in ok if r["reason"] == "drop"]
    outs = {}
    for r in rows:
        outs[r["outcome"]] = outs.get(r["outcome"], 0) + 1
    imp = np.array([r["impact"] for r in ok if r["impact"] is not None], float)
    off = np.array([r["offset_cm"] for r in ok if r["offset_cm"] is not None], float)
    bands = []
    for lo, hi in BANDS:
        rb = [r for r in rows if lo <= r["pad_speed"] < hi]
        bands.append(dict(band=f"{lo}-{int(hi)}", n=len(rb), safe=sum(r["outcome"] == "success" for r in rb),
                          drop=sum(r["outcome"] == "success" and r["reason"] == "drop" for r in rb)))
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")
    return dict(n=n, safe=len(ok), drop=len(drops), auto_cut=len(ok) - len(drops), outcomes=outs,
                safe_ci=wilson(len(ok), n), drop_ci=wilson(len(drops), n),
                fail_bound=rule_of_three(n) if len(ok) == n else None,
                impact_mean=float(imp.mean()) if len(imp) else float("nan"), impact_p95=q(imp, 95),
                impact_max=float(imp.max()) if len(imp) else float("nan"),
                offset_mean=float(off.mean()) if len(off) else float("nan"), offset_p95=q(off, 95),
                time_mean=float(np.mean([r["time_s"] for r in rows])),
                time_safe=float(np.mean([r["time_s"] for r in ok])) if ok else float("nan"),
                vetoes=int(sum(r["vetoes"] for r in rows)), bands=bands)


# ------------------------------------------------------------------ flying
def worker(w, tasks, students, renderer, q):
    import torch
    torch.set_num_threads(1)
    from v6_drop.envs import landing_env as E
    from v6_drop.scripts.dice_exam import fly_dice, sigma_norm
    from v6_drop.scripts.eval_student import load
    models = {name: load(ck, "cpu")[0] for name, ck, _ in students}
    shields = {name: sh for name, _, sh in students}
    sig_n = sigma_norm()
    env = E.LandingEnv(renderer=renderer, seed=0)
    rows, t0 = [], time.time()
    try:
        for j, (name, seed) in enumerate(tasks):
            env.drop_shield = shields[name]                    # per-flight switch, same world
            row = fly_dice(env, models[name], seed, "clean", sig_n)
            row.update(student=name, shield=shields[name])
            rows.append(row)
            if (j + 1) % 25 == 0 or j + 1 == len(tasks):
                print(f"[w{w}] {j + 1}/{len(tasks)} flights ({(time.time() - t0) / (j + 1):.1f} s/flight)",
                      flush=True)
    finally:
        env.close()
    q.put(rows)


def parse_students(items, runs_root):
    """['name=path[:shield]', ...] -> list; default = STUDENTS under runs_root."""
    if not items:
        return [(n, os.path.join(runs_root, p), s) for n, p, s in STUDENTS]
    out = []
    for it in items:
        name, rest = it.split("=", 1)
        if ":" in rest:
            path, sh = rest.rsplit(":", 1)
            out.append((name, path, float(sh)))
        else:
            out.append((name, rest, None))
    return out


def report(by, students, pairs, out, seeds):
    names = [n for n, _, _ in students]
    D = {n: describe(by[n]) for n in names}
    f = lambda x, s="{:.2f}": "-" if x is None or (isinstance(x, float) and math.isnan(x)) else s.format(x)
    L = [f"FINAL EVALUATION: {len(names)} students x {len(seeds)} fresh seeds ({seeds[0]}..{seeds[-1]}), "
         f"clean flights, CPU, same seeds for all", ""]
    L.append(f"{'student':18s} {'shield':6s} {'safe % [95% CI]':22s} {'drop % [95% CI]':22s} {'hard':>4s} "
             f"{'timeout':>7s} {'other':>5s} {'impact mean/p95':>15s} {'offset cm':>9s} {'time s':>6s}")
    for n, ck, sh in students:
        d = D[n]
        o = d["outcomes"]
        other = d["n"] - d["safe"] - o.get("hard_landing", 0) - o.get("timeout", 0)
        L.append(f"{n:18s} {('-' if sh is None else str(sh)):6s} "
                 f"{100 * d['safe'] / d['n']:5.1f} [{d['safe_ci'][0]:5.1f}-{d['safe_ci'][1]:5.1f}]   "
                 f"{100 * d['drop'] / d['n']:5.1f} [{d['drop_ci'][0]:5.1f}-{d['drop_ci'][1]:5.1f}]   "
                 f"{o.get('hard_landing', 0):4d} {o.get('timeout', 0):7d} {other:5d} "
                 f"{f(d['impact_mean'])}/{f(d['impact_p95'])}      {f(d['offset_mean'], '{:.1f}'):>6s} "
                 f"{d['time_mean']:6.1f}")
    L.append("")
    for n, ck, sh in students:
        d = D[n]
        L += [f"== {n}  ({os.path.abspath(ck)}; shield {sh})",
              f"safe {d['safe']}/{d['n']} (CI {d['safe_ci'][0]:.1f}-{d['safe_ci'][1]:.1f}%)"
              + ("" if d["fail_bound"] is None else f"; 0 failures -> failure rate < {d['fail_bound']:.2f}% "
                                                     f"(rule of three, 95%)"),
              f"by drop {d['drop']} (CI {d['drop_ci'][0]:.1f}-{d['drop_ci'][1]:.1f}%), by auto-cut {d['auto_cut']}; "
              f"outcomes {d['outcomes']}",
              f"impact (safe) mean {f(d['impact_mean'])}, p95 {f(d['impact_p95'])}, max {f(d['impact_max'])} m/s; "
              f"offset mean {f(d['offset_mean'], '{:.1f}')}, p95 {f(d['offset_p95'], '{:.1f}')} cm",
              f"time mean {d['time_mean']:.2f} s (safe flights {f(d['time_safe'])} s); shield vetoes {d['vetoes']}",
              "by pad speed (m/s): " + ", ".join(f"{b['band']}: safe {b['safe']}/{b['n']}, drop {b['drop']}"
                                                  for b in d["bands"]), ""]
    L.append("PAIRED COMPARISONS (same seeds; McNemar exact: only seeds where the two disagree count)")
    for a, b in pairs:
        if a not in by or b not in by:
            continue
        ra = sorted(by[a], key=lambda r: r["seed"])
        rb = sorted(by[b], key=lambda r: r["seed"])
        assert [r["seed"] for r in ra] == [r["seed"] for r in rb]
        s = mcnemar([r["outcome"] == "success" for r in ra], [r["outcome"] == "success" for r in rb])
        dd = mcnemar([r["outcome"] == "success" and r["reason"] == "drop" for r in ra],
                     [r["outcome"] == "success" and r["reason"] == "drop" for r in rb])
        t = paired_diff([r["time_s"] for r in ra], [r["time_s"] for r in rb])
        L.append(f"{a} -> {b}: safe only-{a} {s[0]}, only-{b} {s[1]}, p = {s[2]:.2g} | "
                 f"drop only-{a} {dd[0]}, only-{b} {dd[1]}, p = {dd[2]:.2g} | "
                 f"time {b} - {a} = {t[0]:+.2f} s [{t[1]:+.2f}, {t[2]:+.2f}]")
    txt = "\n".join(L) + "\n"
    open(os.path.join(out, "summary.txt"), "w").write(txt)
    print(txt, flush=True)
    try:
        figure(D, names, os.path.join(out, "compare.png"))
    except Exception as e:                                    # the numbers matter more than the picture
        print(f"[final] figure failed: {e}", flush=True)
    return D


def figure(D, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(names))
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5), dpi=120, gridspec_kw=dict(wspace=0.25))
    for k, (key, ci, col, lab) in enumerate([("safe", "safe_ci", "#1f4e9c", "safe %"),
                                             ("drop", "drop_ci", "#2e8b57", "landed by drop %")]):
        v = [100 * D[n][key] / D[n]["n"] for n in names]
        lo = [v[i] - D[n][ci][0] for i, n in enumerate(names)]
        hi = [D[n][ci][1] - v[i] for i, n in enumerate(names)]
        a.bar(x + (k - 0.5) * 0.38, v, 0.38, yerr=[lo, hi], capsize=3, color=col, label=lab)
    a.set_xticks(x)
    a.set_xticklabels(names, rotation=25, ha="right")
    a.set_ylim(0, 105)
    a.set_ylabel("% of flights (95% CI)")
    a.legend(frameon=False)
    a.set_title("Safe and drop landings, fresh seeds", loc="left", weight="bold")
    b.bar(x, [D[n]["time_mean"] for n in names], 0.6, color="#e67e22")
    b.set_xticks(x)
    b.set_xticklabels(names, rotation=25, ha="right")
    b.set_ylabel("mean flight time (s)")
    b.set_title("Flight time (timeouts count as 30 s)", loc="left", weight="bold")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=9500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--renderer", choices=["egl", "tiny"], default="egl")
    ap.add_argument("--runs-root", default=RUNS)
    ap.add_argument("--out-root", default=OUT)
    ap.add_argument("--students", nargs="*", default=None, help="name=ckpt[:shield] (default: the 7 of E1)")
    ap.add_argument("--force", action="store_true", help="only to redo a CRASHED run")
    a = ap.parse_args(argv)
    students = parse_students(a.students, a.runs_root)
    missing = [ck for _, ck, _ in students if not os.path.exists(ck)]
    if missing:
        raise SystemExit("missing checkpoints:\n  " + "\n  ".join(missing))
    out = os.path.join(a.out_root, a.tag)
    if os.path.exists(os.path.join(out, "summary.txt")) and not a.force:
        raise SystemExit(f"{out} already has a final result; the test is run ONCE (8.1). "
                         f"Use --force only to redo a crashed run.")
    os.makedirs(out, exist_ok=True)
    seeds = list(range(a.seed0, a.seed0 + a.n))
    tasks = [(name, s) for s in seeds for name, _, _ in students]
    print(f"[final] {len(students)} students x {a.n} seeds = {len(tasks)} flights, {a.workers} workers", flush=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(w, tasks[w::a.workers], students, a.renderer, q))
             for w in range(a.workers)]
    for p in procs:
        p.start()
    rows = []
    for _ in procs:
        rows += q.get()
    for p in procs:
        p.join()
    print(f"[final] {len(rows)} flights in {(time.time() - t0) / 60:.1f} min", flush=True)
    by = {n: sorted([r for r in rows if r["student"] == n], key=lambda r: r["seed"]) for n, _, _ in students}
    for n, rs in by.items():
        with open(os.path.join(out, f"flights_{n}.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rs[0]))
            wr.writeheader()
            wr.writerows(rs)
    report(by, students, PAIRS, out, seeds)


if __name__ == "__main__":
    main()
