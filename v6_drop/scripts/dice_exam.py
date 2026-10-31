"""v6_drop Step 7b: the DICE EXAM (decided 30 Oct, P4).

Why: PPO flies with dice (7.1): move = the student's mean + small noise, drop = a coin with
the student's drop chance p. The real exam (eval_student) has no dice. Before any PPO
training we measure what the dice alone do to the frozen student (dagger_r4): how many
random early drops, hard landings and other failures. This is PPO's starting point.

Four modes, each on the same dev seeds (9000..):
  clean  no dice (= the real exam: mean move, drop if p > 0.5). Should repeat the r4 exam.
  move   move noise only; drop as in clean
  drop   drop coin only; move as in clean
  both   move noise + drop coin (exactly what PPO will fly)
Move noise width (P4): vx, vy 0.2 m/s; vz 0.05 m/s; yaw rate 3 deg/s (Gaussian, then clipped
to the action range [-1, 1] like the env does). The dice of flight (seed, mode) come from
their own random stream, so every run is reproducible.

Per flight it also records the drop chance p at every step:
  p_far   steps with the feet > 0.3 m above the pad top (far from any drop)
  coin_risk = 1 - prod(1 - p) over the steps where p <= 0.5: the chance that a coin would
            have dropped somewhere the clean rule would not (computed in every mode)
  coin_drop = the flight ended by a drop with p <= 0.5 (only possible with the coin)

Shield (Step 7e): --shield L flies with the camera-based drop shield (envs/landing_env.py,
predicted impact <= L m/s). Rows then also count the blocked drop presses ('vetoes'); the
output folder gets the suffix _shield<L x 100>, e.g. dice_exam_100_shield80.
--veto-cost c (Step 7f) only changes the RETURN (c per blocked press); flights are identical.

Parallel: W worker processes, one EGL simulator each (B5 rule), student on the CPU with one
thread per worker (as dagger_collect). The summary also gives decisions per second (P7).

Run (from the repo root; performance mode first):
  nohup python -m v6_drop.scripts.dice_exam --ckpt v6_drop/results/runs/dagger_r4/last.pt \\
        --n 100 --workers 6 > v6_drop/results/runs/dagger_r4/dice_exam.log 2>&1 &
Watch:  tail -f v6_drop/results/runs/dagger_r4/dice_exam.log | grep --line-buffered -E "^\\[(w|dice)"
Stop:   pkill -f "v6_drop.scripts.dice_exam"
Output: <run folder>/dice_exam_<n>/flights.csv and summary.txt (git-ignored); files of a
        checkpoint other than last.pt go to <run folder>/<stem>_dice_exam_<n>/.
"""
import argparse
import csv
import multiprocessing as mp
import os
import time

import numpy as np

MODES = ("clean", "move", "drop", "both")
SIGMA = (0.2, 0.2, 0.05, 3.0)          # vx, vy [m/s], vz [m/s], yaw rate [deg/s]  (P4)
FAR_FEET = 0.3                         # m: 'far' steps for the drop-chance statistics
SEED0 = 9000                           # dev exam seeds


def sigma_norm(sigma=SIGMA):
    """Noise width in the policy's units (action / ACT_SCALE)."""
    from v6_drop.envs.landing_env import ACT_SCALE
    s = np.array([sigma[0], sigma[1], sigma[2], np.radians(sigma[3])], float)
    return s / ACT_SCALE


def dice_rng(seed, mode):
    return np.random.default_rng([int(seed), MODES.index(mode), 7])


def choose(mean, p, mode, sig_n, rng):
    """The action one step of a flight takes. Returns (move float32 (4,), drop int, coin bool).
    coin = the drop came from the coin while the clean rule (p > 0.5) would not drop."""
    assert mode in MODES
    move = np.asarray(mean, float)
    if mode in ("move", "both"):
        move = move + rng.normal(0.0, 1.0, 4) * sig_n
    move = np.clip(move, -1.0, 1.0).astype(np.float32)
    if mode in ("drop", "both"):
        drop = bool(rng.random() < p)
    else:
        drop = bool(p > 0.5)
    return move, int(drop), bool(drop and p <= 0.5)


def fly_dice(env, model, seed, mode, sig_n):
    """One flight of the dice exam. Returns a row dict. A drop the shield blocks does not
    count as a drop (feet/chance at the drop are those of the drop that happened)."""
    import torch
    from v6_drop.envs import lmf2_params as P
    from v6_drop.envs.platform_s1 import PAD_TOP_Z
    rng = dice_rng(seed, mode)
    obs, info0 = env.reset(seed=seed)
    hc, steps, ret = None, 0, 0.0
    p_far, log_keep, feet_drop, coin, p_last, vetoes = [], 0.0, None, False, 0.0, 0
    while True:
        s = env.world.true_state()
        hf = float(s["pos"][2] - PAD_TOP_Z - P.REST_HEIGHT)
        img = torch.as_tensor(obs["image"]).reshape(1, 1, *obs["image"].shape)
        st = torch.as_tensor(obs["state"], dtype=torch.float32).reshape(1, 1, -1)
        with torch.no_grad():
            mean, logit, _, hc = model(img, st, hc)
        p = float(torch.sigmoid(logit).item())
        move, drop, coin_now = choose(mean[0, 0].numpy(), p, mode, sig_n, rng)
        if hf > FAR_FEET:
            p_far.append(p)
        if p <= 0.5:
            log_keep += np.log1p(-min(p, 1 - 1e-12))
        obs, r, term, trunc, info = env.step({"move": move, "drop": drop})
        if info.get("shield_veto"):
            vetoes += 1
        elif drop:
            feet_drop, p_last, coin = hf, p, coin_now
        ret += r
        steps += 1
        if term or trunc:
            break
    td = info.get("touchdown") or {}
    pf = np.array(p_far) if p_far else np.zeros(1)
    return dict(seed=seed, mode=mode, pad_speed=round(info0["pad_speed"], 3), outcome=info["outcome"],
                reason=td.get("reason"),
                impact=None if td.get("impact") is None else round(td["impact"], 3),
                offset_cm=None if not td.get("land_xy") else round(100 * float(np.linalg.norm(td["land_xy"])), 1),
                feet_at_drop_cm=None if feet_drop is None else round(100 * feet_drop, 1),
                drop_chance_at_drop=None if feet_drop is None else round(p_last, 4),
                coin_drop=coin, steps=steps, time_s=round(steps * P.POLICY_DT, 1), ret=round(ret, 3),
                p_far_median=round(float(np.median(pf)), 6), p_far_max=round(float(pf.max()), 4),
                coin_risk=round(float(1.0 - np.exp(log_keep)), 4), vetoes=vetoes)


def worker(w, tasks, ckpt, renderer, sigma, q, shield=None, veto_cost=0.0):
    import torch
    torch.set_num_threads(1)
    from v6_drop.envs import landing_env as E
    from v6_drop.scripts.eval_student import load
    model, _ = load(ckpt, "cpu")
    sig_n = sigma_norm(sigma)
    env = E.LandingEnv(renderer=renderer, seed=0, drop_shield=shield, veto_cost=veto_cost)
    rows, t0 = [], time.time()
    try:
        for j, (mode, seed) in enumerate(tasks):
            rows.append(fly_dice(env, model, seed, mode, sig_n))
            if (j + 1) % 10 == 0 or j + 1 == len(tasks):
                safe = sum(r["outcome"] == "success" for r in rows)
                print(f"[w{w}] {j + 1}/{len(tasks)} flights, safe {safe}, "
                      f"{(time.time() - t0) / (j + 1):.1f} s/flight", flush=True)
    finally:
        env.close()
    q.put(rows)


def wilson(k, n, z=1.96):
    c = (k + z * z / 2) / (n + z * z)
    hw = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return 100 * (c - hw), 100 * (c + hw)


def _f(x, fmt="{:.2f}"):
    return "-" if x is None else fmt.format(x)


def summarize(rows, out, ckpt, sigma, wall_s, workers, shield=None, veto_cost=0.0):
    rows = sorted(rows, key=lambda r: (MODES.index(r["mode"]), r["seed"]))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "flights.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    steps = sum(r["steps"] for r in rows)
    lines = [f"Dice exam: {os.path.abspath(ckpt)}",
             f"move noise sigma: vx {sigma[0]}, vy {sigma[1]}, vz {sigma[2]} m/s, yaw rate {sigma[3]} deg/s",
             "drop shield: " + ("off" if shield is None else f"predicted impact <= {shield} m/s (camera + IMU)")
             + f"; press cost {veto_cost} per blocked press",
             f"speed: {len(rows)} flights, {steps:,} decisions in {wall_s / 60:.1f} min with {workers} workers "
             f"= {steps / wall_s:.0f} decisions/s", ""]
    for mode in MODES:
        rs = [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        n = len(rs)
        ok = [r for r in rs if r["outcome"] == "success"]
        lo, hi = wilson(len(ok), n)
        outs = {}
        for r in rs:
            outs[r["outcome"]] = outs.get(r["outcome"], 0) + 1
        drops = [r for r in rs if r["reason"] == "drop"]
        coin = [r for r in rs if r["coin_drop"]]
        bands = []
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 4.01)):
            rb = [r for r in rs if a <= r["pad_speed"] < b]
            bands.append(f"{a}-{int(b)} {sum(r['outcome'] == 'success' for r in rb)}/{len(rb)}")
        imp = [r["impact"] for r in ok if r["impact"] is not None]
        fd = [r["feet_at_drop_cm"] for r in drops]
        lines += [f"== {mode} ({n} flights, seeds {rs[0]['seed']}..{rs[-1]['seed']})",
                  f"safe: {len(ok)}/{n} = {100 * len(ok) / n:.1f}% (95% CI {lo:.1f}-{hi:.1f}%); "
                  f"by drop {sum(r['reason'] == 'drop' for r in ok)}, by auto-cut "
                  f"{sum(r['reason'] == 'auto_cut' for r in ok)}",
                  f"outcomes: {outs}",
                  f"by pad speed (m/s): {', '.join(bands)}",
                  f"drops: {len(drops)} (coin drops with p <= 0.5: {len(coin)}, of them safe "
                  f"{sum(r['outcome'] == 'success' for r in coin)}); feet at the drop median "
                  f"{_f(np.median(fd) if fd else None, '{:.1f}')} cm, max {_f(max(fd) if fd else None, '{:.1f}')} cm",
                  f"impact (safe): mean {_f(np.mean(imp) if imp else None)}, max {_f(max(imp) if imp else None)} m/s",
                  f"shield vetoes (blocked drop presses): {sum(r['vetoes'] for r in rs)} in "
                  f"{sum(r['vetoes'] > 0 for r in rs)} flights",
                  f"return: mean {np.mean([r['ret'] for r in rs]):.2f}; time mean {np.mean([r['time_s'] for r in rs]):.1f} s",
                  f"drop chance far from the pad (feet > {FAR_FEET} m): median over flights of the per-flight "
                  f"median {np.median([r['p_far_median'] for r in rs]):.5f}, highest {max(r['p_far_max'] for r in rs):.4f}",
                  f"coin risk per flight (chance of >= 1 coin drop where clean would not drop): mean "
                  f"{np.mean([r['coin_risk'] for r in rs]):.3f}, median {np.median([r['coin_risk'] for r in rs]):.3f}",
                  ""]
    txt = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(txt)
    print(txt, flush=True)


def out_dir(ckpt, n, shield=None):
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    pre = "" if stem == "last" else stem + "_"
    suf = "" if shield is None else f"_shield{round(100 * shield)}"
    return os.path.join(os.path.dirname(os.path.abspath(ckpt)), f"{pre}dice_exam_{n}{suf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100, help="flights per mode (seeds 9000..)")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--renderer", choices=["egl", "tiny"], default="egl")
    ap.add_argument("--sigma", type=float, nargs=4, default=list(SIGMA),
                    help="vx vy vz [m/s] yaw-rate [deg/s]")
    ap.add_argument("--shield", type=float, default=None, help="drop shield limit, m/s (Step 7e: 0.8)")
    ap.add_argument("--veto-cost", type=float, default=0.0, help="cost per blocked drop press (Step 7f: 0.05)")
    a = ap.parse_args()
    tasks = [(m, SEED0 + i) for m in a.modes for i in range(a.n)]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(w, tasks[w::a.workers], a.ckpt, a.renderer, a.sigma, q, a.shield,
                                                       a.veto_cost))
             for w in range(a.workers)]
    for pr in procs:
        pr.start()
    rows = []
    for _ in procs:
        rows += q.get()
    for pr in procs:
        pr.join()
    wall = time.time() - t0
    print(f"[dice] {len(rows)} flights in {wall / 60:.1f} min", flush=True)
    summarize(rows, out_dir(a.ckpt, a.n, a.shield), a.ckpt, a.sigma, wall, a.workers, a.shield, a.veto_cost)


if __name__ == "__main__":
    main()
