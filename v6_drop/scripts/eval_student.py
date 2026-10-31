"""v6_drop Step 6c: the student's EXAM. It flies alone (camera + own sensors only, its own
memory) on the fixed TEST seeds 9000.. (never used for training). Deterministic: the mean
move, and drop when its drop chance is above 0.5.

Run (from the repo root; performance mode first):
  python -m v6_drop.scripts.eval_student --ckpt v6_drop/results/runs/bc256/best.pt --n 100
  python -m v6_drop.scripts.eval_student --ckpt ... --videos 9000 9001 9002 9003
  (8.3) --shield 0.8 flies with the camera drop shield; --seed0 9500 uses the fresh test seeds;
  --pick-from results/final_eval/test9500/flights_<student>.csv --pick drops|failures chooses
  the video seeds (one drop per pad-speed band + the fastest pad, or up to 4 failures).
  Videos show each drop press (ALLOWED / BLOCKED by the shield) and the motors-off fall in
  slow motion (x10), then the 2 s landing check.
Output: v6_drop/results/runs/<run>/exam_<n>.csv, exam_<n>_summary.txt, exam_paths.png and
        (with --videos) student_<seed>.mp4/.png. It prints nothing until each 10 episodes.
        (6d) Files of last.pt are named exam_<n>*, of any other checkpoint <stem>_exam_<n>*,
        so exams of best.pt and last.pt no longer overwrite each other.
Drop-gate lines (6d, 29 Oct): per flight, the longest run of steps meeting the teacher's
drop gate (true state, teacher's constants), and the longest stay in the 2-4 cm window
with the sinking speed there. These show progress towards DROPPING before drops happen.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.experts import teacher as T
from v6_drop.learn.student import Student
from v6_drop.scripts import show_teacher as ST


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = Student(ck["size"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def snapshot(env, phase, t):
    """The numbers shown in a video frame / stored per step (true state, for display)."""
    s = env.world.true_state()
    rel = s["pad_center"] - s["pos"]
    return dict(t=t, pos=s["pos"].copy(), pad=s["pad_center"].copy(),
                rel_pad=env.world.plat.world_to_pad([s["pos"]])[0],
                hf=float(s["pos"][2] - PAD_TOP_Z - P.REST_HEIGHT), d=float(np.hypot(rel[0], rel[1])),
                phase=phase, flag="",
                vrel=float(np.linalg.norm((s["v"] - s["pad_v"])[:2])),
                vz_rel=float((s["v"] - s["pad_v"])[2]),
                tilt=float(np.degrees(np.arccos(np.clip(s["R"][2, 2], -1, 1)))))


SLOWMO_STEPS = 25              # 8.3: first 0.25 s after the motor cut, one frame per physics step


def drop_note(env, act):
    """8.3: what happens to a drop press (shown in the video)."""
    if not act["drop"]:
        return None, None
    if env.drop_shield is None:
        return "DROP PRESSED -> motors off", ST.GREEN
    ok, pred, _ = env.shield_check()
    if pred is None:
        return "DROP PRESSED -> BLOCKED by shield (no confident marker)", ST.RED
    if ok:
        return f"DROP PRESSED -> ALLOWED (predicted impact {pred:.2f} m/s)", ST.GREEN
    return f"DROP PRESSED -> BLOCKED by shield (predicted {pred:.2f} m/s > {env.drop_shield})", ST.RED


def fly(env, model, seed, device, frames=False, drop_rule=None):
    """One exam flight. drop_rule (6d diagnostic, scripts/oracle_drop.py): an object with
    reset() and __call__(rec) -> bool that may FORCE a drop from the true state; None = the
    student alone (the real exam). frames=True (8.3) also films the motors-off part: the
    first SLOWMO_STEPS physics steps one frame each (slow motion x10), then one frame per
    0.1 s of the 2 s landing check. Filming only renders; the flight is the same."""
    obs, info0 = env.reset(seed=seed)
    hc, log, pics = None, [], []
    if drop_rule is not None:
        drop_rule.reset()
    if frames:
        def hook(k):
            if k <= SLOWMO_STEPS or k % 10 == 0:
                slow = k <= SLOWMO_STEPS
                r = snapshot(env, "MOTORS OFF" + (" (slow motion x10)" if slow else ": 2 s landing check"),
                             env.world.t * P.POLICY_DT + k * P.PHYSICS_DT)
                r["note"], r["note_color"] = "motors off: falling onto the pad" if slow else None, ST.YELLOW
                pics.append(ST.compose_frame(env, r, outcome=None))
        env.world.frame_hook = hook
    try:
        while True:
            act, hc, p_drop = model.act(obs, hc, device)
            rec = snapshot(env, f"student (drop chance {100 * p_drop:.0f}%)", env.world.t * P.POLICY_DT)
            if drop_rule is not None and drop_rule(rec):
                act = dict(act, drop=1)
                rec["forced_drop"] = True
            if frames:
                rec["note"], rec["note_color"] = drop_note(env, act)
                pics.append(ST.compose_frame(env, rec, outcome=None))
            obs, r, term, trunc, info = env.step(act)
            rt = info.get("reward_terms") or {}
            rec["seen"] = rt.get("seen")
            log.append(rec)
            if term or trunc:
                break
    finally:
        env.world.frame_hook = None
    if frames:
        s = env.world.true_state()
        pics += [ST.compose_frame(env, dict(log[-1], pos=s["pos"].copy(), note=None), outcome=info)] * 20
    return log, info, info0, pics


def runs(flags):
    """Longest run of True in a sequence."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def gate_stats(log):
    """Longest run meeting the teacher's drop gate, longest stay in the 2-4 cm window, and
    the mean sinking speed while in the window (None if never there)."""
    ok = [(r["d"] < T.D_DROP and abs(r["hf"] - T.H_DROP) < T.DH_DROP and r["vrel"] < T.V_REL_DROP
           and abs(r["vz_rel"]) < T.V_REL_DROP and r["tilt"] < np.degrees(T.TILT_DROP)) for r in log]
    win = [abs(r["hf"] - T.H_DROP) < T.DH_DROP for r in log]
    sink = [-r["vz_rel"] for r, w in zip(log, win) if w]
    return runs(ok), runs(win), (float(np.mean(sink)) if sink else None)


def wilson(k, n, z=1.96):
    c = (k + z * z / 2) / (n + z * z)
    hw = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return 100 * (c - hw), 100 * (c + hw)


def exam(model, ckpt, n, device, seed0=9000, shield=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.dirname(os.path.abspath(ckpt))
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    pre = ("" if stem == "last" else stem + "_") + ("" if shield is None else f"shield{round(100 * shield)}_")
    pre += "" if seed0 == 9000 else f"seeds{seed0}_"               # 8.3: fresh seeds never overwrite dev files
    kw = {} if shield is None else dict(drop_shield=shield)      # no shield = the old call exactly
    env = E.LandingEnv(renderer="egl", seed=0, **kw)
    rows, paths, t0 = [], [], time.time()
    try:
        for i in range(n):
            log, info, info0, _ = fly(env, model, seed0 + i, device)
            td = info.get("touchdown") or {}
            rows.append(dict(seed=seed0 + i, pad_speed=round(info0["pad_speed"], 3), outcome=info["outcome"],
                             time_s=round(len(log) * P.POLICY_DT, 1), reason=td.get("reason"),
                             impact=None if td.get("impact") is None else round(td["impact"], 3),
                             offset_cm=None if not td.get("land_xy") else round(100 * float(np.linalg.norm(td["land_xy"])), 1)))
            g_run, w_stay, w_sink = gate_stats(log)
            rows[-1].update(gate_run=g_run, window_stay=w_stay,
                            window_sink=None if w_sink is None else round(w_sink, 3))
            paths.append((np.array([r["rel_pad"][0] for r in log]), np.array([r["hf"] for r in log]),
                          info["outcome"] == "success"))
            if (i + 1) % 10 == 0:
                k = sum(r["outcome"] == "success" for r in rows)
                print(f"[exam] {i + 1}/{n} flown, safe so far {k}/{i + 1} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        env.close()
    with open(os.path.join(out, f"{pre}exam_{n}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    k = sum(r["outcome"] == "success" for r in rows)
    lo, hi = wilson(k, n)
    outs = {}
    for r in rows:
        outs[r["outcome"]] = outs.get(r["outcome"], 0) + 1
    ok = [r for r in rows if r["outcome"] == "success"]
    lines = [f"Student exam: {os.path.abspath(ckpt)} on {n} test episodes (seeds {seed0}..{seed0 + n - 1})",
             f"safe landings: {k}/{n} = {100 * k / n:.1f}%  (95% CI {lo:.1f}-{hi:.1f}%)",
             f"outcomes: {outs}"]
    if ok:
        imp = np.array([r["impact"] for r in ok])
        off = np.array([r["offset_cm"] for r in ok])
        tim = np.array([r["time_s"] for r in ok])
        lines += [f"landed by drop: {sum(r['reason'] == 'drop' for r in ok)}, by auto-cut: "
                  f"{sum(r['reason'] == 'auto_cut' for r in ok)}",
                  f"impact [m/s] (safe ones): mean {imp.mean():.2f}, max {imp.max():.2f}",
                  f"distance from the centre [cm]: mean {off.mean():.1f}, max {off.max():.1f}",
                  f"time to land [s]: mean {tim.mean():.1f}, max {tim.max():.1f}"]
    by_speed = []
    for lo_v, hi_v in ((0, 1), (1, 2), (2, 3), (3, 4.01)):
        rs = [r for r in rows if lo_v <= r["pad_speed"] < hi_v]
        if rs:
            by_speed.append(f"{lo_v}-{int(hi_v)} m/s: {sum(r['outcome'] == 'success' for r in rs)}/{len(rs)}")
    lines.append("safe by pad speed: " + ", ".join(by_speed))
    reached = [r for r in rows if r["window_stay"] > 0]
    lines.append(f"drop gate held >= {T.HOLD_STEPS} steps: {sum(r['gate_run'] >= T.HOLD_STEPS for r in rows)}/{n} flights; "
                 f"2-4 cm window reached in {len(reached)}/{n}")
    if reached:
        lines.append(f"  in the window: longest stay median {np.median([r['window_stay'] for r in reached]):.0f} "
                     f"steps, sinking speed median {np.median([r['window_sink'] for r in reached]):.2f} m/s "
                     f"(teacher: 5 steps, 0.01 m/s)")
    txt = "\n".join(lines)
    open(os.path.join(out, f"{pre}exam_{n}_summary.txt"), "w").write(txt + "\n")
    print(txt, flush=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between([-0.75, 0.75], -0.08, 0, color="gray", alpha=0.6, label="pad")
    for x, h, good in paths:
        ax.plot(x, h, "-", color="b" if good else "r", lw=0.6, alpha=0.35)
    ax.plot([], [], "b-", label="safe")
    ax.plot([], [], "r-", label="not safe")
    ax.set_xlabel("along the pad's travel [m] (negative = behind)")
    ax.set_ylabel("feet height above the pad [m]")
    ax.set_title(f"Student: {n} test flights relative to the moving pad ({100 * k / n:.0f}% safe)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{pre}exam_paths.png"), dpi=110)
    plt.close(fig)


def pick_seeds(csv_path, kind):
    """8.3: choose video seeds from a final_eval flights_<student>.csv.
    kind 'drops': the first DROP landing in each pad-speed band + the fastest pad;
    kind 'failures': up to 4 flights that were not safe."""
    rows = list(csv.DictReader(open(csv_path)))
    if kind == "failures":
        return [int(r["seed"]) for r in rows if r["outcome"] != "success"][:4]
    seeds = []
    for lo_v, hi_v in ((0, 1), (1, 2), (2, 3), (3, 4.01)):
        for r in rows:
            if lo_v <= float(r["pad_speed"]) < hi_v and r["outcome"] == "success" and r["reason"] == "drop":
                seeds.append(int(r["seed"]))
                break
    fast = max(rows, key=lambda r: float(r["pad_speed"]))
    if int(fast["seed"]) not in seeds:
        seeds.append(int(fast["seed"]))
    return seeds


def videos(model, ckpt, seeds, device, shield=None):
    out = os.path.dirname(os.path.abspath(ckpt))
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    pre = ("" if stem == "last" else stem + "_") + ("" if shield is None else f"shield{round(100 * shield)}_")
    kw = {} if shield is None else dict(drop_shield=shield)
    env = E.LandingEnv(renderer="egl", seed=0, **kw)
    try:
        for seed in seeds:
            log, info, info0, frames = fly(env, model, seed, device, frames=True)
            base = os.path.join(out, f"{pre}student_{seed}")
            ST.write_video(frames, base + ".mp4")
            ST.plot_episode(log, info, seed, info0["pad_speed"], base + ".png", title="Student")
            print(f"[video] seed {seed}: pad {info0['pad_speed']:.2f} m/s, {info['outcome']} -> {base}.mp4/.png",
                  flush=True)
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--videos", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shield", type=float, default=None, help="drop shield limit, m/s (Step 7e: 0.8)")
    ap.add_argument("--seed0", type=int, default=9000, help="first exam seed (8.3: 9500 = fresh test seeds)")
    ap.add_argument("--pick-from", default=None, help="8.3: final_eval flights_<student>.csv to choose video seeds")
    ap.add_argument("--pick", choices=["drops", "failures"], default="drops")
    a = ap.parse_args()
    model, ck = load(a.ckpt, a.device)
    when = f"epoch {ck['epoch']}" if "epoch" in ck else f"PPO round {ck.get('round')}"   # 7e: PPO files
    print(f"[exam] loaded {a.ckpt} ({when}, size {ck['size']})", flush=True)
    if a.pick_from:
        a.videos = pick_seeds(a.pick_from, a.pick)
        print(f"[video] picked seeds ({a.pick}): {a.videos}", flush=True)
    if a.videos:
        videos(model, a.ckpt, a.videos, a.device, shield=a.shield)
    else:
        exam(model, a.ckpt, a.n, a.device, seed0=a.seed0, shield=a.shield)


if __name__ == "__main__":
    main()
