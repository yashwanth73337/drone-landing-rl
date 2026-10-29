"""v6_drop Step 6a: SEE the teacher fly (videos, trajectory plots, summary table).

Two modes (run from the repo root):

  1. Videos + plots of a few episodes (renders the cameras, ~20-40 s per episode):
       python -m v6_drop.scripts.show_teacher videos --seeds 9000 9001 9002 9003
     -> v6_drop/results/teacher/ep_<seed>.mp4   left: chase camera, right: the drone's
                                                own camera with detected markers drawn
        v6_drop/results/teacher/ep_<seed>.png   top view + side view + time plots

  3. One video per TROUBLE kind (Step 6b), on the training episode record_demos records:
       python -m v6_drop.scripts.show_teacher troubles
     -> v6_drop/results/teacher/trouble_<kind>_<seed>.mp4/.png  (red "TROUBLE: ..." text)

  2. Many episodes without pictures (fast) -> success table + all paths in one figure:
       python -m v6_drop.scripts.show_teacher batch --n 500
     -> v6_drop/results/teacher/batch.csv, batch_summary.txt, batch_paths.png

Seeds 9000+i are the fixed test seeds (same episodes every time). The 'results/' folder is
ignored by git. Videos are written as MPEG-4 (mp4v); if ffmpeg is installed they are also
converted to H.264 (plays in browsers and PowerPoint).
"""
import argparse
import csv
import os
import shutil
import subprocess
import time

import cv2
import numpy as np
import pybullet as p

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs import pad_board as PB
from v6_drop.envs.platform_s1 import PAD_SIZE, PAD_TOP_Z
from v6_drop.experts.teacher import K_BACK, Teacher

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "teacher")
CHASE_W, CHASE_H = 512, 320
GREEN, RED, WHITE, YELLOW = (60, 200, 60), (60, 60, 230), (255, 255, 255), (0, 220, 255)


# ---------------------------------------------------------------- one episode
def run_episode(env, teacher, seed, frames=False, trouble=None):
    """Fly one episode with the teacher. trouble: None or a troubles.KINDS name (Step 6b;
    the trouble's random numbers come from the seed, as in record_demos).
    Returns (log list, final info, reset info, frames list)."""
    obs, info0 = env.reset(seed=seed)
    teacher.reset()
    tr_maker = None
    if trouble is not None:
        from v6_drop.experts.troubles import Troubles, pick_kind
        rng = np.random.default_rng(seed)
        pick_kind(rng)                         # same draw order as record_demos
        tr_maker = Troubles()
        tr_maker.reset(trouble, rng)
    log, pics, seen = [], [], True
    while True:
        s = env.world.true_state()
        act, ti = teacher.act(env)
        flag = ""
        if tr_maker is not None:
            act, flag = tr_maker.apply(env, act, seen)
        rec = dict(t=env.world.t * P.POLICY_DT, pos=s["pos"].copy(), pad=s["pad_center"].copy(),
                   rel_pad=env.world.plat.world_to_pad([s["pos"]])[0],
                   hf=ti["hf"], d=ti["d"], phase=ti["phase"], flag=flag,
                   vrel=float(np.linalg.norm((s["v"] - s["pad_v"])[:2])),
                   tilt=float(np.degrees(np.arccos(np.clip(s["R"][2, 2], -1, 1)))))
        if frames:
            pics.append(compose_frame(env, rec, outcome=None))
        obs, r, term, trunc, info = env.step(act)
        rt = info.get("reward_terms") or {}
        rec["seen"] = rt.get("seen")
        if rt.get("seen") is not None:
            seen = bool(rt["seen"])
        log.append(rec)
        if term or trunc:
            break
    if frames:                                  # after the 2 s landing check: hold the result
        s = env.world.true_state()
        last = dict(log[-1], pos=s["pos"].copy(), pad=s["pad_center"].copy())
        end_frame = compose_frame(env, last, outcome=info)
        pics += [end_frame] * 20
    return log, info, info0, pics


# ---------------------------------------------------------------- pictures
def chase_image(env):
    """A film camera 2.2 m behind and 1.0 m above the drone, looking ahead and down."""
    w = env.world
    s = w.true_state()
    u = np.array([np.cos(w.plat.psi), np.sin(w.plat.psi), 0.0])
    eye = s["pos"] - 2.2 * u + np.array([0, 0, 1.0])
    target = s["pos"] + 0.8 * u + np.array([0, 0, -0.4])
    view = p.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(60, CHASE_W / CHASE_H, 0.05, 200)
    _, _, rgba, _, _ = p.getCameraImage(CHASE_W, CHASE_H, view, proj,
                                        renderer=w.cam._pflag, flags=p.ER_NO_SEGMENTATION_MASK,
                                        physicsClientId=w.cid)
    rgba = np.asarray(rgba, np.uint8).reshape(CHASE_H, CHASE_W, 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def drone_view(env):
    img = env.world.render()
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    corners, ids, _ = env.world.det.detectMarkers(img)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(bgr, corners, ids, borderColor=(0, 255, 0))
    return bgr, (0 if ids is None else len(ids))


def put(img, text, y, color=WHITE, scale=0.5):
    """Text on a dark box (readable after video compression)."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x0, y0, x1, y1 = 4, y - h - 4, 4 + w + 8, y + base + 2
    roi = img[max(y0, 0):y1, x0:min(x1, img.shape[1])]
    roi[:] = (roi * 0.25).astype(np.uint8)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def compose_frame(env, rec, outcome):
    left = chase_image(env)
    right, n_ids = drone_view(env)
    put(left, f"t = {rec['t']:4.1f} s   phase: {rec['phase']}", 20, YELLOW)
    put(left, f"feet above pad {rec['hf']:5.2f} m   to centre {rec['d']:5.2f} m", 42)
    put(left, f"pad {env.world.plat.v:3.1f} m/s   rel. speed {rec['vrel']:4.2f} m/s   tilt {rec['tilt']:4.1f} deg", 64)
    if rec.get("flag"):
        put(left, f"TROUBLE: {rec['flag']}", CHASE_H - 14, RED, 0.7)
    put(right, f"drone camera   markers seen: {n_ids}", 20, GREEN if n_ids else RED)
    if outcome is not None:
        o = outcome["outcome"]
        td = outcome.get("touchdown") or {}
        col = GREEN if o == "success" else RED
        put(left, f"RESULT: {o.upper()}", 110, col, 0.9)
        if td.get("impact") is not None:
            put(left, f"impact {td['impact']:.2f} m/s ({td['reason']})", 140, col, 0.6)
            r = np.linalg.norm(td["land_xy"]) if td.get("land_xy") else np.nan
            put(left, f"landed {r * 100:.1f} cm from the centre", 165, col, 0.6)
    return np.hstack([left, right])


def write_video(frames, path_mp4):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    if shutil.which("ffmpeg"):                    # H.264 copy for browsers / PowerPoint
        h264 = path_mp4.replace(".mp4", "_h264.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path_mp4, "-c:v", "libx264",
                        "-preset", "veryfast", "-pix_fmt", "yuv420p", h264], check=False)
        if os.path.exists(h264):
            os.replace(h264, path_mp4)


def plot_episode(log, info, seed, pad_speed, path_png, title="Teacher"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = np.array([r["pos"] for r in log])
    pad = np.array([r["pad"] for r in log])
    rel = np.array([r["rel_pad"] for r in log])
    hf = np.array([r["hf"] for r in log])
    t = np.array([r["t"] for r in log])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].plot(pad[:, 0], pad[:, 1], "k-", lw=3, alpha=0.4, label="pad centre")
    ax[0].plot(pos[:, 0], pos[:, 1], "b-", label="drone")
    ax[0].plot(pos[0, 0], pos[0, 1], "bo", label="start")
    ax[0].plot(pos[-1, 0], pos[-1, 1], "r*", ms=14, label="drop / touchdown")
    ax[0].set_title("Top view (world)")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[0].axis("equal")
    ax[0].legend(fontsize=8)
    hmax = max(hf.max(), 1.0)
    xs = np.linspace(-hmax * 1.2, hmax * 0.4, 50)
    ax[1].fill_between([-PAD_SIZE / 2, PAD_SIZE / 2], -0.08, 0, color="gray", alpha=0.6, label="pad")
    ax[1].plot(xs, np.abs(xs), "g--", lw=1, label="45 deg reward line")
    hh = np.linspace(0, hmax, 20)
    ax[1].plot(-K_BACK * hh, hh, "m:", lw=1.5, label="teacher approach line")
    ax[1].plot(rel[:, 0], hf, "b-", label="drone (feet)")
    ax[1].plot(rel[-1, 0], hf[-1], "r*", ms=14)
    ax[1].set_title("Side view, relative to the moving pad")
    ax[1].set_xlabel("along the pad's travel [m] (negative = behind)")
    ax[1].set_ylabel("feet height above the pad [m]")
    ax[1].set_xlim(min(rel[:, 0].min(), -1) - 0.3, max(rel[:, 0].max(), 1) + 0.3)
    ax[1].set_ylim(-0.1, hmax * 1.1)
    ax[1].legend(fontsize=8)
    ax[2].plot(t, hf, label="feet height [m]")
    ax[2].plot(t, [r["d"] for r in log], label="distance to centre [m]")
    ax[2].plot(t, [r["vrel"] for r in log], label="speed rel. to pad [m/s]")
    seen = [r["seen"] for r in log]
    blind = [ti for ti, sv in zip(t, seen) if sv is False]
    if blind:
        ax[2].plot(blind, np.zeros(len(blind)) - 0.15, "r|", ms=8, label="no marker seen")
    ax[2].set_title("Over time")
    ax[2].set_xlabel("time [s]")
    ax[2].legend(fontsize=8)
    td = info.get("touchdown") or {}
    imp = f", impact {td['impact']:.2f} m/s" if td.get("impact") is not None else ""
    fl = [ti for ti, r in zip([r["t"] for r in log], log) if r.get("flag")]
    if fl:
        ax[2].plot(fl, np.zeros(len(fl)) - 0.3, "m|", ms=8, label="trouble active")
        ax[2].legend(fontsize=8)
    fig.suptitle(f"{title}, seed {seed}, pad {pad_speed:.1f} m/s: {info['outcome']}{imp}")
    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- modes
def mode_videos(seeds):
    os.makedirs(OUT, exist_ok=True)
    env = E.LandingEnv(renderer="egl", seed=0)
    teacher = Teacher()
    try:
        for seed in seeds:
            t0 = time.time()
            log, info, info0, frames = run_episode(env, teacher, seed, frames=True)
            base = os.path.join(OUT, f"ep_{seed}")
            write_video(frames, base + ".mp4")
            plot_episode(log, info, seed, info0["pad_speed"], base + ".png")
            td = info.get("touchdown") or {}
            print(f"[video] seed {seed}: pad {info0['pad_speed']:.2f} m/s, {info['outcome']}, "
                  f"{len(log) * P.POLICY_DT:.1f} s, impact {td.get('impact', float('nan')):.2f} m/s "
                  f"-> {base}.mp4/.png ({time.time() - t0:.0f} s)", flush=True)
    finally:
        env.close()


def mode_troubles(kinds):
    """One video + plot per trouble kind, on the first TRAINING attempt of that kind
    (seed 100000 + e, the same episode record_demos records)."""
    from v6_drop.experts.troubles import pick_kind
    os.makedirs(OUT, exist_ok=True)
    env = E.LandingEnv(renderer="egl", seed=0)
    teacher = Teacher()
    try:
        for kind in kinds:
            e = next(e for e in range(10000) if pick_kind(np.random.default_rng(100000 + e)) == kind)
            seed = 100000 + e
            t0 = time.time()
            log, info, info0, frames = run_episode(env, teacher, seed, frames=True, trouble=kind)
            base = os.path.join(OUT, f"trouble_{kind}_{seed}")
            write_video(frames, base + ".mp4")
            plot_episode(log, info, seed, info0["pad_speed"], base + ".png", title=f"Teacher with trouble '{kind}'")
            ga = any(r["phase"] == "go-around" for r in log)
            print(f"[trouble] {kind}: seed {seed}, pad {info0['pad_speed']:.2f} m/s, {info['outcome']}, "
                  f"{len(log) * P.POLICY_DT:.1f} s, trouble steps {sum(bool(r['flag']) for r in log)}, "
                  f"go-around {ga} -> {base}.mp4/.png ({time.time() - t0:.0f} s)", flush=True)
    finally:
        env.close()


def mode_batch(n, seed0=9000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    env = E.LandingEnv(renderer="egl", seed=0, headless=True)
    teacher = Teacher()
    rows, paths = [], []
    t0 = time.time()
    try:
        for i in range(n):
            log, info, info0, _ = run_episode(env, teacher, seed0 + i)
            td = info.get("touchdown") or {}
            rows.append(dict(seed=seed0 + i, pad_speed=round(info0["pad_speed"], 3),
                             outcome=info["outcome"], time_s=round(len(log) * P.POLICY_DT, 1),
                             reason=td.get("reason"),
                             impact=None if td.get("impact") is None else round(td["impact"], 3),
                             offset_cm=None if not td.get("land_xy") else
                             round(100 * float(np.linalg.norm(td["land_xy"])), 1),
                             go_around=int(any(r["phase"] == "go-around" for r in log))))
            paths.append((np.array([r["rel_pad"][0] for r in log]), np.array([r["hf"] for r in log])))
            if (i + 1) % 50 == 0:
                print(f"[batch] {i + 1}/{n} done ({time.time() - t0:.0f} s)", flush=True)
    finally:
        env.close()
    with open(os.path.join(OUT, "batch.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    ok = [r for r in rows if r["outcome"] == "success"]
    k, nn = len(ok), len(rows)
    z = 1.96                                       # Wilson 95% interval
    c = (k + z * z / 2) / (nn + z * z)
    hw = z * np.sqrt(k * (nn - k) / nn + z * z / 4) / (nn + z * z)
    imp = np.array([r["impact"] for r in ok])
    off = np.array([r["offset_cm"] for r in ok])
    tim = np.array([r["time_s"] for r in ok])
    outcomes = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    lines = [f"Teacher on {nn} fixed test episodes (seeds {seed0}..{seed0 + nn - 1})",
             f"safe landings: {k}/{nn} = {100 * k / nn:.1f}%  (95% CI {100 * (c - hw):.1f}-{100 * (c + hw):.1f}%)",
             f"outcomes: {outcomes}",
             f"landed by drop: {sum(r['reason'] == 'drop' for r in ok)}, by auto-cut: "
             f"{sum(r['reason'] == 'auto_cut' for r in ok)}",
             f"impact [m/s]: mean {imp.mean():.2f}, max {imp.max():.2f} (limit 1.00)",
             f"distance from the pad centre [cm]: mean {off.mean():.1f}, max {off.max():.1f}",
             f"time to land [s]: mean {tim.mean():.1f}, max {tim.max():.1f} (limit 30)",
             f"episodes with a go-around: {sum(r['go_around'] for r in rows)}"]
    txt = "\n".join(lines)
    open(os.path.join(OUT, "batch_summary.txt"), "w").write(txt + "\n")
    print(txt)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between([-PAD_SIZE / 2, PAD_SIZE / 2], -0.08, 0, color="gray", alpha=0.6, label="pad")
    for x, h in paths:
        ax.plot(x, h, "b-", lw=0.6, alpha=0.25)
    hh = np.linspace(0, 5, 20)
    ax.plot(-K_BACK * hh, hh, "m:", lw=2, label="teacher approach line")
    ax.set_xlabel("along the pad's travel [m] (negative = behind)")
    ax.set_ylabel("feet height above the pad [m]")
    ax.set_title(f"All {nn} teacher paths, relative to the moving pad ({100 * k / nn:.1f}% safe)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "batch_paths.png"), dpi=110)
    plt.close(fig)
    print(f"[batch] wrote {OUT}/batch.csv, batch_summary.txt, batch_paths.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    v = sub.add_parser("videos")
    v.add_argument("--seeds", type=int, nargs="+", default=[9000, 9001, 9002, 9003])
    b = sub.add_parser("batch")
    b.add_argument("--n", type=int, default=500)
    t = sub.add_parser("troubles")
    t.add_argument("--kinds", nargs="+", default=["wobble", "gust", "pad_lost", "bad_approach"])
    a = ap.parse_args()
    if a.mode == "videos":
        mode_videos(a.seeds)
    elif a.mode == "troubles":
        mode_troubles(a.kinds)
    else:
        mode_batch(a.n)


if __name__ == "__main__":
    main()
