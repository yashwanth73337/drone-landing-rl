"""v7 S3 diagnostic (20 Nov 2026): why does the BC student get lost on every dev flight?

Read-only: no training, no change to any file except the report it writes.

Part A (memory over a whole flight): the student reads each VALIDATION demo flight from the
first to the last decision with its memory carried along, as in a real flight ("full").
The same flights again with the memory restarted every 48 decisions (4.8 s, the length it
was trained on: 16 warm-up + 32) ("restarted"). Speed error vs the teacher's labels in m/s,
by time in the flight: 0-1.6 s, 1.6-5 s, 5-10 s, 10-20 s, > 20 s. If "full" grows with time
and "restarted" does not, the memory drifts on long flights.
Part B (closed loop): the student flies 3 dev flights alone on a slow straight pad (start
speed 2 m/s), with the student rules. Every 0.5 s: where the drone is in the pad's frame
(x ahead, y left, m), the skid height, the student's command and what the TEACHER would
command in that same moment (m/s; heading frame), marker seen, drop chance.
21 Nov: also the HEADING error (drone nose vs pad heading, deg) and both yaw-rate commands
(deg/s): the gimbal camera turns with the nose, and the speed commands are in the nose's
frame, so a drifting heading would hide the pad AND point every command the wrong way.
--pad-speed and --mode choose the pad for Part B; --skip-a skips Part A.

Run (repo root):
  python -m v7.scripts.diag_student_v7 --ckpt v7/results/runs/bc_v7/last.pt
Output: printed, and saved as diag_<ckpt stem>.txt next to the checkpoint.
"""
import argparse
import os
import sys

import numpy as np
import torch

BANDS = ((0, 16), (16, 50), (50, 100), (100, 200), (200, 10 ** 6))
BAND_NAMES = ("0-1.6 s", "1.6-5 s", "5-10 s", "10-20 s", "> 20 s")
WINDOW = 48


def part_a(model, data, device, scale, max_eps=None):
    from v6_drop.learn.dataset import Demos
    d = Demos(data)
    eps = d.episodes("val")[:max_eps]
    err = {"full": [[] for _ in BANDS], "restarted": [[] for _ in BANDS]}
    for e in eps:
        n, r0 = int(d.length[e]), int(d.start[e])
        img = torch.as_tensor(d.images[r0:r0 + n], device=device).unsqueeze(1)
        st = torch.as_tensor(d.a["state"][r0:r0 + n], device=device, dtype=torch.float32).unsqueeze(1)
        lab = d.a["move"][r0:r0 + n]
        with torch.no_grad():
            full = model(img, st)[0][:, 0].float().cpu().numpy()
            rest = np.concatenate([model(img[i:i + WINDOW], st[i:i + WINDOW])[0][:, 0].float().cpu().numpy()
                                   for i in range(0, n, WINDOW)])
        for name, pred in (("full", full), ("restarted", rest)):
            e_xy = np.abs(pred[:, :2] - lab[:, :2]) * scale[:2]          # m/s
            for b, (lo, hi) in enumerate(BANDS):
                if lo < n:
                    err[name][b].append(e_xy[lo:min(hi, n)])
    lines = [f"Part A: {len(eps)} validation demo flights (decisions per flight: median "
             f"{int(np.median(d.length[eps]))}, max {int(d.length[eps].max())})",
             f"  speed error vx / vy (m/s) by time in the flight; memory 'full' vs 'restarted' every 4.8 s",
             f"  {'time':10s} {'steps':>7s} {'full vx':>8s} {'vy':>6s} {'restarted vx':>13s} {'vy':>6s}"]
    for b, name in enumerate(BAND_NAMES):
        if not err["full"][b]:
            continue
        f = np.concatenate(err["full"][b])
        r = np.concatenate(err["restarted"][b])
        lines.append(f"  {name:10s} {len(f):7d} {f[:, 0].mean():8.2f} {f[:, 1].mean():6.2f} "
                     f"{r[:, 0].mean():13.2f} {r[:, 1].mean():6.2f}")
    return lines


def part_b(model, device, seeds, every=5, pad_speed=2.0, mode="straight"):
    from v7.config import make_env
    from v7.envs.platform_v7 import PAD_TOP_Z
    from v7.envs.s500 import lowest_z
    from v7.experts.teacher_v7 import TeacherV7
    env = make_env(student_rules=True, renderer="egl", seed=7)
    lines = ["", f"Part B: the student flies alone; {mode} pad (start {pad_speed:g} m/s); every 0.5 s.",
             "  pad frame: x = ahead of the pad centre (+) / behind (-), y = left (+); height = skids",
             "  commands in m/s (vx forward, vy left, vz up), heading frame; hdg = nose minus pad"
             " heading (deg); yaw = yaw-rate command student/teacher (deg/s)"]
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed, options=dict(mode=mode, pad_speed=pad_speed))
            teacher = TeacherV7()
            teacher.reset()
            hc, k, first_bad = None, 0, None
            lines.append(f"\n  seed {seed}")
            lines.append(f"  {'t s':>5s} {'x':>6s} {'y':>6s} {'height':>6s} | {'student vx':>10s} {'vy':>5s} "
                         f"{'vz':>5s} | {'teacher vx':>10s} {'vy':>5s} {'vz':>5s} | {'hdg':>5s} {'yaw s/t':>9s} | seen  drop%")
            while True:
                s = env.world.true_state()
                act, hc, p = model.act(obs, hc, device)
                t_act, _ = teacher.act(env)
                cs = np.asarray(act["move"]) * env.act_scale
                ct = np.asarray(t_act["move"]) * env.act_scale
                if first_bad is None and np.hypot(*(cs - ct)[:2]) > 1.0:
                    first_bad = k
                if k % every == 0:
                    xy = env.world.plat.world_to_pad(s["pos"][None, :])[0]
                    h = lowest_z(s["R"], s["pos"]) - PAD_TOP_Z
                    seen = (env.last_meas is not None and bool(env.last_meas.ids))
                    yaw = np.arctan2(s["R"][1, 0], s["R"][0, 0])
                    hdg = np.degrees((yaw - s["pad_psi"] + np.pi) % (2 * np.pi) - np.pi)
                    lines.append(f"  {k / 10:5.1f} {xy[0]:6.1f} {xy[1]:6.1f} {h:6.2f} | {cs[0]:10.2f} {cs[1]:5.2f} "
                                 f"{cs[2]:5.2f} | {ct[0]:10.2f} {ct[1]:5.2f} {ct[2]:5.2f} | {hdg:5.0f} "
                                 f"{np.degrees(cs[3]):4.0f}/{np.degrees(ct[3]):<4.0f} | {'yes' if seen else ' no'}  "
                                 f"{100 * p:4.0f}")
                obs, _, term, trunc, info = env.step(act)
                k += 1
                if term or trunc:
                    break
            lines.append(f"  -> {info['outcome']} after {k / 10:.1f} s; first moment the student's horizontal "
                         f"command differs from the teacher's by > 1 m/s: "
                         f"{'never' if first_bad is None else f'{first_bad / 10:.1f} s'}")
    finally:
        env.close()
    return lines


def main(argv=None):
    from v6_drop.scripts.eval_student import load
    from v7.envs.landing_env_v7 import VXY_MAX, VZ_MAX, YAW_RATE_MAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join("v7", "results", "bc_data"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[30000, 30001, 30002])
    ap.add_argument("--max-eps", type=int, default=None, help="tests: limit Part A flights")
    ap.add_argument("--pad-speed", type=float, default=2.0)
    ap.add_argument("--mode", default="straight")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--tag", default="", help="added to the output file name")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    model, _ = load(a.ckpt, a.device)
    scale = np.array([VXY_MAX, VXY_MAX, VZ_MAX, YAW_RATE_MAX])
    lines = [f"Diagnostic of {a.ckpt}"]
    if not a.skip_a:
        lines += part_a(model, a.data, a.device, scale, a.max_eps)
    print("\n".join(lines), flush=True)
    lb = part_b(model, a.device, a.seeds, pad_speed=a.pad_speed, mode=a.mode)
    print("\n".join(lb), flush=True)
    out = os.path.join(os.path.dirname(a.ckpt),
                       f"diag_{os.path.splitext(os.path.basename(a.ckpt))[0]}{a.tag}.txt")
    open(out, "w").write("\n".join(lines + lb) + "\n")
    print(f"\nsaved {out}", flush=True)
    return lines + lb


if __name__ == "__main__":
    main(sys.argv[1:])
