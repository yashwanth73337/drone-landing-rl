"""v7 S4 diagnostic 2 (20 Nov 2026): is the student's error where the pad is OUT of view?

DAgger rounds 1 and 2 did not lift the student (1/400, 0/400) and its training move loss
stays ~0.027 (BC on the teacher's flights: 0.004): many labels cannot be predicted from what
the student has. This check splits the student's speed error by whether the pad centre is
inside the camera picture at that step.

"In view" is computed from the SAVED true state (no simulation): the pad centre seen from
the body ('priv' x 5), the drone's attitude (quaternion in 'state'), the 45 deg gimbal and
the camera's field of view -> projected pixel inside the 256 x 160 picture.
Out-of-view steps are also split by the time since the pad was last in view (0-1 s, 1-3 s,
> 3 s): memory can carry about 1 to 3 s; beyond that the picture holds no clue.

The student runs each flight from the start with its memory carried along (as in flight).
Read-only. Run (repo root):
  python -m v7.scripts.diag_view_v7 --ckpt v7/results/runs/dagger_r2/last.pt
Output: printed, and saved as diagview_<ckpt stem>.txt next to the checkpoint.
"""
import argparse
import os
import sys

import numpy as np
import pybullet as p
import torch

PARTS = (("teacher demos (BC val)", "bc_data", "val"),
         ("DAgger r1 (pure)", os.path.join("dagger_data", "r1"), "all"),
         ("DAgger r2 (beta 0.5)", os.path.join("dagger_data", "r2"), "all"))
GROUPS = ("in view", "out 0-1 s", "out 1-3 s", "out > 3 s")


def quat_to_R(q):
    return np.array(p.getMatrixFromQuaternion([float(x) for x in q])).reshape(3, 3)


def in_view(cam, priv, state):
    """Per step: is the pad centre inside the picture? (from saved true state)"""
    rel_b = priv[:, :3].astype(np.float64) * 5.0
    out = np.zeros(len(priv), bool)
    for k in range(len(priv)):
        R = quat_to_R(state[k, 3:7])
        uv, z = cam.project(R @ rel_b[k], R, np.zeros(3))
        u, v = uv[0]
        out[k] = bool(z[0] > 0 and 0 <= u <= cam.w and 0 <= v <= cam.h)
    return out


def groups(vis):
    """Group index per step: 0 in view; 1/2/3 out of view for 0-1 / 1-3 / > 3 s since seen."""
    g = np.zeros(len(vis), int)
    since = 10 ** 6
    for k, s in enumerate(vis):
        since = 0 if s else since + 1
        g[k] = 0 if s else (1 if since <= 10 else (2 if since <= 30 else 3))
    return g


def main(argv=None):
    from v6_drop.learn.dataset import Demos
    from v6_drop.scripts.eval_student import load
    from v7.config import PITCH_DEG
    from v7.envs.gimbal_camera import GimbalCamera
    from v7.envs.landing_env_v7 import VXY_MAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--results", default=os.path.join("v7", "results"))
    ap.add_argument("--max-eps", type=int, default=150, help="flights per part (speed)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    model, _ = load(a.ckpt, a.device)
    cid = p.connect(p.DIRECT)
    cam = GimbalCamera(cid, pitch_deg=PITCH_DEG, renderer="tiny")
    lines = [f"Student {a.ckpt}: speed error (m/s, |student - teacher label|) by pad visibility",
             f"  camera {PITCH_DEG:g} deg gimbal, field of view {cam.vfov:.0f} deg vertical; "
             f"'out x-y s' = pad out of the picture for x to y s",
             f"  {'data':24s} {'group':10s} {'share':>6s} {'err vx':>7s} {'err vy':>7s}"]
    rng = np.random.default_rng(0)
    for name, sub, split in PARTS:
        path = os.path.join(a.results, sub)
        if not os.path.exists(os.path.join(path, "arrays.npz")):
            lines.append(f"  {name:24s} (missing: {path})")
            continue
        d = Demos(path)
        eps = d.episodes("val") if split == "val" else np.arange(len(d.ep))
        if len(eps) > a.max_eps:
            eps = np.sort(rng.choice(eps, a.max_eps, replace=False))
        err, grp = [], []
        for e in eps:
            n, r0 = int(d.length[e]), int(d.start[e])
            img = torch.as_tensor(d.images[r0:r0 + n], device=a.device).unsqueeze(1)
            st = d.a["state"][r0:r0 + n]
            with torch.no_grad():
                pred = model(img, torch.as_tensor(st, device=a.device, dtype=torch.float32).unsqueeze(1))[0]
            pred = pred[:, 0].float().cpu().numpy()
            err.append(np.abs(pred[:, :2] - d.a["move"][r0:r0 + n, :2]) * VXY_MAX)
            grp.append(groups(in_view(cam, d.a["priv"][r0:r0 + n], st)))
        err, grp = np.concatenate(err), np.concatenate(grp)
        for gi, gname in enumerate(GROUPS):
            m = grp == gi
            if m.any():
                lines.append(f"  {name:24s} {gname:10s} {100 * m.mean():5.1f}% {err[m, 0].mean():7.2f} "
                             f"{err[m, 1].mean():7.2f}")
        lines.append(f"  {name:24s} {'ALL':10s} {100.0:5.1f}% {err[:, 0].mean():7.2f} {err[:, 1].mean():7.2f}")
    p.disconnect(cid)
    txt = "\n".join(lines)
    print(txt, flush=True)
    out = os.path.join(os.path.dirname(a.ckpt), f"diagview_{os.path.splitext(os.path.basename(a.ckpt))[0]}.txt")
    open(out, "w").write(txt + "\n")
    print(f"saved {out}", flush=True)
    return lines


if __name__ == "__main__":
    main(sys.argv[1:])
