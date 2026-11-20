"""v7 S4 diagnostic 3 (20 Nov 2026): does the student actually USE the camera?

Suspect ("copycat", a known behaviour-cloning trap): in the teacher's smooth flights, "next
command = current command + a bit" predicts the label well without looking at the picture,
so the network may lean on its own numbers instead of the image.

The student reads each flight from the start with its memory carried along (as in flight),
four ways, and we compare its speed error vs the teacher labels (m/s):
  normal          nothing removed
  other pictures  the camera images come from a DIFFERENT flight (same step numbers,
                  cycling): real-looking pictures that say nothing about THIS pad
  no command      the 'command in effect' inputs (state[7:11]) set to 0
  no own speed    the drone's own speed inputs (state[0:3]) set to 0
If 'other pictures' is barely worse than 'normal', the student hardly uses the camera.
Steps are split into 'pad in view' (where the picture should matter) and all steps.
Read-only. Run (repo root):
  python -m v7.scripts.diag_inputs_v7 --ckpt v7/results/runs/dagger_r2m/last.pt
Output: printed, and saved as diaginputs_<ckpt stem>.txt next to the checkpoint.
"""
import argparse
import os
import sys

import numpy as np
import pybullet as p
import torch

PARTS = (("teacher demos (BC val)", "bc_data", "val"),
         ("DAgger r2 (beta 0.5)", os.path.join("dagger_data", "r2"), "all"))
VARIANTS = ("normal", "other pictures", "no command", "no own speed")


def main(argv=None):
    from v6_drop.learn.dataset import Demos
    from v6_drop.scripts.eval_student import load
    from v7.config import PITCH_DEG
    from v7.envs.gimbal_camera import GimbalCamera
    from v7.envs.landing_env_v7 import VXY_MAX
    from v7.scripts.diag_view_v7 import in_view
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--results", default=os.path.join("v7", "results"))
    ap.add_argument("--max-eps", type=int, default=100, help="flights per part (speed)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    model, _ = load(a.ckpt, a.device)
    cid = p.connect(p.DIRECT)
    cam = GimbalCamera(cid, pitch_deg=PITCH_DEG, renderer="tiny")
    lines = [f"Student {a.ckpt}: speed error vx / vy (m/s) with some inputs removed",
             f"  {'data':24s} {'variant':15s} {'pad in view':>16s} {'all steps':>16s}"]
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
        err = {v: [] for v in VARIANTS}
        vis_all = []
        for j, e in enumerate(eps):
            n, r0 = int(d.length[e]), int(d.start[e])
            o = eps[(j + 1) % len(eps)]                       # another flight for its pictures
            on, o0 = int(d.length[o]), int(d.start[o])
            img = d.images[r0:r0 + n]
            other = d.images[o0 + (np.arange(n) % on)]
            st = d.a["state"][r0:r0 + n].astype(np.float32)
            lab = d.a["move"][r0:r0 + n, :2]
            vis_all.append(in_view(cam, d.a["priv"][r0:r0 + n], st))
            for v in VARIANTS:
                im, s2 = img, st.copy()
                if v == "other pictures":
                    im = other
                elif v == "no command":
                    s2[:, 7:11] = 0.0
                elif v == "no own speed":
                    s2[:, 0:3] = 0.0
                with torch.no_grad():
                    pred = model(torch.as_tensor(im, device=a.device).unsqueeze(1),
                                 torch.as_tensor(s2, device=a.device).unsqueeze(1))[0]
                err[v].append(np.abs(pred[:, 0, :2].float().cpu().numpy() - lab) * VXY_MAX)
        vis = np.concatenate(vis_all)
        for v in VARIANTS:
            e = np.concatenate(err[v])
            iv = e[vis]
            lines.append(f"  {name:24s} {v:15s} {iv[:, 0].mean():7.2f} / {iv[:, 1].mean():5.2f}  "
                         f"{e[:, 0].mean():7.2f} / {e[:, 1].mean():5.2f}")
        lines.append(f"  {name:24s} ({len(eps)} flights, pad in view {100 * vis.mean():.0f} % of steps)")
    p.disconnect(cid)
    txt = "\n".join(lines)
    print(txt, flush=True)
    out = os.path.join(os.path.dirname(a.ckpt),
                       f"diaginputs_{os.path.splitext(os.path.basename(a.ckpt))[0]}.txt")
    open(out, "w").write(txt + "\n")
    print(f"saved {out}", flush=True)
    return lines


if __name__ == "__main__":
    main(sys.argv[1:])
