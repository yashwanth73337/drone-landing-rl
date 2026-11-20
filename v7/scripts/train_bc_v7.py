"""v7 S3b: behaviour cloning for V7 (the student copies the V7 teacher), 20 Nov 2026.

The V6 BC trainer (v6_drop/scripts/train_bc.py) with the SAME loss, network and settings
(D3, 20 Nov "V6 settings"): clips of 32 steps (3.2 s) after up to 16 warm-up steps, move MSE
+ drop BCE with drop labels weighted 20 x, 20 epochs, Adam 3e-4, batch 16 clips, network 256,
validation = episodes with id % 10 == 0, checkpoints best.pt / last.pt, log.csv.

Only three V6 numbers are replaced (the V6 file is not edited; its names are re-pointed in
THIS process only):
  ACT_SCALE    +-6 m/s (V6)   -> the V7 env's +-10 m/s, +-1 m/s, +-60 deg/s (error reports in
               m/s; the move labels are already in [-1, 1])
  feet height  priv centre height - 0.152 m (V6 legs) -> - 0.22 m (S500), for the "low
               steps" report and --low-weight
  RESULTS      v6_drop/results -> v7/results (runs go to v7/results/runs/<name>)
No drop relabel is needed: the V7 teacher already labels with the safe30m rule.

S4c (20 Nov): --blind-mask SEC gives NO loss (move and drop) on steps where the pad centre has
been OUT of the camera picture for more than SEC seconds (computed per step from the saved
true state with the 45 deg gimbal geometry, v7/scripts/diag_view_v7.in_view). Reason: there
the teacher's label ("the pad is 15 m behind on the left") cannot be known from what the
student has; diag_view_v7 on DAgger r1/r2 showed error 2 to 3 m/s on these steps (29 to 47 %
of the student's own flights) vs ~0.7 in view. Default: off (exactly the V6 loss). The run
folder gets v7_config.json with the setting and the masked share.

Build the dataset first (V6 builder, unchanged; ~8.5 GB image file for ~200k samples):
  python -m v6_drop.learn.dataset --src v7/results/demos --dst v7/results/bc_data
Run (repo root; charger in; performance mode):
  nohup python -m v7.scripts.train_bc_v7 --name bc_v7 > v7/results/bc_v7.log 2>&1 &
  S4c: ... --blind-mask 3 (no loss after 3 s without the pad in view)
Watch:  tail -f v7/results/bc_v7.log | grep --line-buffered "^\\[bc"
Stop:   pkill -f "v7.scripts.train_bc_v7"
"""
import json
import os
import sys

import numpy as np

from v6_drop.envs.landing_env import PRIV_SCALE
from v6_drop.learn.dataset import BURN, CLIP, Demos
from v6_drop.scripts import train_bc as B6

from ..envs.landing_env_v7 import VXY_MAX, VZ_MAX, YAW_RATE_MAX
from ..envs.s500 import REST_HEIGHT

ACT_SCALE = np.array([VXY_MAX, VXY_MAX, VZ_MAX, YAW_RATE_MAX])
RESULTS = os.path.join("v7", "results")
assert PRIV_SCALE[7] == 5.0          # height scale in 'priv' is the same in V6 and V7


def feet_height(priv):
    """Skid height above the pad top [m] from 'priv' (last entry: drone centre height above
    the pad top / 5), for a level S500 (0.22 m standing height)."""
    return priv[..., 7] * float(PRIV_SCALE[7]) - REST_HEIGHT


def known_steps(d, blind_s):
    """1.0 where the pad was in the picture within the last blind_s seconds (per episode),
    else 0.0."""
    import pybullet as p
    from ..config import PITCH_DEG
    from ..envs.gimbal_camera import GimbalCamera
    from .diag_view_v7 import in_view
    cid = p.connect(p.DIRECT)
    try:
        cam = GimbalCamera(cid, pitch_deg=PITCH_DEG, renderer="tiny")
        vis = in_view(cam, d.a["priv"], d.a["state"])
    finally:
        p.disconnect(cid)
    known = np.zeros(len(vis), np.float32)
    lim = int(round(blind_s * 10))
    for r0, n in zip(d.start, d.length):
        since = 10 ** 6
        for k in range(int(r0), int(r0 + n)):
            since = 0 if vis[k] else since + 1
            known[k] = float(since <= lim)
    return known


class DemosV7(Demos):
    """V6 Demos; with blind_s, the loss mask is also 0 on long-blind steps."""

    def __init__(self, paths, drop_gate=None, blind_s=None):
        super().__init__(paths, drop_gate=drop_gate)
        self.blind_s = blind_s
        self.known = None if blind_s is None else known_steps(self, blind_s)

    def clip(self, e, s):
        out = super().clip(e, s)
        if self.known is not None:
            n, r0 = int(self.length[e]), int(self.start[e])
            lo, hi = max(s, 0), min(s + CLIP, n)
            b0 = max(0, lo - BURN)
            idx = np.arange(r0 + b0, r0 + hi)
            out["mask"][:len(idx)] *= self.known[idx]
        return out


def use_v7_numbers():
    B6.ACT_SCALE = ACT_SCALE
    B6.feet_height = feet_height
    B6.RESULTS = RESULTS


def main(argv=None):
    use_v7_numbers()
    argv = list(sys.argv[1:] if argv is None else argv)
    blind_s = None
    if "--blind-mask" in argv:
        i = argv.index("--blind-mask")
        blind_s = float(argv[i + 1])
        del argv[i:i + 2]
    if "--name" not in argv:
        argv = ["--name", "bc_v7"] + argv
    name = argv[argv.index("--name") + 1]
    info = {}

    def make_demos(paths, drop_gate=None):
        d = DemosV7(paths, drop_gate=drop_gate, blind_s=blind_s)
        if blind_s is not None:
            info["masked_share"] = round(float(1.0 - d.known.mean()), 4)
            print(f"[bc]   blind mask: no loss on steps with the pad out of view > {blind_s:g} s: "
                  f"{100 * info['masked_share']:.1f}% of all steps", flush=True)
        return d
    B6.Demos = make_demos
    sys.argv = [sys.argv[0]] + argv
    B6.main()
    out = os.path.join(B6.RESULTS, "runs", name)
    json.dump(dict(blind_mask_s=blind_s, **info), open(os.path.join(out, "v7_config.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
