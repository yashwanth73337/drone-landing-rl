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

Build the dataset first (V6 builder, unchanged; ~8.5 GB image file for ~200k samples):
  python -m v6_drop.learn.dataset --src v7/results/demos --dst v7/results/bc_data
Run (repo root; charger in; performance mode):
  nohup python -m v7.scripts.train_bc_v7 --name bc_v7 > v7/results/bc_v7.log 2>&1 &
Watch:  tail -f v7/results/bc_v7.log | grep --line-buffered "^\\[bc"
Stop:   pkill -f "v7.scripts.train_bc_v7"
"""
import os
import sys

import numpy as np

from v6_drop.envs.landing_env import PRIV_SCALE
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


def use_v7_numbers():
    B6.ACT_SCALE = ACT_SCALE
    B6.feet_height = feet_height
    B6.RESULTS = RESULTS


def main(argv=None):
    use_v7_numbers()
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--name" not in argv:
        argv = ["--name", "bc_v7"] + argv
    sys.argv = [sys.argv[0]] + argv
    B6.main()


if __name__ == "__main__":
    main()
