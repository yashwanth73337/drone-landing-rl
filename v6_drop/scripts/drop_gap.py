"""v6_drop Step 6d diagnostic: WHY does the student never reach the teacher's drop gate?

The teacher drops only when ALL of these hold for HOLD_STEPS (3) decisions in a row:
  d       horizontal distance to the pad centre      < D_DROP     (6 cm)
  height  |feet height - H_DROP (3 cm)|              < DH_DROP    (1 cm), i.e. 2-4 cm
  v_xy    horizontal speed relative to the pad       < V_REL_DROP (0.10 m/s)
  v_z     vertical speed relative to the pad         < V_REL_DROP (0.10 m/s)
  tilt                                               < TILT_DROP  (4 deg)
This script rebuilds those numbers OFFLINE for every saved step (from 'priv', the critic's
true relative state, and the quaternion in 'state'), and reports for the student's flights:
  1. which condition fails most often in the last 2 s before the flight ended
  2. how the student crosses the 2-4 cm height window (how many steps it stays there,
     and how fast it is sinking when it does)
  3. what-if: how many flights WOULD have met a wider gate (nothing is changed; this only
     measures)
Run (from the repo root; reads the raw .npz files, a few seconds):
  python -m v6_drop.scripts.drop_gap --src v6_drop/results/dagger/r2
  python -m v6_drop.scripts.drop_gap --src v6_drop/results/demos        (the teacher, to compare)
"""
import argparse
import csv
import glob
import os

import numpy as np

from v6_drop.envs import lmf2_params as P
from v6_drop.envs.landing_env import PRIV_SCALE
from v6_drop.experts import teacher as T

LAST = 20                       # decisions = the last 2 s of a flight


def quat_R(q):
    """(N, 4) quaternions (x, y, z, w) -> (N, 3, 3) rotation matrices (body -> world)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def gate_numbers(priv, state):
    """Per step: d, feet height, |v_xy| and |v_z| relative to the pad, tilt (rad)."""
    pr = priv.astype(np.float64) * PRIV_SCALE
    R = quat_R(state[:, 3:7].astype(np.float64))
    rel = np.einsum("nij,nj->ni", R, pr[:, 0:3])          # pad centre - drone, world frame
    vrel = -np.einsum("nij,nj->ni", R, pr[:, 3:6])         # drone - pad velocity, world frame
    return dict(d=np.hypot(rel[:, 0], rel[:, 1]), hf=pr[:, 7] - P.REST_HEIGHT,
                vxy=np.hypot(vrel[:, 0], vrel[:, 1]), vz=vrel[:, 2],
                tilt=np.arccos(np.clip(R[:, 2, 2], -1, 1)))


def gate(g, d_max=T.D_DROP, dh=T.DH_DROP, v_max=T.V_REL_DROP, tilt=T.TILT_DROP, hold=T.HOLD_STEPS):
    """The teacher's gate on arrays: per-step ok, and drop (ok held 'hold' steps in a row)."""
    ok = ((g["d"] < d_max) & (np.abs(g["hf"] - T.H_DROP) < dh) & (g["vxy"] < v_max)
          & (np.abs(g["vz"]) < v_max) & (g["tilt"] < tilt))
    run = np.zeros(len(ok), int)
    for i in range(len(ok)):
        run[i] = (run[i - 1] + 1 if i else 1) if ok[i] else 0
    return ok, run >= hold


def load(src):
    files = sorted(glob.glob(os.path.join(src, "dagger_*.npz")) + glob.glob(os.path.join(src, "demos_*.npz")))
    assert files, f"no .npz in {src}"
    eps = {}
    for f in files:
        z = np.load(f)
        ep = z["ep"]
        for e in np.unique(ep):
            m = ep == e
            eps[int(e)] = dict(priv=z["priv"][m], state=z["state"][m], drop=z["drop"][m], t=z["t"][m])
    rows = {}
    for name in ("flights.csv", "episodes.csv"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                rows[int(r["ep"])] = r
    return eps, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    eps, rows = load(a.src)
    groups = {}
    for e in eps:
        r = rows.get(e, {})
        key = r.get("outcome", "?") + ("/" + r["reason"] if r.get("reason") else "")
        groups.setdefault(key, []).append(e)
    print(f"[gap] {a.src}: {len(eps)} flights; " + ", ".join(f"{k} {len(v)}" for k, v in sorted(groups.items())))

    # 1. which condition fails in the last 2 s (low steps only: feet below 10 cm)
    names = ("d < 6 cm", "height 2-4 cm", "v_xy < 0.10", "|v_z| < 0.10", "tilt < 4 deg")
    for key, es in sorted(groups.items()):
        cond, n_low, closest = np.zeros(5), 0, []
        for e in es:
            g = gate_numbers(eps[e]["priv"], eps[e]["state"])
            s = slice(-LAST, None)
            low = g["hf"][s] < 0.10
            n_low += low.sum()
            c = np.stack([g["d"][s] < T.D_DROP, np.abs(g["hf"][s] - T.H_DROP) < T.DH_DROP,
                          g["vxy"][s] < T.V_REL_DROP, np.abs(g["vz"][s]) < T.V_REL_DROP,
                          g["tilt"][s] < T.TILT_DROP])
            cond += (c[:, low]).sum(1)
            if low.any():
                closest.append(g["d"][s][low].min())
        if n_low:
            print(f"\n[gap] {key}: last 2 s, steps with feet < 10 cm: {n_low}; share passing each condition:")
            print("      " + " | ".join(f"{nm} {100 * c / n_low:4.0f}%" for nm, c in zip(names, cond)))
            print(f"      closest approach to the centre while low: median {100 * np.median(closest):.0f} cm, "
                  f"best {100 * np.min(closest):.0f} cm")

    # 2. the height window
    stay, sink, n_cross = [], [], 0
    for e in eps:
        g = gate_numbers(eps[e]["priv"], eps[e]["state"])
        inwin = np.abs(g["hf"] - T.H_DROP) < T.DH_DROP
        if inwin.any():
            n_cross += 1
            runs = np.diff(np.flatnonzero(np.diff(np.r_[0, inwin.astype(int), 0])))[::2]
            stay.append(runs.max())
            sink.append(-g["vz"][inwin].mean())
    print(f"\n[gap] height window 2-4 cm: reached in {n_cross}/{len(eps)} flights")
    if stay:
        stay, sink = np.array(stay), np.array(sink)
        print(f"      longest stay in the window [steps of 0.1 s]: median {np.median(stay):.0f}, "
              f"share >= 3 steps {100 * np.mean(stay >= 3):.0f}%")
        print(f"      sinking speed while in the window [m/s]: median {np.median(sink):.2f}")

    # 3. what-if gates (measurement only)
    print("\n[gap] what-if: flights whose steps meet a gate for 3 steps in a row (labels in the data: "
          f"{int(sum(eps[e]['drop'].sum() for e in eps))} drop labels in "
          f"{sum(eps[e]['drop'].any() for e in eps)} flights)")
    for d_max, v_max, dh in ((0.06, 0.10, 0.010), (0.10, 0.10, 0.010), (0.15, 0.10, 0.010),
                             (0.20, 0.10, 0.010), (0.20, 0.15, 0.010), (0.20, 0.15, 0.015),
                             (0.30, 0.20, 0.015)):
        n = 0
        for e in eps:
            g = gate_numbers(eps[e]["priv"], eps[e]["state"])
            n += gate(g, d_max, dh, v_max)[1].any()
        print(f"      d < {100 * d_max:2.0f} cm, v < {v_max:.2f} m/s, height {100 * (T.H_DROP - dh):.1f}-"
              f"{100 * (T.H_DROP + dh):.1f} cm: {n}/{len(eps)} flights")


if __name__ == "__main__":
    main()
