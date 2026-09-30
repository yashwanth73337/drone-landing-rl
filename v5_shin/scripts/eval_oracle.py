"""Block 5: evaluate the privileged oracle (and random policies) on the Table I task.

Per-episode seeding: episode i uses seed seed_base + i (gains and motor
constants re-sampled per episode). Writes a per-episode CSV and a summary JSON
to v5_shin/runs/<name>/.

Usage (from the repo root):
  python -m v5_shin.scripts.eval_oracle --policy oracle --c 1.0 --episodes 200 --seed-base 9000
  python -m v5_shin.scripts.eval_oracle --policy random --hold 1 --c 0.0 --episodes 200
"""
import argparse
import csv
import json
import os
import time
from collections import Counter

import numpy as np

from v5_shin.envs.dr import DRConfig
from v5_shin.envs.landing_sim import LandingSim
from v5_shin.policies.oracle import OracleController, random_policy

OUTCOMES = ["success", "crash_ground", "crash_platform", "tilt", "drift", "timeout"]


def run_episodes(policy="oracle", c=1.0, episodes=200, seed_base=9000, hold=1, sim=None,
                 dr=True):
    own = sim is None
    sim = sim or LandingSim(seed=seed_base, dr=DRConfig() if dr else DRConfig.off())
    oracle = OracleController()
    rows = []
    try:
        for i in range(episodes):
            sim.reseed(seed_base + i)
            info0 = sim.reset(c=c)
            act = oracle if policy == "oracle" else random_policy(
                np.random.default_rng(10 ** 6 + seed_base + i), hold)
            z_min = np.inf
            start_xy = sim.quad.state()["pos"][:2].copy()
            while True:
                rel_p, rel_v, s = sim.true_relative_state()
                yaw = np.arctan2(s["R"][1, 0], s["R"][0, 0])
                z_min = min(z_min, -rel_p[2])
                te, tr, info = sim.step(act(rel_p, rel_v, sim.plat.velocity(), yaw))
                if te or tr:
                    break
            s = sim.quad.state()
            sp = info0["spawn"]
            rows.append(dict(
                episode=i, seed=seed_base + i, outcome=info["outcome"], steps=info["t"],
                com_over_pad=info["com_over_pad"],
                rel_pad_x=info["rel_pos_pad"][0], rel_pad_y=info["rel_pos_pad"][1],
                rel_vx=info["rel_vel"][0], rel_vy=info["rel_vel"][1], rel_vz=info["rel_vel"][2],
                tilt_deg=info["tilt_deg"], min_height_above_pad=z_min,
                net_xy_travel=float(np.linalg.norm(s["pos"][:2] - start_xy)),
                dx0=sp["dxyz0"][0], dy0=sp["dxyz0"][1], dz0=sp["dxyz0"][2],
                psi0_deg=np.degrees(sp["psi0"]),
            ))
    finally:
        if own:
            sim.close()
    return rows


def summarize(rows):
    n = len(rows)
    cnt = Counter(r["outcome"] for r in rows)
    succ = [r for r in rows if r["outcome"] == "success"]
    strict = sum(r["com_over_pad"] for r in succ)
    out = {k: cnt.get(k, 0) for k in OUTCOMES}
    out.update(
        episodes=n, success_rate=cnt.get("success", 0) / n, strict_success_rate=strict / n,
        mean_steps_success=float(np.mean([r["steps"] for r in succ])) if succ else None,
        touchdown_err_p95_m=float(np.percentile(
            [np.hypot(r["rel_pad_x"], r["rel_pad_y"]) for r in succ], 95)) if succ else None,
        touchdown_vz_mean=float(np.mean([r["rel_vz"] for r in succ])) if succ else None,
        min_height_median=float(np.median([r["min_height_above_pad"] for r in rows])),
        net_xy_travel_median=float(np.median([r["net_xy_travel"] for r in rows])),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["oracle", "random"], default="oracle")
    ap.add_argument("--hold", type=int, default=1, help="random: steps each action is held")
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--dr", choices=["on", "off"], default="on",
                    help="Table II domain randomisation (default on = paper)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or (f"eval_{a.policy}{'' if a.policy == 'oracle' else f'_h{a.hold}'}"
                      f"_c{a.c}_dr{a.dr}_s{a.seed_base}")
    out = os.path.join(os.path.dirname(__file__), "..", "runs", name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rows = run_episodes(a.policy, a.c, a.episodes, a.seed_base, a.hold, dr=(a.dr == "on"))
    summ = summarize(rows)
    summ.update(vars(a), wall_s=round(time.time() - t0, 1))
    with open(os.path.join(out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
