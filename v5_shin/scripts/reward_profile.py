"""Block 9 diagnostic: reward landscape of scripted descents (cf. V4's diag_v4_reward_profile).

Drone starts 6 m above a stationary pad (c = 0, DR off, true state), holds position
laterally, and descends at a fixed rate (or hovers). Reports per-step reward by height
band, and undiscounted and discounted (gamma = 0.99) returns, for both D3 variants.

Usage: python -m v5_shin.scripts.reward_profile [--lateral 0.0] [--height 6.0]
"""
import argparse
import json
import os

import numpy as np

from v5_shin.envs.dr import DRConfig
from v5_shin.envs.platform import PAD_TOP_Z
from v5_shin.envs.reward import ShinReward
from v5_shin.envs.shin_env import ACTION_SCALE, ShinLandingEnv

RATES = [0.0, 0.3, 0.5, 1.0, 1.5, 2.5]          # m/s; 0.0 = hover
BANDS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 7.0)]   # CoM height above pad top (m)
GAMMA = 0.99


def fly(rate, vz_penalty, height=6.0, lateral=0.0):
    env = ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=0,
                         dr=DRConfig.off(), reward_fn=ShinReward(vz_penalty), gains_mode="nominal")
    try:
        sp = dict(pos=np.array([-lateral, 0.0, PAD_TOP_Z + height]), yaw=0.0, psi_plat=0.0)
        env.reset(seed=0, options={"c": 0.0, "spawn": sp})
        rows, ret, disc, k = [], 0.0, 0.0, 0
        while True:
            rel = env.sim.plat.pad_center() - env.sim.quad.state()["pos"]
            cmd = np.array([1.0 * rel[0], 1.0 * rel[1], -rate, 0.0])
            _, r, te, tr, info = env.step(cmd / ACTION_SCALE)
            h = -rel[2]
            rows.append((h, r))
            ret += r
            disc += GAMMA ** k * r
            k += 1
            if te or tr:
                break
        return dict(outcome=info["outcome"], steps=k, ret=ret, disc=disc, rows=rows)
    finally:
        env.close()


def hover_sweep(c, episodes=200, seed_base=9000, vz_penalty="literal"):
    """Discounted return of a pure hover (zero command) from Table I spawns, full DR.
    At c > 0 the platform drives away, so lateral progress turns negative."""
    env = ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=seed_base,
                         reward_fn=ShinReward(vz_penalty))
    out = []
    try:
        for i in range(episodes):
            env.reset(seed=seed_base + i, options={"c": c})
            disc, k = 0.0, 0
            while True:
                _, r, te, tr, info = env.step(np.zeros(4))
                disc += GAMMA ** k * r
                k += 1
                if te or tr:
                    break
            out.append((disc, info["outcome"]))
    finally:
        env.close()
    d = np.array([o[0] for o in out])
    drift = sum(o[1] == "drift" for o in out) / len(out)
    return dict(c=c, mean=float(d.mean()), median=float(np.median(d)), p10=float(np.percentile(d, 10)),
                p90=float(np.percentile(d, 90)), drift_frac=drift)


def policy_sweep(policy, c, episodes=200, seed_base=9000, vz_penalty="literal", v_down_max=1.5):
    """Discounted return of the privileged oracle from Table I spawns (full DR).
    policy='follow': oracle laterally, but never descends (vz = 0): the tracking stall.
    policy='land':   oracle with its descent-rate cap set to v_down_max."""
    from v5_shin.policies.oracle import OracleController
    orc = OracleController(v_down_max=v_down_max, v_down_min=min(0.4, v_down_max))
    env = ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=seed_base,
                         reward_fn=ShinReward(vz_penalty))
    out = []
    try:
        for i in range(episodes):
            env.reset(seed=seed_base + i, options={"c": c})
            disc, k = 0.0, 0
            while True:
                rel_p, rel_v, st = env.sim.true_relative_state()
                yaw = np.arctan2(st["R"][1, 0], st["R"][0, 0])
                cmd = orc(rel_p, rel_v, env.sim.plat.velocity(), yaw)
                if policy == "follow":
                    cmd[2] = 0.0
                _, r, te, tr, info = env.step(cmd / ACTION_SCALE)
                disc += GAMMA ** k * r
                k += 1
                if te or tr:
                    break
            out.append((disc, info["outcome"]))
    finally:
        env.close()
    d = np.array([o[0] for o in out])
    succ = sum(o[1] == "success" for o in out) / len(out)
    return dict(policy=policy, c=c, v_down_max=v_down_max, mean=float(d.mean()),
                median=float(np.median(d)), success=succ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, default=6.0)
    ap.add_argument("--lateral", type=float, default=0.0)
    ap.add_argument("--hover-sweep", action="store_true",
                    help="hover return vs curriculum c from Table I spawns (full DR)")
    ap.add_argument("--policy-sweep", action="store_true",
                    help="follow-without-descending vs landing (oracle) returns, both D3 variants")
    a = ap.parse_args()
    if a.policy_sweep:
        res = []
        print(f"{'vz_pen':>8} {'policy':>7} {'v_down':>6} {'c':>6} {'mean':>7} {'median':>7} {'succ':>5}")
        for vzp in ("prose", "literal"):
            for c in (0.125, 1.0):
                for pol, vd in (("follow", 1.5), ("land", 0.5), ("land", 1.5)):
                    r = policy_sweep(pol, c, episodes=100, vz_penalty=vzp, v_down_max=vd)
                    r["vz_penalty"] = vzp
                    res.append(r)
                    print(f"{vzp:>8} {pol:>7} {vd:>6.1f} {c:>6.3f} {r['mean']:>7.2f} "
                          f"{r['median']:>7.2f} {r['success']:>5.2f}", flush=True)
        run = os.path.join(os.path.dirname(__file__), "..", "runs", "reward_profile_policy_sweep")
        os.makedirs(run, exist_ok=True)
        with open(os.path.join(run, "policy_sweep.json"), "w") as f:
            json.dump(res, f, indent=2)
        return
    if a.hover_sweep:
        res = [hover_sweep(c) for c in (0.0, 0.125, 0.25, 0.5, 1.0)]
        print(f"{'c':>6} {'mean':>7} {'median':>7} {'p10':>7} {'p90':>7} {'drift':>6}")
        for r in res:
            print(f"{r['c']:>6.3f} {r['mean']:>7.2f} {r['median']:>7.2f} {r['p10']:>7.2f} "
                  f"{r['p90']:>7.2f} {r['drift_frac']:>6.2f}")
        run = os.path.join(os.path.dirname(__file__), "..", "runs", "reward_profile_hover_sweep")
        os.makedirs(run, exist_ok=True)
        with open(os.path.join(run, "hover_sweep.json"), "w") as f:
            json.dump(res, f, indent=2)
        return
    out = {}
    for variant in ("prose", "literal"):
        print(f"\nvz_penalty = {variant}" + ("   (D3 default)" if variant == "literal" else "   (ablation)"))
        print(f"{'rate':>5} {'outcome':>8} {'steps':>5} {'return':>8} {'disc.ret':>8}  "
              + "  ".join(f"r/step h{lo:.0f}-{hi:.0f}" for lo, hi in BANDS))
        for rate in RATES:
            res = fly(rate, variant, a.height, a.lateral)
            rows = np.array(res["rows"])
            bands = []
            for lo, hi in BANDS:
                m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
                # exclude the terminal +-10 from the per-step shaping means
                sel = rows[m & (np.abs(rows[:, 1]) < 9.99), 1]
                bands.append(float(sel.mean()) if len(sel) else float("nan"))
            print(f"{rate:>5.1f} {res['outcome']:>8} {res['steps']:>5} {res['ret']:>8.2f} "
                  f"{res['disc']:>8.2f}  " + "  ".join(f"{b:>12.4f}" for b in bands))
            out[f"{variant}_{rate}"] = dict(outcome=res["outcome"], steps=res["steps"],
                                           ret=res["ret"], disc=res["disc"], bands=bands)
    run = os.path.join(os.path.dirname(__file__), "..", "runs",
                       f"reward_profile_h{a.height}_lat{a.lateral}")
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, "profile.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
