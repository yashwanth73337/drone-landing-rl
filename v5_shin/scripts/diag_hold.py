"""Hold diagnostic (evaluation only, no training; SPEC D22 follow-up).

Starts episodes ON the pad (the D22 practice start: in contact, moving with the pad, 0.3 m/s soft
descent) and flies the 1 s hold with
  policy : a trained privileged (P) checkpoint, deterministic mean actions (needs --run)
  oracle : a script that commands the pad-surface velocity under the drone (heading frame),
           v_z = --oracle-vz (default -0.3 m/s; several values allowed), yaw rate 0.
           A simple best-case controller. A more negative v_z presses the drone onto the pad
           (less thrust, more weight on the legs, more friction).
For each difficulty c it reports the outcome counts, the median |v_rel| (drone minus pad surface,
m/s) at each completed hold step, and how often the max |v_rel| over the last 5 hold steps is
<= 0.1 / 0.2 / 0.3 m/s (the D20 success rule uses 0.1).

Usage (repo root):
  python -m v5_shin.scripts.diag_hold --run P_td_v4p_s1 --episodes 200 --c 0 0.5 1
  python -m v5_shin.scripts.diag_hold --run P_td_v4p_s1 --episodes 200 --c 0 0.5 1 --dr-off
  python -m v5_shin.scripts.diag_hold --episodes 200 --c 1            (oracle only)
  python -m v5_shin.scripts.diag_hold --episodes 200 --c 0.5 1 --oracle-vz -0.3 -1.0
  D23 options: --pad-smooth (smooth pad speed changes), --hold-vz-down 5 (sit-down range in the
  hold; the oracle can then command down to -5 m/s, e.g. --oracle-vz -1 -5).
  --friction MU: effective drone-pad friction for this diagnostic only (PyBullet multiplies the
  two bodies' values; the sim uses 1.0 x 0.5 = 0.5 since D24). Sets pad 1.0 and drone MU.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

from v5_shin.envs.dr import DRConfig
from v5_shin.envs.landing_sim import HOLD_STEPS, HOLD_STILL_STEPS
from v5_shin.envs.reward import TouchdownReward
from v5_shin.envs.shin_env import ShinLandingEnv

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
KEYS = ("u", "critic", "target", "s_rel")
OUTCOMES = ("success", "bounce", "slide_off", "tip_over", "not_still", "crash_platform",
            "crash_ground", "tilt", "drift")


def oracle_action(env, vz=-0.3):
    """Command the pad-surface velocity the pad will have at the END of this policy step (where
    |v_rel| is measured). Without --pad-smooth that is the current velocity; with it the oracle
    also knows the step's ramp target (privileged, best case)."""
    sim = env.sim
    s = sim.quad.state()
    r = s["pos"] - sim.plat.pad_center()
    v_end, w_end = sim.pad_target()
    w_avg = 0.5 * (sim.plat.w + w_end)
    psi_end = sim.plat.psi + w_avg * 0.1
    v = v_end * np.array([np.cos(psi_end), np.sin(psi_end), 0.0]) + np.array([-w_end * r[1], w_end * r[0], 0.0])
    yaw = np.arctan2(s["R"][1, 0], s["R"][0, 0])
    c, sn = np.cos(yaw), np.sin(yaw)
    cmd = np.array([c * v[0] + sn * v[1], -sn * v[0] + c * v[1], vz, 0.0])
    return np.clip(cmd / env.current_action_scale(np.sign(cmd)), -1.0, 1.0)


def load_policy(run, ckpt):
    import torch
    from v5_shin.policies.shin_policy import ShinPolicy
    cfg = json.load(open(os.path.join(RUNS, run, "config.json")))
    assert cfg["mode"] == "privileged", "diag_hold supports privileged (P) runs only"
    d = torch.load(os.path.join(RUNS, run, ckpt), map_location="cpu", weights_only=False)
    pol = ShinPolicy(cfg["mode"])
    pol.load_state_dict(d["policy"])
    pol.eval()
    return pol, cfg


def run_episode(env, seed, c, pol=None, oracle_vz=-0.3):
    env.sim.reseed(seed)
    obs, info = env.reset(options={"c": c, "on_pad": True})
    state = start = None
    if pol is not None:
        import torch
        state = pol.initial_state(1)
        start = torch.ones(1)
    while True:
        if pol is None:
            a = oracle_action(env, oracle_vz)
        else:
            import torch
            tob = {k: torch.as_tensor(np.asarray(obs[k]))[None] for k in KEYS}
            with torch.no_grad():
                mu, _, state = pol.actor_seq({k: v[None] for k, v in tob.items()}, state, start[None])
            a = mu[0, 0].numpy()
            start = torch.zeros(1)
        obs, r, te, tr, info = env.step(a)
        if te or tr:
            return info["outcome"], list(env.sim._vrel_hist)


def report(name, c, results):
    n = len(results)
    cnt = Counter(o for o, _ in results)
    print(f"\n[{name}] c = {c}: " + "  ".join(f"{k} {cnt[k]}" for k in OUTCOMES if cnt[k]))
    med = []
    for k in range(HOLD_STEPS):
        vals = [h[k] for _, h in results if len(h) > k]
        med.append(f"{np.median(vals):.2f}" if vals else "-")
    print(f"  median |v_rel| per hold step 1..{HOLD_STEPS} (m/s): " + " ".join(med))
    full = [max(h[-HOLD_STILL_STEPS:]) for _, h in results if len(h) >= HOLD_STEPS]
    if full:
        print(f"  episodes that stayed the full hold: {len(full)}/{n}; max |v_rel| in the last "
              f"{HOLD_STILL_STEPS} steps <= 0.1: {np.mean(np.array(full) <= 0.1):.2f}, "
              f"<= 0.2: {np.mean(np.array(full) <= 0.2):.2f}, <= 0.3: {np.mean(np.array(full) <= 0.3):.2f}, "
              f"median {np.median(full):.2f}")
    else:
        print(f"  no episode stayed the full hold (0/{n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="privileged run folder (omit for oracle only)")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--c", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--seed-base", type=int, default=30000)
    ap.add_argument("--dr-off", action="store_true", help="no push forces, no noise, no visual DR")
    ap.add_argument("--pad-smooth", action="store_true", help="D23 smooth pad speed changes")
    ap.add_argument("--hold-vz-down", type=float, default=None,
                    help="D23 sit-down limit in the hold, m/s (default: as the run, else off)")
    ap.add_argument("--friction", type=float, default=None,
                    help="effective drone-pad friction for this run (default: the sim's, 0.5)")
    ap.add_argument("--oracle-vz", type=float, nargs="+", default=[-0.3],
                    help="oracle vertical command(s), m/s (more negative = presses harder)")
    a = ap.parse_args()
    pol, cfg = load_policy(a.run, a.ckpt) if a.run else (None, {})
    env = ShinLandingEnv(mode="privileged", renderer="tiny", egl=False, seed=0,
                         dr=DRConfig.off() if a.dr_off else DRConfig(),
                         reward_fn=TouchdownReward("v4", cfg.get("vz_penalty", "literal")),
                         vz_max=cfg.get("vz_max", 1.0), hold_steps=HOLD_STEPS,
                         hold_vz_down=(a.hold_vz_down if a.hold_vz_down is not None
                                       else cfg.get("hold_vz_down")),
                         pad_smooth=a.pad_smooth)
    import pybullet as pb
    if a.friction is not None:
        pb.changeDynamics(env.sim.plat.body, -1, lateralFriction=1.0, physicsClientId=env.sim.cid)
        pb.changeDynamics(env.sim.quad.body, -1, lateralFriction=a.friction, physicsClientId=env.sim.cid)
    mu = (pb.getDynamicsInfo(env.sim.plat.body, -1, physicsClientId=env.sim.cid)[1]
          * pb.getDynamicsInfo(env.sim.quad.body, -1, physicsClientId=env.sim.cid)[1])
    print(f"diag_hold: run {a.run or '-'}  episodes {a.episodes}  DR {'off' if a.dr_off else 'on'}  "
          f"pad_smooth {a.pad_smooth}  hold_vz_down {env.hold_vz_down}  friction {mu:.2f}  "
          f"seeds {a.seed_base}+i")
    try:
        for c in a.c:
            seeds = [a.seed_base + i for i in range(a.episodes)]
            if pol is not None:
                report("policy", c, [run_episode(env, s, c, pol) for s in seeds])
            for vz in a.oracle_vz:
                report(f"oracle vz {vz:+.1f}", c, [run_episode(env, s, c, None, vz) for s in seeds])
    finally:
        env.close()


if __name__ == "__main__":
    main()
