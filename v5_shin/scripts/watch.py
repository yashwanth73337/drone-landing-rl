"""Watch landings (V5 viewer). Nothing here changes training or evaluation.

Two ways to watch:
  live window   PyBullet's 3D window (needs a display; run on the lab desktop itself)
  --video FILE  an MP4 file: left = chase camera, right = the drone's own grayscale camera,
                with a text overlay (method, episode seed, step, height above the pad,
                vertical speed, distance to the pad, platform speed, outcome)

Policies:
  --run NAME [--ckpt latest.pt]   a trained run from v5_shin/runs/NAME (variant P or A, any ablation)
  --oracle                        the scripted controller that reads the true state (Block 5)

Episodes are the SAME pinned episodes as evaluate.py: episode i uses seed seed_base + i, with
deterministic (mean) actions, so what you see is what the evaluation counted. The motor model and
the true-state injection can be switched exactly as in evaluate.py (--motor, --inject-true-state).

Live window: the drone is shown at its start point for 1.5 s, then flies in real time with the
camera following it; it pauses 2 s on touchdown. Outcomes are printed in the terminal.
--slow 2 plays at half speed; --hold keeps the window open at the end (for screen recording).

Usage (from ~/mtp/drone-landing-rl):
  python -m v5_shin.scripts.watch --run P_smoke_vz1_s1 --c 1.0 --episodes 3
  python -m v5_shin.scripts.watch --run A_s1 --c 0.5 --episodes 3 --video v5_shin/runs/A_s1/watch_c0.5.mp4
  python -m v5_shin.scripts.watch --oracle --c 1.0 --episodes 2 --video v5_shin/runs/oracle_c1.mp4
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import pybullet as p
import torch

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
W, H = 640, 400          # each half of the video frame


def load_policy(run, ckpt):
    from v5_shin.policies.shin_policy import ShinPolicy
    cfg = json.load(open(os.path.join(RUNS, run, "config.json")))
    d = torch.load(os.path.join(RUNS, run, ckpt), map_location="cpu", weights_only=False)
    pol = ShinPolicy(cfg["mode"])
    pol.load_state_dict(d["policy"])
    pol.eval()
    return pol, cfg, int(d.get("global_step", -1))


def _to_pixel(pt, view, proj):
    V = np.asarray(view).reshape(4, 4, order="F")
    P = np.asarray(proj).reshape(4, 4, order="F")
    c = P @ V @ np.r_[pt, 1.0]
    if c[3] <= 1e-6:
        return None
    x, y = c[0] / c[3], c[1] / c[3]
    return int((x + 1) / 2 * W), int((1 - y) / 2 * H)


def chase_image(sim):
    """Third-person view (TinyRenderer). The drone is marked with a red circle drawn on the
    IMAGE only, with a dashed line down to pad-top height: the scene itself is not changed,
    so the drone's own camera sees exactly what it sees in evaluation."""
    s = sim.quad.state()
    pad = sim.plat.pad_center()
    target = 0.65 * s["pos"] + 0.35 * pad
    dist = float(np.clip(3.0 + 0.6 * np.linalg.norm(s["pos"] - pad), 4.0, 10.0))
    yaw = np.degrees(sim.plat.psi) - 90.0          # look along the platform's direction of travel
    view = p.computeViewMatrixFromYawPitchRoll(target.tolist(), dist, yaw, -25.0, 0.0, 2,
                                               physicsClientId=sim.cid)
    proj = p.computeProjectionMatrixFOV(60.0, W / H, 0.1, 200.0, physicsClientId=sim.cid)
    _, _, rgba, _, _ = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER,
                                        flags=p.ER_NO_SEGMENTATION_MASK, physicsClientId=sim.cid)
    img = cv2.cvtColor(np.asarray(rgba, np.uint8).reshape(H, W, 4), cv2.COLOR_RGBA2BGR)
    d = _to_pixel(s["pos"], view, proj)
    g = _to_pixel(np.array([s["pos"][0], s["pos"][1], pad[2]]), view, proj)
    if d is not None:
        if g is not None:
            cv2.line(img, d, g, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(img, d, 14, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def overlay(img, lines, color=(255, 255, 255)):
    for i, t in enumerate(lines):
        y = 22 + 22 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return img


def frame(sim, title, seed, k, cmd, outcome=None, info=None):
    s = sim.quad.state()
    rel = s["pos"] - sim.plat.pad_center()
    left = chase_image(sim)
    onboard = cv2.cvtColor(cv2.resize(sim.render(), (W, H), interpolation=cv2.INTER_NEAREST),
                           cv2.COLOR_GRAY2BGR)
    lines = [title, f"episode seed {seed}   step {k} ({k * 0.1:.1f} s)",
             f"height of drone centre above pad top {rel[2]:5.2f} m",
             f"horizontal distance to pad centre   {np.hypot(rel[0], rel[1]):5.2f} m",
             f"drone vertical speed {s['v'][2]:+5.2f} m/s   commanded {cmd:+5.2f} m/s",
             f"platform speed {sim.plat.v:4.1f} m/s"]
    col = (255, 255, 255)
    if outcome is not None:
        vz = info.get("rel_vel", [np.nan] * 3)[2]
        lines.append(f"OUTCOME: {outcome}   touchdown vertical speed {vz:+.2f} m/s")
        col = (80, 255, 80) if outcome == "success" else (80, 80, 255)
    overlay(left, lines, col)
    overlay(onboard, ["drone camera (what the vision policy sees)"])
    return np.hstack([left, onboard])


class LiveView:
    """Live 3D window: real-time playback (refreshed after every 10 ms physics step), a camera
    that follows the drone, and short pauses at the start and end of each episode.
    The simulated drone's own shape is a plain 0.25 m slab; for the window only, it is dressed
    as a quadrotor (dark body, two arms, four rotor discs, the two FRONT rotors red). These
    parts are visual only (no mass, no collision) and sit behind/above the drone's camera,
    outside its field of view, so the policy sees exactly what it sees in evaluation."""

    PARTS = (  # (shape, size, local position, yaw deg, colour)
        ("box", (0.09, 0.09, 0.035), (0.0, 0.0, 0.0), 0, (0.15, 0.15, 0.15, 1)),
        ("box", (0.19, 0.012, 0.012), (0.0, 0.0, 0.02), 45, (0.25, 0.25, 0.25, 1)),
        ("box", (0.19, 0.012, 0.012), (0.0, 0.0, 0.02), -45, (0.25, 0.25, 0.25, 1)),
        ("cyl", (0.075, 0.008), (0.13, 0.13, 0.035), 0, (0.9, 0.1, 0.1, 1)),
        ("cyl", (0.075, 0.008), (0.13, -0.13, 0.035), 0, (0.9, 0.1, 0.1, 1)),
        ("cyl", (0.075, 0.008), (-0.13, 0.13, 0.035), 0, (0.05, 0.05, 0.05, 1)),
        ("cyl", (0.075, 0.008), (-0.13, -0.13, 0.035), 0, (0.05, 0.05, 0.05, 1)),
    )

    def __init__(self, sim, slow):
        self.sim, self.slow, self.cid = sim, slow, sim.cid
        for flag in (p.COV_ENABLE_GUI, p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                     p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
            p.configureDebugVisualizer(flag, 0, physicsClientId=self.cid)
        p.changeVisualShape(sim.quad.body, -1, rgbaColor=[0.15, 0.15, 0.15, 1], physicsClientId=self.cid)
        self.parts = []
        for shape, size, pos, yaw, rgba in self.PARTS:
            if shape == "box":
                v = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=rgba, physicsClientId=self.cid)
            else:
                v = p.createVisualShape(p.GEOM_CYLINDER, radius=size[0], length=size[1], rgbaColor=rgba,
                                        physicsClientId=self.cid)
            b = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=v,
                                  physicsClientId=self.cid)
            self.parts.append((b, np.array(pos), p.getQuaternionFromEuler([0, 0, np.radians(yaw)])))
        self.cam = None
        self._orig_step = p.stepSimulation
        p.stepSimulation = self._step            # this process only; restored in close()

    def close(self):
        p.stepSimulation = self._orig_step

    def _step(self, *args, **kw):
        r = self._orig_step(*args, **kw)
        self.follow()
        time.sleep(0.01 * self.slow)
        return r

    def follow(self):
        s = self.sim.quad.state()
        pos, quat = s["pos"], s["quat"]
        for b, lp, lq in self.parts:                 # move the quadrotor dressing with the drone
            wp, wq = p.multiplyTransforms(pos.tolist(), quat.tolist(), lp.tolist(), lq)
            p.resetBasePositionAndOrientation(b, wp, wq, physicsClientId=self.cid)
        # chase camera: 2.5 m behind and above the drone, looking the way the drone faces (towards the pad)
        yaw = float(p.getEulerFromQuaternion(quat.tolist())[2])
        cam = np.array([*pos, np.cos(yaw), np.sin(yaw)])
        self.cam = cam if self.cam is None else 0.85 * self.cam + 0.15 * cam   # smooth, no jitter
        fwd = self.cam[3:] / (np.linalg.norm(self.cam[3:]) + 1e-9)
        tgt = self.cam[:3] + np.array([0.5 * fwd[0], 0.5 * fwd[1], -0.6])
        p.resetDebugVisualizerCamera(2.5, np.degrees(np.arctan2(fwd[1], fwd[0])) - 90.0, -45.0,
                                     tgt.tolist(), physicsClientId=self.cid)

    def start_episode(self, title, seed):
        self.cam = None
        self.follow()
        time.sleep(1.5 * self.slow)              # show the drone at its start point in the air

    def status(self, k, cmd_vz):
        pass

    def end_episode(self, outcome, vz):
        time.sleep(2.0 * self.slow)              # pause on the touchdown


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--oracle", action="store_true")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--c", type=float, default=1.0, help="difficulty (platform motion), 0..1")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--video", default=None, help="write an MP4 instead of opening a window")
    ap.add_argument("--slow", type=float, default=1.0, help="live window: 1 = real time, 2 = half speed")
    ap.add_argument("--motor", choices=["asym", "sym_slow", "sym_fast"], default=None)
    ap.add_argument("--inject-true-state", action="store_true")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--hold", action="store_true", help="live window: keep it open at the end until Enter")
    a = ap.parse_args()

    from v5_shin.envs.reward import ShinReward
    from v5_shin.envs.shin_env import ShinLandingEnv
    from v5_shin.policies.oracle import OracleController

    if a.oracle:
        pol, cfg, step = None, {"mode": "privileged"}, 0
        title = "scripted oracle (true state)"
    else:
        pol, cfg, step = load_policy(a.run, a.ckpt)
        title = f"{a.run} / {a.ckpt} ({step:,} steps)" + ("  + TRUE STATE INJECTED" if a.inject_true_state else "")
    motor = a.motor or cfg.get("motor", "asym")
    if motor != "asym":
        title += f"  motors {motor}"
    title += f"  c={a.c}"
    live = a.video is None
    env = ShinLandingEnv(mode=cfg["mode"], renderer="tiny", egl=False, seed=0, c=a.c,
                         reward_fn=ShinReward(cfg.get("vz_penalty", "literal")),
                         vz_max=cfg.get("vz_max", 3.0), motor_mode=motor, gui=live)
    view = LiveView(env.sim, a.slow) if live else None
    if view:
        view.title_txt = title
    writer, path = None, a.video
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (2 * W, H))
        if not writer.isOpened():
            path = os.path.splitext(path)[0] + ".avi"
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (2 * W, H))
    keys = ("image", "u", "critic", "target", "s_rel") if cfg["mode"] == "vision" else ("u", "critic", "target", "s_rel")
    oracle = OracleController()
    try:
        for i in range(a.episodes):
            seed = a.seed_base + i
            env.sim.reseed(seed)
            obs, info = env.reset(options={"c": a.c})
            state = pol.initial_state(1) if pol is not None else None
            start = torch.ones(1)
            outcome, k, cmd_vz = None, 0, 0.0
            if view:
                view.start_episode(title, seed)
            while k < a.max_steps:
                if pol is None:
                    rel_p, rel_v, s = env.sim.true_relative_state()
                    yaw = np.arctan2(s["R"][1, 0], s["R"][0, 0])
                    cmd = oracle(rel_p, rel_v, env.sim.plat.velocity(), yaw)
                    te, tr, info = env.sim.step(cmd)
                else:
                    tob = {kk: torch.as_tensor(np.asarray(obs[kk]))[None, None] for kk in keys}
                    with torch.no_grad():
                        mu, _, state = pol.actor_seq(tob, state, start[None],
                                                     inject_true_state=a.inject_true_state)
                    obs, _, te, tr, info = env.step(mu[0, 0].numpy())
                    cmd = info["action_cmd"]
                    start = torch.zeros(1)
                cmd_vz = float(cmd[2])
                k += 1
                done = te or tr
                if done:
                    outcome = info["outcome"]
                if writer is not None:
                    f = frame(env.sim, title, seed, k, cmd_vz, outcome, info)
                    for _ in range(10 if done else 1):          # hold the last frame 1 s
                        writer.write(f)
                else:
                    view.status(k, cmd_vz)
                if done:
                    break
            vz = info.get("rel_vel", [np.nan] * 3)[2] if outcome else float("nan")
            print(f"episode seed {seed}: {outcome or 'stopped'} after {k} steps ({k * 0.1:.1f} s), "
                  f"touchdown vertical speed {vz:+.2f} m/s", flush=True)
            if view:
                view.end_episode(outcome, vz)
        if view and a.hold:
            input("Done. Press Enter to close the window... ")
    finally:
        if view:
            view.close()
        if writer is not None:
            writer.release()
            print(f"video written: {os.path.abspath(path)}")
        env.close()


if __name__ == "__main__":
    main()
