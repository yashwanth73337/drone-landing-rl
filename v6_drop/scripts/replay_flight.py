"""v6_drop T6 (10 Nov): REPLAY and film single flights of a final_eval run, EXACTLY.

Why: eval_student.py (videos) and final_eval.py (the test) run the student's network with a
different number of CPU threads (final_eval workers: 1). That changes the network's output by
~1e-8, and in a closed loop of ~60 decisions such tiny differences can grow: an edge case
(e.g. seed 9520, ppo2 seed 2, very slow pad) timed out in the test but landed in 5 s in the
eval_student video. So a video from eval_student is a real flight, but not THE flight of the
table. This script flies the seed with final_eval's own code path: dice_exam.fly_dice in
'clean' mode, 1 CPU thread, the student's shield from the csv, the same touch mode, a fresh
LandingEnv. Filming only LOOKS (renders, reads): fly_dice's before/after hooks.

Every replay is CHECKED: all columns of the new row must equal the csv row (outcome, reason,
impact, time, touches, delay, ...). MATCH -> the video is that flight. MISMATCH -> the fields
that differ are printed, and the files are still written (named ..._MISMATCH).

N1 (10 Nov): --noise LEVEL must match the final_eval run's --noise (default none).

V1 (12 Nov): the video shows what happens INSIDE the drone, because a normal landing video
looks the same with or without the touch detector or noise. Every frame gets a third panel:
the drone's own camera height, own vertical speed, the touch detector's SEE / FEEL / HOLD
(a bar that fills 1..10), MOTORS ON / OFF and why; with --noise also "drone reads" vs "true"
speed and the tilt error; at the bottom the TRUTH (for us only): true feet height and whether
a foot touches. The drone-camera panel shows the picture the drone really used (with camera
grain when noise is on). Near a touch (sensor mode) one frame per 0.01 s physics step is shown
(slow motion), so the 0.1 s hold is visible. All of it only LOOKS: World.step_hook is called
after each motors-on physics step and renders / reads, the flight is unchanged (tested).

Run (from the repo root), e.g. the missed touch of ppo2 seed 2 in touch9500:
  python -m v6_drop.scripts.replay_flight --csv v6_drop/results/final_eval/touch9500/flights_ppo2_s2.csv \\
      --ckpt v6_drop/results/runs/ppo2_s2/r0299.pt --touch sensor --seeds 9520
Output next to the csv: replay_<student>_<seed>.mp4 / .png
"""
import argparse
import csv
import os

import cv2
import numpy as np
import torch

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.scripts import eval_student as ES
from v6_drop.scripts import show_teacher as ST
from v6_drop.scripts.dice_exam import fly_dice, sigma_norm
from v6_drop.scripts.train_ppo import touch_kw

SKIP = ("student", "shield")          # added by final_eval, not by fly_dice


def as_csv(v):
    """How csv.DictWriter writes a value."""
    return "" if v is None else str(v)


def compare(row, ref):
    """Fields where the new row differs from the csv row: {name: (csv, replay)}."""
    return {k: (ref[k], as_csv(row.get(k))) for k in ref if k not in SKIP and as_csv(row.get(k)) != ref[k]}


PANEL_W = 430
STRIP_H = 80                     # info strip under the chase + drone camera panels
GREY = (170, 170, 170)


def text(img, t, y, color=ST.WHITE, scale=0.5, x=10, thick=1):
    cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def drone_cam_panel(env, motors_on):
    """The picture the drone really used (last frame, with grain if noise is on) while the motors
    are on; a fresh render after the cut (the drone no longer flies)."""
    img = env.last_frame if (motors_on and getattr(env, "last_frame", None) is not None) else env.world.render()
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    corners, ids, _ = env.world.det.detectMarkers(img)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(bgr, corners, ids, borderColor=(0, 255, 0))
    n = 0 if ids is None else len(ids)
    ST.put(bgr, "drone camera (what the drone used)" if motors_on else "drone camera", 20, ST.WHITE)
    ST.put(bgr, f"markers seen: {n}", 42, ST.GREEN if n else ST.RED)
    if env.noise is not None and motors_on:
        ST.put(bgr, f"with camera grain (noise {env.noise.sp:.0f} grey levels)", 64, ST.YELLOW)
        hh, ww = img.shape
        crop = bgr[hh // 2 - 40:hh // 2 + 40, ww // 2 - 64:ww // 2 + 64]          # zoom so the grain shows
        z = cv2.resize(crop, (256, 160), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(z, (0, 0), (255, 159), ST.YELLOW, 2)
        bgr[hh - 164:hh - 4, ww - 260:ww - 4] = z
        ST.put(bgr, "zoom x2: camera grain", hh - 170, ST.YELLOW, 0.45)
    return bgr


def mind_panel(env, state):
    """'Inside the drone': what it senses, the touch detector, the motors; the truth below."""
    w = env.world
    h = ST.CHASE_H + STRIP_H
    img = np.full((h, PANEL_W, 3), 25, np.uint8)
    text(img, "INSIDE THE DRONE", 24, ST.YELLOW, 0.65, thick=2)
    text(img, "(only its own sensors)", 44, GREY, 0.45)
    y = 70
    det = w.detector
    vz_own = w.own_vz()
    if det is not None:
        fe = det.feet_est()
        if fe is None:
            text(img, "feet height: unknown (no marker)", y)
        elif det.age <= 10:
            text(img, f"feet height (camera): {100 * fe:5.1f} cm", y)
        else:
            text(img, f"feet height (guess): {100 * fe:5.1f} cm", y, ST.YELLOW)
            text(img, f"camera blind {det.age * P.PHYSICS_DT:.2f} s, using own sinking", y + 18, ST.YELLOW, 0.42)
            y += 18
        text(img, f"own vertical speed: {vz_own:+.2f} m/s", y + 22)
        y += 54
        text(img, "TOUCH DETECTOR", y, ST.YELLOW, 0.55, thick=2)
        see = fe is not None and fe < det.feet_max
        feel = abs(vz_own) < det.vz_max
        text(img, f"SEE   feet < {100 * det.feet_max:.0f} cm", y + 24)
        text(img, "YES" if see else "no", y + 24, ST.GREEN if see else ST.RED, 0.55, x=300, thick=2)
        text(img, f"FEEL  |speed| < {det.vz_max:.2f} m/s", y + 46)
        text(img, "YES" if feel else "no", y + 46, ST.GREEN if feel else ST.RED, 0.55, x=300, thick=2)
        text(img, f"HOLD  both for {det.hold_steps * P.PHYSICS_DT:.1f} s:", y + 68)
        bx, by, bw = 215, y + 56, 150
        cv2.rectangle(img, (bx, by), (bx + bw, by + 14), GREY, 1)
        frac = min(det.count, det.hold_steps) / det.hold_steps
        cv2.rectangle(img, (bx + 1, by + 1), (bx + 1 + int((bw - 2) * frac), by + 13), ST.GREEN, -1)
        text(img, f"{min(det.count, det.hold_steps)}/{det.hold_steps}", y + 68, ST.WHITE, 0.45, x=bx + bw + 4)
        y += 92
    else:
        text(img, f"own vertical speed: {vz_own:+.2f} m/s", y)
        y += 32
    col = ST.GREEN if state["motors"] == "ON" else ST.RED
    text(img, f"MOTORS {state['motors']}", y, col, 0.7, thick=2)
    if state.get("why"):
        text(img, state["why"], y + 22, col, 0.48)
    yt = h - 48
    if env.noise is not None:                        # V1: own vs true vertical speed (the FEEL input)
        s = w.true_state()
        text(img, f"own speed is off by {vz_own - s['v'][2]:+.2f} m/s (noise)", yt - 30, ST.YELLOW, 0.45)
    cv2.line(img, (8, yt - 18), (PANEL_W - 8, yt - 18), GREY, 1)
    text(img, "TRUTH (for us only; the drone cannot see it)", yt - 2, GREY, 0.42)
    tf = w._lowest_z() - PAD_TOP_Z
    touching = w._contact_kind() == "top"
    text(img, f"true feet height {100 * max(tf, 0):.1f} cm", yt + 20, GREY, 0.45)
    text(img, "foot touches pad: " + ("YES" if touching else "no"), yt + 38,
         ST.GREEN if touching else GREY, 0.45)
    return img


class Recorder:
    """Builds the frames and log of a flight, only by looking. V1: three panels per frame (chase
    camera | drone camera | inside the drone), slow motion near a touch."""

    def __init__(self, env, film=True):
        self.env, self.film, self.log, self.pics = env, film, [], []
        self.state = dict(motors="ON", why=None)
        self.touched = False

    def frame(self, rec, outcome=None):
        env = self.env
        on = self.state["motors"] == "ON"
        base = ST.compose_frame(env, rec, outcome=outcome)
        base[:, ST.CHASE_W:] = drone_cam_panel(env, on)
        strip = np.full((STRIP_H, base.shape[1], 3), 40, np.uint8)
        if env.noise is not None:
            s = env.world.true_state()
            _, v_own = env.own()
            text(strip, f"SENSOR NOISE ({env.noise.level}), never seen in training:", 24, ST.YELLOW, 0.6, thick=2)
            text(strip, f"own speed: drone reads {v_own[2]:+.2f} m/s, truth {s['v'][2]:+.2f} m/s   |   "
                        f"own tilt off by {np.degrees(np.linalg.norm(env.noise.r_err)):.1f} deg   |   "
                        f"camera grain {env.noise.sp:.0f} grey levels", 52, ST.WHITE, 0.52)
        else:
            text(strip, "No sensor noise. The drone uses only its camera and its own motion sensors.", 30, GREY, 0.55)
            text(strip, "The touch detector (right) decides 'landed' and cuts the motors.", 56, GREY, 0.55)
        return np.hstack([np.vstack([base, strip]), mind_panel(env, self.state)])

    def before(self, env, act, p):
        rec = ES.snapshot(env, f"student (drop chance {100 * p:.0f}%)", env.world.t * P.POLICY_DT)
        rec["note"], rec["note_color"] = ES.drop_note(env, act)
        if act["drop"] and rec["note_color"] == ST.GREEN:
            self.state = dict(motors="OFF", why="the student pressed DROP")
        if self.film:
            self.pics.append(self.frame(rec))
        self.log.append(rec)

    def after(self, env, info):
        rt = info.get("reward_terms") or {}
        self.log[-1]["seen"] = rt.get("seen")

    def physics_step(self, k):
        """V1: after each motors-on physics step. Slow motion while the drone is on or near the pad."""
        env = self.env
        w = env.world
        det = w.detector
        touching = w._contact_kind() == "top"
        note = None
        if touching and not self.touched:
            note = "FEET TOUCH THE PAD (motors still on)"
        self.touched = self.touched or touching
        fired = det is not None and det.count >= det.hold_steps
        if fired:
            self.state = dict(motors="OFF", why="cut by the TOUCH DETECTOR")
            note = "TOUCH DETECTOR: landed -> MOTORS OFF"
        near = touching or (det is not None and det.count > 0) or fired
        if self.film and near and w.touch == "sensor":
            r = ES.snapshot(env, "slow motion x10 (one frame per 0.01 s)", k * P.PHYSICS_DT)
            r["note"], r["note_color"] = note, (ST.RED if fired else ST.YELLOW)
            self.pics.append(self.frame(r))

    def motors_off(self, k):
        env = self.env
        if self.state["motors"] == "ON":                       # perfect mode auto-cut
            self.state = dict(motors="OFF", why="simulator auto-cut (old mode)")
        if self.film and (k <= ES.SLOWMO_STEPS or k % 10 == 0):
            slow = k <= ES.SLOWMO_STEPS
            r = ES.snapshot(env, "MOTORS OFF" + (" (slow motion x10)" if slow else ": 2 s landing check"),
                            env.world.t * P.POLICY_DT + k * P.PHYSICS_DT)
            detcut = self.state.get("why") == "cut by the TOUCH DETECTOR"
            r["note"] = (("motors off: resting on the pad" if detcut else "motors off: falling onto the pad")
                         if slow else None)
            r["note_color"] = ST.YELLOW
            self.pics.append(self.frame(r))


def replay(env, model, seed, ref, film=True):
    """Fly one seed exactly like final_eval; returns (row, differences, recorder, info)."""
    shield = float(ref["shield"]) if ref.get("shield") else None
    env.drop_shield = shield
    rec = Recorder(env, film)
    holder = {}

    def after(e, info):
        rec.after(e, info)
        holder["info"] = info

    env.world.frame_hook = rec.motors_off
    env.world.step_hook = rec.physics_step if film else None
    try:
        row = fly_dice(env, model, seed, "clean", sigma_norm(), before=rec.before, after=after)
    finally:
        env.world.frame_hook = None
        env.world.step_hook = None
    return row, compare(row, ref), rec, holder["info"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="final_eval flights_<student>.csv")
    ap.add_argument("--ckpt", required=True, help="the checkpoint of that student")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--touch", choices=["perfect", "sensor"], default="perfect",
                    help="must be the touch mode of the final_eval run")
    ap.add_argument("--touch-hold", type=int, default=10)
    ap.add_argument("--noise", choices=["none", "low", "medium", "high"], default="none",
                    help="N1: sensor noise level for the drone's own estimates (envs/sensor_noise.py)")
    ap.add_argument("--renderer", choices=["egl", "tiny"], default="egl",
                    help="must be the renderer of the final_eval run (default egl)")
    ap.add_argument("--no-video", action="store_true", help="only check the match (fast)")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)                                   # as final_eval's workers
    rows = {int(r["seed"]): r for r in csv.DictReader(open(a.csv))}
    some = next(iter(rows.values()))
    if ("touches" in some) != (a.touch == "sensor"):
        raise SystemExit(f"{a.csv} was made with touch={'sensor' if 'touches' in some else 'perfect'}; "
                         f"pass the same --touch")
    model, _ = ES.load(a.ckpt, "cpu")
    student = os.path.basename(a.csv)[len("flights_"):-len(".csv")]
    out = os.path.dirname(os.path.abspath(a.csv))
    env = E.LandingEnv(renderer=a.renderer, seed=0, **touch_kw(a.touch, a.touch_hold), **ES.noise_kw(a.noise)[0])
    n_match = 0
    try:
        for seed in a.seeds:
            if seed not in rows:
                print(f"[replay] seed {seed}: not in {a.csv}", flush=True)
                continue
            ref = rows[seed]
            row, diff, rec, info = replay(env, model, seed, ref, film=not a.no_video)
            n_match += not diff
            tag = "MATCH" if not diff else "MISMATCH"
            print(f"[replay] {student} seed {seed}: {tag} | outcome {row['outcome']} ({row['reason']}), "
                  f"impact {row['impact']}, time {row['time_s']} s"
                  + ("" if a.touch == "perfect" else f", touches {row.get('touches')}, delay {row.get('detect_delay_s')}"),
                  flush=True)
            for k, (c, r) in diff.items():
                print(f"           {k}: csv {c!r} vs replay {r!r}", flush=True)
            if not a.no_video:
                base = os.path.join(out, f"replay_{student}_{seed}" + ("" if not diff else "_MISMATCH"))
                s = env.world.true_state()
                pics = rec.pics + [rec.frame(dict(rec.log[-1], pos=s["pos"].copy(), note=None), outcome=info)] * 20
                ST.write_video(pics, base + ".mp4")
                ST.plot_episode(rec.log, info, seed, float(ref["pad_speed"]), base + ".png",
                                title=f"Replay {student}")
                print(f"           -> {base}.mp4/.png", flush=True)
    finally:
        env.close()
    print(f"[replay] {n_match}/{len(a.seeds)} match the csv", flush=True)
    return n_match


if __name__ == "__main__":
    main()
