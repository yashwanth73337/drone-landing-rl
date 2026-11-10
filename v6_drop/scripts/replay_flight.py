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

Run (from the repo root), e.g. the missed touch of ppo2 seed 2 in touch9500:
  python -m v6_drop.scripts.replay_flight --csv v6_drop/results/final_eval/touch9500/flights_ppo2_s2.csv \\
      --ckpt v6_drop/results/runs/ppo2_s2/r0299.pt --touch sensor --seeds 9520
Output next to the csv: replay_<student>_<seed>.mp4 / .png
"""
import argparse
import csv
import os

import torch

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
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


class Recorder:
    """Builds the same frames and log as eval_student.fly(frames=True), only by looking."""

    def __init__(self, env, film=True):
        self.env, self.film, self.log, self.pics = env, film, [], []

    def before(self, env, act, p):
        rec = ES.snapshot(env, f"student (drop chance {100 * p:.0f}%)", env.world.t * P.POLICY_DT)
        rec["note"], rec["note_color"] = ES.drop_note(env, act)
        if self.film:
            self.pics.append(ST.compose_frame(env, rec, outcome=None))
        self.log.append(rec)

    def after(self, env, info):
        rt = info.get("reward_terms") or {}
        self.log[-1]["seen"] = rt.get("seen")

    def motors_off(self, k):
        env = self.env
        if self.film and (k <= ES.SLOWMO_STEPS or k % 10 == 0):
            slow = k <= ES.SLOWMO_STEPS
            r = ES.snapshot(env, "MOTORS OFF" + (" (slow motion x10)" if slow else ": 2 s landing check"),
                            env.world.t * P.POLICY_DT + k * P.PHYSICS_DT)
            r["note"], r["note_color"] = "motors off: falling onto the pad" if slow else None, ST.YELLOW
            self.pics.append(ST.compose_frame(env, r, outcome=None))


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
    try:
        row = fly_dice(env, model, seed, "clean", sigma_norm(), before=rec.before, after=after)
    finally:
        env.world.frame_hook = None
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
                pics = rec.pics + [ST.compose_frame(env, dict(rec.log[-1], pos=s["pos"].copy(), note=None),
                                                    outcome=info)] * 20
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
