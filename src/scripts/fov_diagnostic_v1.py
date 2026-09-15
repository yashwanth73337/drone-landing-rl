"""
fov_diagnostic_v1.py
====================

Stage V1 diagnostics for VisionLanderAviary. Trains nothing, saves no model,
and never touches LanderAIAviary.py, train_lander.py or the six trusted
checkpoints.

Four modes, intended to be run in this order:

    --mode equivalence   GATE. Proves the camera and the visual marker did
                         not perturb the physics. Everything downstream is
                         meaningless if this fails, so run it first.

    --mode frames        Saves rendered camera images at specified heights
                         and horizontal offsets, including the worst cases
                         the curriculum can actually produce.

    --mode sweep         Analytic closed-form + measured dropout heights
                         across marker sizes and FOVs. No rendering, so it
                         is cheap enough to explore the design space before
                         committing compute.

    --mode episodes      The real measurement: fly episodes and record, per
                         step, whether the marker is visible, why it is not,
                         and where it sits in the image.

Usage
-----
    python fov_diagnostic_v1.py --mode equivalence
    python fov_diagnostic_v1.py --mode frames  --outdir v1_out
    python fov_diagnostic_v1.py --mode sweep
    python fov_diagnostic_v1.py --mode episodes \
        --model results_lander/moving_025_seed1_final.zip \
        --episodes 50 --platform-speed 0.25 --render

POLICY SOURCE
-------------
--model PATH loads an SB3 TD3 checkpoint. That is what the reported numbers
must ultimately come from.

--policy scripted uses a proportional controller on the same privileged
observation. It exists so the instrumentation itself can be validated
without a checkpoint present, and so equivalence can be tested with a policy
that is guaranteed identical between the two environments. Numbers produced
with the scripted policy describe the SCRIPTED policy's trajectories, not
the trained one's, and are labelled as such in every output.
"""

import os
import sys
import csv
import json
import argparse
from collections import Counter

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.LanderAIAviary import LanderAIAviary            # noqa: E402
from envs.VisionLanderAviary import VisionLanderAviary    # noqa: E402


# ---------------------------------------------------------------------------
# Environment configuration.
#
# These mirror train_lander.py's ENV_KWARGS plus the curriculum's final level
# (RadiusCurriculumCallback.LEVELS[-1] = spawn radius 0.11-0.30 m, platform
# speed 0.25 m/s). The trusted moving_025 checkpoints were trained and scored
# at exactly this distribution, so the diagnostic must reproduce it or the
# visibility statistics describe a task nobody trained on.
# ---------------------------------------------------------------------------
BASE_ENV_KWARGS = {
    "tilt_limit": 1.0,             # train_lander.py ENV_KWARGS
    "spawn_radius_min": 0.11,      # RadiusCurriculumCallback.radius_min
    "spawn_radius_max": 0.30,      # LEVELS[-1][0]
    "platform_speed_range": 0.25,  # LEVELS[-1][1]
}

VISION_KWARGS = {
    "camera_res": (128, 128),
    "camera_fov_deg": 90.0,
    "marker_size": 0.25,
    "camera_mode": "geometry",
}


# ===========================================================================
# POLICIES
# ===========================================================================

class ScriptedPolicy:
    """Proportional descent controller on the privileged observation.

    obs layout is LanderAIAviary Eq. 3: [theta(3), v(3), omega(3), d(3),
    delta_v(3)], each normalised. d is divided by rel_pos_bound = 5.0, so the
    true relative position is obs[9:12] * 5.0.

    Horizontal gain is higher than vertical so the drone aligns before it
    descends, which is qualitatively what the trained policy does (the reward
    carries an explicit HORIZONTAL_ALPHA = 5.0 term for exactly this).
    """

    name = "scripted-proportional"

    def __init__(self, k_xy=8.0, k_z=3.0, rel_pos_bound=5.0):
        self.k_xy = k_xy
        self.k_z = k_z
        self.rel_pos_bound = rel_pos_bound

    def predict(self, obs, deterministic=True):
        d = np.asarray(obs, dtype=np.float64)[9:12] * self.rel_pos_bound
        a = np.array([
            self.k_xy * d[0],
            self.k_xy * d[1],
            self.k_z * d[2],
        ])
        return np.clip(a, -1.0, 1.0).astype(np.float32), None


class ReplayPolicy:
    """Emits a fixed, pre-generated action sequence, ignoring observations.

    This is the right instrument for the equivalence gate: it removes any
    possibility that a trajectory difference came from the two environments
    being fed different actions. Any divergence must then be physics.
    """

    name = "fixed-action-sequence"

    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.i = 0

    def reset(self):
        self.i = 0

    def predict(self, obs, deterministic=True):
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return a, None


def load_policy(args):
    """Return (policy, label)."""
    if args.model:
        from stable_baselines3 import TD3
        model = TD3.load(args.model)
        return model, f"TD3 checkpoint: {os.path.basename(args.model)}"
    return ScriptedPolicy(), f"SCRIPTED proportional policy (no checkpoint)"


# ===========================================================================
# MODE 1 -- PHYSICS EQUIVALENCE GATE
# ===========================================================================

def mode_equivalence(args):
    """Run matched seeded episodes in both environments and compare.

    Two policies are used:

      1. A fixed action sequence, identical in both environments. This is the
         strict test -- observations cannot feed back, so any divergence is
         unambiguously physics.

      2. The scripted closed-loop policy. This is the realistic test: small
         physics differences would be amplified through feedback, so it
         catches drift that an open-loop rollout might hide.
    """
    print("=" * 74)
    print("  MODE: PHYSICS EQUIVALENCE GATE")
    print("=" * 74)
    print("  baseline : LanderAIAviary          (no camera, no marker)")
    print("  candidate: VisionLanderAviary      (camera + visual ArUco marker)")
    print(f"  episodes : {args.episodes}")
    print()

    vision_kwargs = dict(VISION_KWARGS)
    vision_kwargs["camera_mode"] = args.camera_mode

    rng = np.random.default_rng(20260912)
    horizon = 400

    results = []

    for test_name, use_fixed in (("fixed-action-sequence", True),
                                 ("closed-loop-scripted", False)):
        print(f"--- test: {test_name} " + "-" * (56 - len(test_name)))

        worst = {
            "pos": 0.0, "vel": 0.0, "plat": 0.0,
            "reward": 0.0, "obs": 0.0,
        }
        mismatched_term = 0
        mismatched_success = 0
        n_steps_total = 0

        for ep in range(args.episodes):
            seed = 5000 + ep

            if use_fixed:
                actions = rng.uniform(-1.0, 1.0, size=(horizon, 3))
                pol_a = ReplayPolicy(actions)
                pol_b = ReplayPolicy(actions)
            else:
                pol_a = ScriptedPolicy()
                pol_b = ScriptedPolicy()

            traj_a = _rollout_record(
                LanderAIAviary(**BASE_ENV_KWARGS), pol_a, seed, horizon
            )
            traj_b = _rollout_record(
                VisionLanderAviary(**BASE_ENV_KWARGS, **vision_kwargs),
                pol_b, seed, horizon
            )

            n = min(len(traj_a["pos"]), len(traj_b["pos"]))
            n_steps_total += n

            for key in ("pos", "vel", "plat", "obs"):
                a = np.asarray(traj_a[key][:n])
                b = np.asarray(traj_b[key][:n])
                d = float(np.max(np.abs(a - b))) if n else 0.0
                worst[key] = max(worst[key], d)

            ra = np.asarray(traj_a["reward"][:n])
            rb = np.asarray(traj_b["reward"][:n])
            worst["reward"] = max(
                worst["reward"], float(np.max(np.abs(ra - rb))) if n else 0.0
            )

            if traj_a["n_steps"] != traj_b["n_steps"]:
                mismatched_term += 1
            if traj_a["success"] != traj_b["success"]:
                mismatched_success += 1

        tol = 1e-9
        passed = (
            worst["pos"] <= tol and worst["vel"] <= tol
            and worst["plat"] <= tol and worst["reward"] <= tol
            and worst["obs"] <= tol
            and mismatched_term == 0 and mismatched_success == 0
        )

        print(f"  steps compared              : {n_steps_total}")
        print(f"  max |delta drone position|  : {worst['pos']:.3e} m")
        print(f"  max |delta drone velocity|  : {worst['vel']:.3e} m/s")
        print(f"  max |delta platform pos|    : {worst['plat']:.3e} m")
        print(f"  max |delta observation|     : {worst['obs']:.3e}")
        print(f"  max |delta reward|          : {worst['reward']:.3e}")
        print(f"  termination-step mismatches : {mismatched_term}"
              f"/{args.episodes}")
        print(f"  success-outcome mismatches  : {mismatched_success}"
              f"/{args.episodes}")
        print(f"  RESULT                      : "
              f"{'PASS (bit-identical)' if passed else 'FAIL'}")
        print()

        results.append({
            "test": test_name, "passed": bool(passed),
            "episodes": args.episodes, "steps": n_steps_total,
            **{f"max_delta_{k}": v for k, v in worst.items()},
            "termination_mismatches": mismatched_term,
            "success_mismatches": mismatched_success,
        })

    all_passed = all(r["passed"] for r in results)
    print("=" * 74)
    print(f"  GATE: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 74)

    _write_json(args, "equivalence_results.json", results)
    return all_passed


def _rollout_record(env, policy, seed, horizon):
    """Roll one episode, recording everything the equivalence gate compares."""
    obs, info = env.reset(seed=seed)
    rec = {"pos": [], "vel": [], "plat": [], "reward": [], "obs": [],
           "success": False, "n_steps": 0}
    try:
        for _ in range(horizon):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            state = env._getDroneStateVector(0)
            rec["pos"].append(np.asarray(state[0:3], dtype=np.float64).copy())
            rec["vel"].append(np.asarray(state[10:13], dtype=np.float64).copy())
            rec["plat"].append(
                np.asarray(env._getLandingTargetPos(), dtype=np.float64).copy()
            )
            rec["obs"].append(np.asarray(obs, dtype=np.float64).copy())
            rec["reward"].append(float(reward))
            rec["n_steps"] += 1

            if terminated or truncated:
                rec["success"] = bool(info.get("success", False))
                break
    finally:
        env.close()
    return rec


# ===========================================================================
# MODE 2 -- CAMERA FRAMES
# ===========================================================================

def mode_frames(args):
    """Save rendered camera frames at a grid of heights and offsets.

    The offsets are the ones the task can actually produce: 0 (centred),
    0.11 m (curriculum minimum spawn radius) and 0.30 m (curriculum maximum
    -- the worst case the trained policy ever starts from).
    """
    import cv2

    print("=" * 74)
    print("  MODE: CAMERA FRAMES")
    print("=" * 74)

    outdir = os.path.join(args.outdir, "frames")
    os.makedirs(outdir, exist_ok=True)

    vision_kwargs = dict(VISION_KWARGS)
    vision_kwargs.update({
        "camera_mode": "render",
        "keep_image": True,
        "camera_res": tuple(args.res),
        "camera_fov_deg": args.fov,
        "marker_size": args.marker_size,
    })

    env = VisionLanderAviary(**BASE_ENV_KWARGS, **vision_kwargs)
    env.reset(seed=0)

    # The pad spawns on the curriculum ANNULUS at a random azimuth, so it is
    # NOT at the world origin. Offsets must be measured from the live pad
    # position or they are not offsets from anything meaningful.
    pad = np.asarray(env._getLandingTargetPos(), dtype=float)
    pad_top = float(pad[2])
    heights = [float(h) for h in args.heights]
    offsets = [float(o) for o in args.offsets]

    print(f"  resolution   : {tuple(args.res)}")
    print(f"  FOV          : {args.fov} deg")
    print(f"  marker size  : {args.marker_size} m")
    print(f"  pad top at z = {pad_top:.4f} m")
    print()
    print(f"  {'h_above_pad':>12} {'offset':>8} {'detected':>9} "
          f"{'px/cell':>8} {'corners':>8} {'pnp_err_m':>10}  file")
    print("  " + "-" * 70)

    rows = []
    try:
        for h in heights:
            for off in offsets:
                # Teleport the drone to a pose defined RELATIVE TO THE PAD.
                # The platform is kinematic and stays put between steps, so
                # this samples camera geometry directly without needing a
                # policy to fly there.
                pos = [pad[0] + off, pad[1], pad_top + h]
                _teleport(env, pos)

                info = env._computeInfo()
                gray = info.get("cam_image")

                name = f"h{h:.3f}_off{off:.2f}.png".replace("-", "m")
                path = os.path.join(outdir, name)
                cv2.imwrite(path, gray)
                # Also save an upscaled copy: 128x128 is hard to eyeball.
                cv2.imwrite(
                    os.path.join(outdir, "big_" + name),
                    cv2.resize(gray, (512, 512),
                               interpolation=cv2.INTER_NEAREST),
                )

                det = info.get("cam_aruco_detected", False)
                row = {
                    "height_above_pad": h,
                    "horizontal_offset": off,
                    "detected": bool(det),
                    "px_per_cell": round(info["cam_px_per_cell"], 2),
                    "corners_in_frame": info["cam_corners_in_frame"],
                    "fully_in_frame": info["cam_fully_in_frame"],
                    "centre_in_frame": info["cam_centre_in_frame"],
                    "loss_cause": info["cam_loss_cause"],
                    "pnp_err_m": info.get("cam_aruco_pos_err", float("nan")),
                    "file": name,
                }
                rows.append(row)

                err = row["pnp_err_m"]
                err_s = "  -" if not np.isfinite(err) else f"{err:9.4f}"
                print(f"  {h:12.3f} {off:8.2f} {str(det):>9} "
                      f"{row['px_per_cell']:8.1f} "
                      f"{row['corners_in_frame']:8d} {err_s}  {name}")
    finally:
        env.close()

    _write_csv(args, "frames_summary.csv", rows)
    print()
    print(f"  frames written to {outdir}/")
    return rows


def _teleport(env, pos, rpy=(0.0, 0.0, 0.0)):
    """Move the drone and refresh the cached kinematics BaseAviary reads.

    _getDroneStateVector() reads env.pos/quat/rpy/vel/ang_v, which are only
    refreshed by _updateAndStoreKinematicInformation(). Calling
    resetBasePositionAndOrientation alone leaves those caches stale -- a trap
    that silently broke an earlier static sweep in this project, where every
    height reported the same relative position.
    """
    import pybullet as p
    quat = p.getQuaternionFromEuler(list(rpy))
    p.resetBasePositionAndOrientation(
        int(env.DRONE_IDS[0]), list(pos), quat, physicsClientId=env.CLIENT
    )
    p.resetBaseVelocity(
        int(env.DRONE_IDS[0]), [0, 0, 0], [0, 0, 0],
        physicsClientId=env.CLIENT
    )
    env._updateAndStoreKinematicInformation()


# ===========================================================================
# MODE 3 -- DESIGN SWEEP
# ===========================================================================

def mode_sweep(args):
    """Measured vs closed-form dropout height across marker sizes and FOVs.

    The point of this mode is to make explicit that "the marker leaves the
    frame at height h" is a consequence of the marker size chosen, not a
    discovered property of the system. If measurement and closed form agree,
    the dropout height is design, and any later claim about surviving visual
    dropout has to quote the marker size it was measured at.
    """
    print("=" * 74)
    print("  MODE: DESIGN SWEEP  (analytic geometry, no rendering)")
    print("=" * 74)
    print()
    print(f"  {'marker_m':>9} {'fov_deg':>8} {'offset_m':>9} "
          f"{'h_drop_measured':>16} {'h_drop_closed_form':>19} {'delta':>9}")
    print("  " + "-" * 72)

    rows = []
    for marker in args.sweep_markers:
        for fov in args.sweep_fovs:
            vk = dict(VISION_KWARGS)
            vk.update({"marker_size": marker, "camera_fov_deg": fov,
                       "camera_mode": "geometry",
                       "camera_res": tuple(args.res)})
            env = VisionLanderAviary(**BASE_ENV_KWARGS, **vk)
            env.reset(seed=0)
            pad = np.asarray(env._getLandingTargetPos(), dtype=float)

            try:
                for off in args.offsets:
                    h_meas = _measure_dropout(env, pad, float(off))
                    h_pred = env.predictedDropoutHeight(float(off))
                    never = not np.isfinite(h_meas)
                    rows.append({
                        "marker_size": marker, "fov_deg": fov,
                        "offset_m": off,
                        "never_visible_in_task": never,
                        "h_drop_measured_m": (
                            "" if never else round(h_meas, 4)),
                        "h_drop_closed_form_m": round(h_pred, 4),
                        "delta_m": "" if never else round(h_meas - h_pred, 4),
                        "blind_fraction_of_descent": (
                            1.0 if never else round(h_meas / 0.5, 3)),
                    })
                    if never:
                        print(f"  {marker:9.2f} {fov:8.1f} {off:9.2f} "
                              f"{'NEVER VISIBLE':>16} {h_pred:19.4f} "
                              f"{'-':>9}")
                    else:
                        print(f"  {marker:9.2f} {fov:8.1f} {off:9.2f} "
                              f"{h_meas:16.4f} {h_pred:19.4f} "
                              f"{h_meas - h_pred:9.4f}")
            finally:
                env.close()

    _write_csv(args, "sweep_dropout.csv", rows)
    print()
    print("  h_drop_closed_form = (marker_size/2 + offset) / tan(FOV/2)")
    print("  blind_fraction_of_descent uses the 0.5 m spawn-to-pad height.")
    return rows


def _measure_dropout(env, pad, offset, hi=0.60, lo=0.0, iters=40):
    """Bisect for the height at which all 4 marker corners stop being in frame.

    Visibility is monotone in height for a level drone, so bisection is exact
    to within 2^-iters of the bracket.
    """
    def visible(h):
        _teleport(env, [pad[0] + offset, pad[1], pad[2] + h])
        return env._geometricCameraInfo()["cam_fully_in_frame"]

    if not visible(hi):
        # Never visible anywhere in the bracket. For a 0.5 m spawn height
        # this means the configuration cannot see the marker AT ALL on this
        # task, which is a result, not a missing value.
        return float('inf')
    if visible(lo):
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (hi + lo)
        if visible(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (hi + lo)


# ===========================================================================
# MODE 4 -- EPISODE MEASUREMENT
# ===========================================================================

def mode_episodes(args):
    """Fly episodes and record per-step visibility.

    Reports the two quantities that actually matter for the later stages:

      blind fraction BY TIME   -- what fraction of control steps the marker
                                  is not fully visible. This is what an LSTM
                                  estimator would have to bridge.
      blind fraction BY HEIGHT -- h_drop / 0.5. Pure geometry.

    These are different numbers and the gap between them is the point: the
    drone spends disproportionate time at low altitude during the terminal
    approach, so a 25%-of-height blind zone can be a far larger fraction of
    the clock.
    """
    print("=" * 74)
    print("  MODE: EPISODE MEASUREMENT")
    print("=" * 74)

    policy, label = load_policy(args)
    print(f"  policy        : {label}")
    if not args.model:
        print("  WARNING       : no checkpoint supplied. These numbers")
        print("                  describe the scripted policy's trajectories,")
        print("                  NOT the trained policy's.")

    vk = dict(VISION_KWARGS)
    vk.update({
        "camera_mode": "render" if args.render else "geometry",
        "camera_res": tuple(args.res),
        "camera_fov_deg": args.fov,
        "marker_size": args.marker_size,
    })
    env_kwargs = dict(BASE_ENV_KWARGS)
    env_kwargs["platform_speed_range"] = args.platform_speed

    print(f"  platform speed: {args.platform_speed} m/s")
    print(f"  spawn radius  : {env_kwargs['spawn_radius_min']}"
          f"-{env_kwargs['spawn_radius_max']} m")
    print(f"  camera        : {tuple(args.res)} @ {args.fov} deg, nadir")
    print(f"  marker        : {args.marker_size} m")
    print(f"  episodes      : {args.episodes}")
    print()

    env = VisionLanderAviary(**env_kwargs, **vk)

    per_step = []
    per_episode = []
    causes = Counter()

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed_base + ep)
            steps = 0
            n_fully = 0
            n_detected = 0
            first_terminal_blind_h = float('nan')
            blind_run_start_h = float('nan')

            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1

                h = float(info["height_above_pad"])
                dxy = float(info["horizontal_error"])
                full = bool(info["cam_fully_in_frame"])

                if full:
                    n_fully += 1
                    blind_run_start_h = float('nan')
                else:
                    causes[info["cam_loss_cause"]] += 1
                    if np.isnan(blind_run_start_h):
                        blind_run_start_h = h

                if args.render and info.get("cam_aruco_detected"):
                    n_detected += 1

                per_step.append({
                    "episode": ep, "step": steps,
                    "height_above_pad": round(h, 5),
                    "horizontal_error": round(dxy, 5),
                    "cam_u": round(float(info["cam_u"]), 4),
                    "cam_v": round(float(info["cam_v"]), 4),
                    "corners_in_frame": info["cam_corners_in_frame"],
                    "fully_in_frame": full,
                    "centre_in_frame": info["cam_centre_in_frame"],
                    "px_per_cell": round(float(info["cam_px_per_cell"]), 2),
                    "loss_cause": info["cam_loss_cause"],
                    "aruco_detected": info.get("cam_aruco_detected", ""),
                    "aruco_pos_err": info.get("cam_aruco_pos_err", ""),
                })

                if terminated or truncated:
                    # Height at which the FINAL uninterrupted blind run began
                    # -- the number directly comparable to Strand A's
                    # 0.15-0.20 m dropout finding.
                    first_terminal_blind_h = blind_run_start_h
                    break

            per_episode.append({
                "episode": ep,
                "steps": steps,
                "success": bool(info.get("success", False)),
                "truncation_reason": info.get("truncation_reason"),
                "blind_fraction_time": round(1.0 - n_fully / steps, 4),
                "detect_fraction_time": (
                    round(n_detected / steps, 4) if args.render else ""
                ),
                "terminal_dropout_height_m": (
                    "" if np.isnan(first_terminal_blind_h)
                    else round(first_terminal_blind_h, 4)
                ),
                "plat_speed": round(float(info.get("plat_speed", 0.0)), 4),
            })
    finally:
        env.close()

    _report_episodes(env_kwargs, vk, per_episode, causes, args)
    _write_csv(args, "episode_steps.csv", per_step)
    _write_csv(args, "episode_summary.csv", per_episode)
    return per_episode


def _report_episodes(env_kwargs, vk, per_episode, causes, args):
    n = len(per_episode)
    succ = sum(1 for r in per_episode if r["success"])
    blind = np.array([r["blind_fraction_time"] for r in per_episode])
    drops = np.array([
        r["terminal_dropout_height_m"] for r in per_episode
        if r["terminal_dropout_height_m"] != ""
    ], dtype=float)

    half_fov = np.radians(vk["camera_fov_deg"]) / 2.0
    h_pred = (vk["marker_size"] / 2.0) / np.tan(half_fov)

    print("-" * 74)
    print("  RESULTS")
    print("-" * 74)
    print(f"  episodes                        : {n}")
    print(f"  landed successfully             : {succ}/{n} "
          f"({100.0 * succ / n:.1f}%)")
    print()
    print(f"  blind fraction BY TIME          : "
          f"{blind.mean() * 100:.1f}% +/- {blind.std() * 100:.1f}%")
    print(f"  blind fraction BY HEIGHT        : "
          f"{h_pred / 0.5 * 100:.1f}%   (closed form, centred)")
    print()
    if len(drops):
        print(f"  terminal dropout height         : "
              f"{drops.mean():.4f} +/- {drops.std():.4f} m  "
              f"(n={len(drops)})")
    print(f"  closed-form prediction          : {h_pred:.4f} m  "
          f"(= marker_size/2 / tan(FOV/2))")
    print()
    print("  why the marker was not fully in frame (step counts):")
    total_loss = sum(causes.values()) or 1
    for cause, cnt in causes.most_common():
        print(f"    {str(cause):12s}: {cnt:7d}  "
              f"({100.0 * cnt / total_loss:5.1f}% of blind steps)")
    print("-" * 74)


# ===========================================================================
# IO HELPERS
# ===========================================================================

def _write_csv(args, name, rows):
    if not rows:
        return
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def _write_json(args, name, obj):
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["equivalence", "frames", "sweep", "episodes"])
    ap.add_argument("--outdir", default="v1_out")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--model", type=str, default=None,
                    help="SB3 TD3 checkpoint; omit to use the scripted policy")
    ap.add_argument("--render", action="store_true",
                    help="rasterise and run cv2.aruco (slower)")
    ap.add_argument("--camera-mode", default="render",
                    choices=["off", "geometry", "render"],
                    help="camera mode used during the equivalence gate")
    ap.add_argument("--res", type=int, nargs=2, default=[128, 128])
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--marker-size", type=float, default=0.25)
    ap.add_argument("--platform-speed", type=float, default=0.25)
    ap.add_argument("--heights", type=float, nargs="+",
                    default=[0.50, 0.40, 0.30, 0.20, 0.15, 0.125, 0.10, 0.05])
    ap.add_argument("--offsets", type=float, nargs="+",
                    default=[0.0, 0.11, 0.30])
    ap.add_argument("--sweep-markers", type=float, nargs="+",
                    default=[0.15, 0.20, 0.25, 0.30, 0.40])
    ap.add_argument("--sweep-fovs", type=float, nargs="+",
                    default=[60.0, 90.0, 120.0])
    args = ap.parse_args()

    if args.mode == "equivalence":
        ok = mode_equivalence(args)
        sys.exit(0 if ok else 1)
    elif args.mode == "frames":
        mode_frames(args)
    elif args.mode == "sweep":
        mode_sweep(args)
    elif args.mode == "episodes":
        mode_episodes(args)


if __name__ == "__main__":
    main()
