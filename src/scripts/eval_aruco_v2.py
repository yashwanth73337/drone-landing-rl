"""
eval_aruco_v2.py
================

Stage V2 evaluation. Trains nothing, saves no model, and does not modify
LanderAIAviary.py, VisionLanderAviary.py, or any checkpoint.

TWO MODES, AND THE ORDER MATTERS
--------------------------------

    --mode selfcheck    RUN THIS FIRST. Flies on the PRIVILEGED observation
                        (estimator_mode='off') so the trajectory is known-
                        good, while the ArUco estimator runs alongside and
                        is compared against ground truth. Reports AXIS-WISE
                        SIGNED errors, which is what actually catches a
                        camera/body/world frame swap or a marker datum
                        offset. A norm would hide both.

    --mode evaluate     The real comparison. Runs three conditions on
                        IDENTICAL seeds with the same frozen checkpoint:
                          privileged   ground-truth observation (baseline)
                          hold         ArUco + zero-order hold
                          predict      ArUco + constant-velocity prediction

Do not run --mode evaluate until selfcheck is clean. An undetected frame
error would produce a plausible-looking success-rate drop that has nothing
to do with the research question.


HOW TO READ THE SELFCHECK
-------------------------
The estimator is only meaningful on steps where a detection actually
occurred, so the headline table is restricted to those. What to look for:

    mean signed error near zero on all three position axes
        -> transform chain is right

    a constant non-zero offset on ONE axis
        -> frame swap or datum error. The usual suspects are
           MARKER_PLANE_OFFSET (dz) and CAM_OFFSET_BODY (dz), or an x/y
           transpose in the camera basis (dx and dy swapped in sign or
           magnitude).

    dz biased by about +0.001 m
        -> MARKER_PLANE_OFFSET not subtracted

    dz biased by about +0.01 m
        -> CAM_OFFSET_BODY not applied

Velocity errors are expected to be noisier than position: delta_v is a
finite difference of a noisy quantity over a single 1/30 s control step, so
detection noise is amplified by the control rate.


USAGE
-----
    python eval_aruco_v2.py --mode selfcheck \
        --model results_lander/moving_025_seed1_final.zip --episodes 10

    python eval_aruco_v2.py --mode evaluate \
        --model results_lander/moving_025_seed1_final.zip \
        --episodes 50 --outdir v2_out_seed1
"""

import os
import sys
import csv
import json
import argparse
from collections import Counter

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.ArucoLanderAviary import ArucoLanderAviary   # noqa: E402


# Matches train_lander.py's ENV_KWARGS plus the curriculum's final level
# (RadiusCurriculumCallback.LEVELS[-1]), i.e. the distribution the
# moving_025 checkpoints were trained and scored at.
BASE_ENV_KWARGS = {
    "tilt_limit": 1.0,
    "spawn_radius_min": 0.11,
    "spawn_radius_max": 0.30,
    "platform_speed_range": 0.25,
}

VISION_KWARGS = {
    "camera_res": (128, 128),
    "camera_fov_deg": 90.0,
    "marker_size": 0.25,
}

# A failed episode counts as a NEAR MISS when the drone spent at least this
# many consecutive control steps within NEAR_MISS_HEIGHT of the pad surface
# without the success criterion firing. This is the behaviour V1 found in
# the privileged baseline (drone parked ~1-2 cm above the pad for most of
# the episode), and it has to be measured at the same sample size as V2 or
# a V2 success-rate drop will be wrongly attributed to vision.
NEAR_MISS_HEIGHT = 0.03
NEAR_MISS_STEPS = 10


def load_model(path):
    from stable_baselines3 import TD3
    return TD3.load(path)


def make_env(estimator_mode):
    return ArucoLanderAviary(
        **BASE_ENV_KWARGS, **VISION_KWARGS, estimator_mode=estimator_mode
    )


# ===========================================================================
# MODE 1 -- SELFCHECK
# ===========================================================================

def mode_selfcheck(args):
    print("=" * 78)
    print("  MODE: SELFCHECK  (privileged flight, estimator logged only)")
    print("=" * 78)
    print(f"  model    : {os.path.basename(args.model)}")
    print(f"  episodes : {args.episodes}")
    print("  The policy flies on GROUND TRUTH here. The ArUco estimate is")
    print("  computed alongside and compared, nothing more.")
    print()

    model = load_model(args.model)
    env = make_env('off')

    rows = []
    first_det_steps = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed_base + ep)
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if info.get('est_detected'):
                    rows.append({
                        'episode': ep,
                        'height_above_pad': info['height_above_pad'],
                        'horizontal_error': info['horizontal_error'],
                        'dx': info['est_err_dx'],
                        'dy': info['est_err_dy'],
                        'dz': info['est_err_dz'],
                        'dvx': info['est_err_dvx'],
                        'dvy': info['est_err_dvy'],
                        'dvz': info['est_err_dvz'],
                        'pos_norm': info['est_err_pos_norm'],
                        'vel_norm': info['est_err_vel_norm'],
                    })
                if terminated or truncated:
                    first_det_steps.append(
                        info.get('est_steps_before_first_detection', -1)
                    )
                    break
    finally:
        env.close()

    if not rows:
        print("  NO DETECTIONS AT ALL across every episode.")
        print("  That is a hard failure, not a marginal result. Check that")
        print("  the marker is being created and the camera is rendering:")
        print("    python fov_diagnostic_v1.py --mode frames")
        return False

    print(f"  steps with a detection : {len(rows)}")
    print()
    print("  AXIS-WISE SIGNED ERROR  (estimate minus ground truth)")
    print("  " + "-" * 68)
    print(f"  {'component':>10} {'mean':>12} {'std':>12} "
          f"{'p50':>12} {'p95(|.|)':>12}")
    print("  " + "-" * 68)

    verdict_ok = True
    for key, unit, tol in (('dx', 'm', 0.02), ('dy', 'm', 0.02),
                           ('dz', 'm', 0.02),
                           ('dvx', 'm/s', 0.60), ('dvy', 'm/s', 0.60),
                           ('dvz', 'm/s', 0.60)):
        v = np.array([r[key] for r in rows], dtype=float)
        print(f"  {key:>10} {v.mean():12.5f} {v.std():12.5f} "
              f"{np.median(v):12.5f} "
              f"{np.percentile(np.abs(v), 95):12.5f}")
        if abs(v.mean()) > tol:
            verdict_ok = False

    pos = np.array([r['pos_norm'] for r in rows], dtype=float)
    vel = np.array([r['vel_norm'] for r in rows], dtype=float)
    print("  " + "-" * 68)
    print(f"  position error norm : mean {pos.mean():.5f} m   "
          f"p95 {np.percentile(pos, 95):.5f} m")
    print(f"  velocity error norm : mean {vel.mean():.5f} m/s "
          f"p95 {np.percentile(vel, 95):.5f} m/s")
    print()
    fd = np.array(first_det_steps, dtype=float)
    print(f"  steps before first detection : mean {fd.mean():.2f}  "
          f"max {int(fd.max())}")
    print()

    print("  " + "-" * 68)
    if verdict_ok:
        print("  VERDICT: transform chain looks correct.")
        print("  Mean signed errors are small on every axis, so there is no")
        print("  constant bias of the kind a frame swap or datum error")
        print("  produces. Safe to proceed to --mode evaluate.")
    else:
        print("  VERDICT: NEEDS ATTENTION.")
        print("  At least one axis carries a mean signed error large enough")
        print("  to indicate a systematic bias rather than detection noise.")
        print("  See the header of this file for which offset produces")
        print("  which signature. DO NOT run --mode evaluate yet.")
    print("=" * 78)

    _write_csv(args, 'selfcheck_steps.csv', rows)
    return verdict_ok


# ===========================================================================
# MODE 2 -- EVALUATE
# ===========================================================================

def mode_evaluate(args):
    print("=" * 78)
    print("  MODE: EVALUATE  (three conditions, identical seeds)")
    print("=" * 78)
    print(f"  model    : {os.path.basename(args.model)}")
    print(f"  episodes : {args.episodes}  (seeds "
          f"{args.seed_base}..{args.seed_base + args.episodes - 1})")
    print()

    model = load_model(args.model)
    conditions = [
        ('privileged', 'off'),
        ('hold', 'hold'),
        ('predict', 'predict'),
    ]

    all_results = {}
    per_episode_rows = []

    for label, mode in conditions:
        print(f"--- running condition: {label} " + "-" * (46 - len(label)))
        res, rows = _run_condition(model, mode, label, args)
        all_results[label] = res
        per_episode_rows.extend(rows)
        _print_condition(label, res)

    _print_comparison(all_results)
    _write_csv(args, 'v2_episodes.csv', per_episode_rows)
    _write_json(args, 'v2_summary.json', all_results)
    return all_results


def _run_condition(model, estimator_mode, label, args):
    env = make_env(estimator_mode)
    rows = []

    try:
        for ep in range(args.episodes):
            seed = args.seed_base + ep
            obs, info = env.reset(seed=seed)

            heights = []
            pos_err = []
            vel_err = []
            low_run = 0
            low_run_max = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                h = float(info['height_above_pad'])
                heights.append(h)
                if h < NEAR_MISS_HEIGHT:
                    low_run += 1
                    low_run_max = max(low_run_max, low_run)
                else:
                    low_run = 0

                if info.get('est_detected'):
                    pos_err.append(info['est_err_pos_norm'])
                    vel_err.append(info['est_err_vel_norm'])

                if terminated or truncated:
                    break

            summ = env.episodeEstimatorSummary()
            success = bool(info.get('success', False))
            near_miss = (not success) and (low_run_max >= NEAR_MISS_STEPS)

            rows.append({
                'condition': label,
                'episode': ep,
                'seed': seed,
                'success': success,
                'near_miss': near_miss,
                'truncation_reason': info.get('truncation_reason'),
                'final_d_target_cm': round(
                    float(info.get('d_target', float('nan'))) * 100.0, 4),
                'steps': summ['steps'],
                'blind_fraction': round(summ['blind_fraction'], 4),
                'steps_before_first_detection':
                    summ['steps_before_first_detection'],
                'ever_detected': summ['ever_detected'],
                'longest_blind_run': summ['longest_blind_run'],
                'mean_blind_run': round(summ['mean_blind_run'], 3),
                'pos_err_mean_m': (
                    round(float(np.mean(pos_err)), 5) if pos_err else ''),
                'vel_err_mean_ms': (
                    round(float(np.mean(vel_err)), 5) if vel_err else ''),
                'blind_runs': ';'.join(str(x) for x in summ['blind_runs']),
            })
    finally:
        env.close()

    return _aggregate(rows), rows


def _aggregate(rows):
    n = len(rows)
    succ = [r for r in rows if r['success']]
    prec = [r['final_d_target_cm'] for r in succ
            if np.isfinite(r['final_d_target_cm'])]
    bf = np.array([r['blind_fraction'] for r in rows], dtype=float)
    fd = np.array([r['steps_before_first_detection'] for r in rows],
                  dtype=float)
    longest = np.array([r['longest_blind_run'] for r in rows], dtype=float)

    pe = [r['pos_err_mean_m'] for r in rows if r['pos_err_mean_m'] != '']
    ve = [r['vel_err_mean_ms'] for r in rows if r['vel_err_mean_ms'] != '']

    reasons = Counter(
        r['truncation_reason'] or 'terminated_no_reason'
        for r in rows if not r['success']
    )

    return {
        'episodes': n,
        'success_rate': round(100.0 * len(succ) / n, 2) if n else 0.0,
        'near_miss_rate': round(
            100.0 * sum(1 for r in rows if r['near_miss']) / n, 2) if n else 0.0,
        'precision_mean_cm': round(float(np.mean(prec)), 3) if prec else None,
        'precision_std_cm': round(float(np.std(prec)), 3) if prec else None,
        'blind_fraction_mean': round(float(bf.mean()) * 100, 2),
        'blind_fraction_std': round(float(bf.std()) * 100, 2),
        'longest_blind_run_mean': round(float(longest.mean()), 2),
        'longest_blind_run_max': int(longest.max()) if n else 0,
        'steps_before_first_detection_mean': round(float(fd.mean()), 2),
        'steps_before_first_detection_max': int(fd.max()) if n else 0,
        'never_detected_episodes': sum(
            1 for r in rows if not r['ever_detected']),
        'pos_err_mean_m': round(float(np.mean(pe)), 5) if pe else None,
        'vel_err_mean_ms': round(float(np.mean(ve)), 5) if ve else None,
        'failure_reasons': dict(reasons),
    }


def _print_condition(label, r):
    print(f"  success            : {r['success_rate']}%")
    if r['precision_mean_cm'] is not None:
        print(f"  precision          : {r['precision_mean_cm']} +/- "
              f"{r['precision_std_cm']} cm")
    print(f"  near-miss rate     : {r['near_miss_rate']}%")
    print(f"  blind fraction     : {r['blind_fraction_mean']} +/- "
          f"{r['blind_fraction_std']}%")
    print(f"  longest blind run  : mean {r['longest_blind_run_mean']} steps, "
          f"max {r['longest_blind_run_max']}")
    print(f"  steps before 1st   : mean "
          f"{r['steps_before_first_detection_mean']}, "
          f"max {r['steps_before_first_detection_max']}")
    if r['never_detected_episodes']:
        print(f"  NEVER detected     : {r['never_detected_episodes']} episodes")
    if r['pos_err_mean_m'] is not None:
        print(f"  est error          : {r['pos_err_mean_m']} m pos, "
              f"{r['vel_err_mean_ms']} m/s vel")
    if r['failure_reasons']:
        print(f"  failures           : {r['failure_reasons']}")
    print()


def _print_comparison(res):
    print("=" * 78)
    print("  COMPARISON  (identical seeds, identical checkpoint)")
    print("=" * 78)
    print(f"  {'condition':<12} {'success':>9} {'precision':>14} "
          f"{'near-miss':>10} {'blind':>9}")
    print("  " + "-" * 62)
    for label in ('privileged', 'hold', 'predict'):
        r = res[label]
        prec = ('n/a' if r['precision_mean_cm'] is None
                else f"{r['precision_mean_cm']:.2f}+/-"
                     f"{r['precision_std_cm']:.2f}cm")
        print(f"  {label:<12} {r['success_rate']:>8.1f}% {prec:>14} "
              f"{r['near_miss_rate']:>9.1f}% "
              f"{r['blind_fraction_mean']:>8.1f}%")
    print("  " + "-" * 62)

    base = res['privileged']['success_rate']
    print()
    print("  Interpretation notes:")
    print(f"  - The privileged near-miss rate "
          f"({res['privileged']['near_miss_rate']}%) is a PRE-EXISTING")
    print("    property of the checkpoint, measured here at the same sample")
    print("    size. Subtract it before attributing any drop to vision.")
    for label in ('hold', 'predict'):
        drop = base - res[label]['success_rate']
        print(f"  - {label:<8}: {-drop:+.1f} pp vs privileged")
    print("  - 'predict' is a constant-velocity dead-reckoning model, NOT an")
    print("    EKF: no covariance propagation, no measurement update. The")
    print("    platform truly moves at constant velocity in this")
    print("    environment, which favours it; say so in any write-up.")
    print("=" * 78)


# ===========================================================================

def _write_csv(args, name, rows):
    if not rows:
        return
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, name)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def _write_json(args, name, obj):
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, name)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', required=True,
                    choices=['selfcheck', 'evaluate'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--seed-base', type=int, default=9000)
    ap.add_argument('--outdir', default='v2_out')
    args = ap.parse_args()

    if args.mode == 'selfcheck':
        ok = mode_selfcheck(args)
        sys.exit(0 if ok else 1)
    else:
        mode_evaluate(args)


if __name__ == '__main__':
    main()
