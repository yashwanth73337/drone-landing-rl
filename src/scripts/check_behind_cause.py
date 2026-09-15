"""
check_behind_cause.py
======================

Answers one specific, previously-unverified question from the V1 report:

    Is the 'behind' loss-cause (camera closer than the near plane) confined
    to the last step or two before touchdown on SUCCESSFUL episodes -- i.e.
    "the drone is now sitting on the pad, of course the camera is degenerate"
    -- or does it happen elsewhere, which would indicate something wrong?

Reads the CSVs fov_diagnostic_v1.py already wrote (episode_steps.csv +
episode_summary.csv) and does not touch the simulator, the environment, or
any checkpoint. Safe to run as many times as you like on existing output.

USAGE
-----
    python check_behind_cause.py --outdir v1_out
    python check_behind_cause.py --outdir v1_out_seed2
    python check_behind_cause.py --outdir v1_out_seed3

    # or all three at once, if run from src/scripts/:
    python check_behind_cause.py --outdir v1_out v1_out_seed2 v1_out_seed3

WHAT "CONFIRMED HARMLESS" MEANS HERE
-------------------------------------
Two conditions, both required:

  1. Every 'behind' step in a SUCCESSFUL episode occurs within the last
     N steps before that episode's termination (default N=3, i.e. within
     0.1 s at 30 Hz control rate -- generous, since contact typically
     resolves in 1-2 steps given SUCCESS_HOLD_STEPS=2).

  2. 'behind' does NOT occur in FAILED episodes in any meaningful volume,
     since a failed episode has no touchdown to be "sitting on".

If either fails, the hypothesis in the V1 report is wrong and the report
needs correcting before being called final.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict


def load_csv(path):
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_one(outdir, near_touchdown_window=3):
    print("=" * 78)
    print(f"  BEHIND-CAUSE CHECK: {outdir}")
    print("=" * 78)

    steps_path = os.path.join(outdir, "episode_steps.csv")
    summary_path = os.path.join(outdir, "episode_summary.csv")

    steps = load_csv(steps_path)
    summary = load_csv(summary_path)
    if steps is None or summary is None:
        print("  Cannot check -- missing file(s). Run --mode episodes first.")
        return None

    # episode -> final step count (from the summary, which is the ground
    # truth for how long that episode actually ran) and success flag.
    ep_info = {}
    for row in summary:
        ep_info[row["episode"]] = {
            "steps": int(row["steps"]),
            "success": row["success"] == "True",
        }

    # Group per-step rows by episode.
    by_ep = defaultdict(list)
    for row in steps:
        by_ep[row["episode"]].append(row)

    behind_success_offsets = []   # steps-from-end, successful episodes only
    behind_success_heights = []
    behind_fail_count = 0
    behind_fail_examples = []
    total_behind = 0

    for ep, rows in by_ep.items():
        info = ep_info.get(ep)
        if info is None:
            continue
        final_step = info["steps"]

        for r in rows:
            if r["loss_cause"] != "behind":
                continue
            total_behind += 1
            step_idx = int(r["step"])
            offset_from_end = final_step - step_idx   # 0 = last step

            if info["success"]:
                behind_success_offsets.append(offset_from_end)
                behind_success_heights.append(float(r["height_above_pad"]))
            else:
                behind_fail_count += 1
                if len(behind_fail_examples) < 5:
                    behind_fail_examples.append(
                        (ep, step_idx, final_step, r["height_above_pad"])
                    )

    n_success_eps = sum(1 for v in ep_info.values() if v["success"])
    n_fail_eps = len(ep_info) - n_success_eps

    print(f"  episodes                 : {len(ep_info)} "
          f"({n_success_eps} successful, {n_fail_eps} failed)")
    print(f"  total 'behind' steps      : {total_behind}")
    print(f"    in successful episodes  : {len(behind_success_offsets)}")
    print(f"    in FAILED episodes      : {behind_fail_count}")
    print()

    # ---- Condition 1: confined to the tail end of successful episodes ----
    if behind_success_offsets:
        max_offset = max(behind_success_offsets)
        within_window = sum(
            1 for o in behind_success_offsets if o < near_touchdown_window
        )
        frac_within = within_window / len(behind_success_offsets)

        print(f"  --- Condition 1: tail-end concentration (successful eps) ---")
        print(f"  steps-from-end range     : 0 (last step) .. {max_offset}")
        print(f"  within last "
              f"{near_touchdown_window} steps  : {within_window}/"
              f"{len(behind_success_offsets)} "
              f"({frac_within * 100:.1f}%)")
        if behind_success_heights:
            print(f"  height range at those steps: "
                  f"{min(behind_success_heights):.4f} .. "
                  f"{max(behind_success_heights):.4f} m")
        cond1_pass = (max_offset < near_touchdown_window)
    else:
        print("  --- Condition 1: no 'behind' steps in successful episodes ---")
        cond1_pass = True   # vacuously fine -- nothing to worry about

    print()

    # ---- Condition 2: absent (or negligible) in failed episodes ----
    print(f"  --- Condition 2: absence in failed episodes ---")
    if behind_fail_count == 0:
        print("  none found -- clean.")
        cond2_pass = True
    else:
        print(f"  {behind_fail_count} 'behind' steps found in FAILED episodes:")
        for ep, step_idx, final_step, h in behind_fail_examples:
            print(f"    episode {ep}: step {step_idx}/{final_step}, "
                  f"height_above_pad={h}")
        # A handful of edge-case steps (e.g. right at a crash into the pad)
        # is a different story from a systematic mid-flight problem --- but
        # ANY occurrence here contradicts the "sitting on the pad after
        # touchdown" story, since failed episodes never touch down.
        cond2_pass = False

    print()
    verdict = cond1_pass and cond2_pass
    print(f"  VERDICT: {'CONFIRMED HARMLESS' if verdict else 'NEEDS ATTENTION'}")
    print("=" * 78)
    print()

    return {
        "outdir": outdir,
        "total_behind": total_behind,
        "behind_in_success": len(behind_success_offsets),
        "behind_in_fail": behind_fail_count,
        "max_offset_from_end_success": (
            max(behind_success_offsets) if behind_success_offsets else None
        ),
        "cond1_tail_confined": cond1_pass,
        "cond2_absent_in_failures": cond2_pass,
        "verdict": "CONFIRMED HARMLESS" if verdict else "NEEDS ATTENTION",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", nargs="+", required=True,
                    help="one or more v1_out-style directories to check")
    ap.add_argument("--window", type=int, default=3,
                    help="how many steps before termination counts as "
                         "'the tail end' (default 3)")
    args = ap.parse_args()

    results = []
    for d in args.outdir:
        r = check_one(d, near_touchdown_window=args.window)
        if r:
            results.append(r)

    if len(results) > 1:
        print("=" * 78)
        print("  SUMMARY ACROSS ALL CHECKED DIRECTORIES")
        print("=" * 78)
        for r in results:
            print(f"  {r['outdir']:20s} : {r['verdict']}  "
                  f"(behind: {r['behind_in_success']} in successes, "
                  f"{r['behind_in_fail']} in failures, "
                  f"max offset from end = {r['max_offset_from_end_success']})")
        print("=" * 78)

    all_pass = all(r["verdict"] == "CONFIRMED HARMLESS" for r in results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
