"""
apply_pnp_fix.py
================

Fixes the root cause of the V3a crash, found by DiagnosticArucoAviary:

    [reject] worker 3: non-finite solvePnP output
             rvec=[nan nan nan]  tvec=[0.037 -0.139 0.432]

cv2.solvePnP with SOLVEPNP_IPPE_SQUARE can return ok=True while producing a
NaN rotation. IPPE is an analytic planar solver; when the four marker corners
are close to collinear in the image -- which is exactly what an untrained
policy produces as it thrashes near the pad at extreme viewing angles -- the
rotation recovery goes singular. Translation often survives, so the failure
does not look like a failure.

ArucoLanderAviary discarded rvec and used only tvec, so the NaN was invisible.
On the fraction of degenerate solves where tvec is ALSO non-finite, the NaN
propagated:

    tvec -> d_meas -> _d_est -> observation -> actor -> action -> next_pos
         -> DSLPIDControl -> Rotation.from_matrix -> "SVD did not converge"

np.clip does not stop this: np.clip(np.nan, -1, 1) is still np.nan.

THE FIX
-------
Validate rvec, tvec and the derived d_meas. If any is non-finite, REJECT the
detection and return (False, None) -- the same path an undetected marker
takes. This is not masking: a degenerate solve is a genuine perception
failure, and the estimator already has correct, documented behaviour for a
blind step (constant-velocity prediction). A bad measurement is discarded
rather than believed.

Rejections are COUNTED, exposed in info[] and in episodeEstimatorSummary(),
so their frequency is reported rather than hidden. If degenerate solves turn
out to be a meaningful fraction of detections, that is a perception property
of this setup and belongs in the V3a report.

APPLIED UNCONDITIONALLY, NOT BEHIND A FLAG
------------------------------------------
This is a bug fix, not an experimental parameter. Gating it would leave a
known NaN path live in the default configuration.

V2 is very unlikely to be affected -- its frozen policy approached the pad
cleanly and rarely produced degenerate geometry -- but that is an assumption,
not a measurement. After applying this patch, re-run the V2 selfcheck and
confirm the numbers are unchanged:

    python eval_aruco_v2.py --mode selfcheck \\
        --model results_lander/moving_025_seed1_final.zip --episodes 10

Expected: position error norm ~0.0129 m, dz mean ~-0.0084, and the same
"transform chain looks correct" verdict. If those move, the patch changed V2
and that must be reported rather than absorbed.

Run from the repo root:

    python tools/apply_pnp_fix.py
"""

import os
import shutil
import sys

TARGET = os.path.join('src', 'envs', 'ArucoLanderAviary.py')


EDITS = [
    # ---- 1. cumulative rejection counter ------------------------------
    (
        "        self._total_renders = 0\n",

        "        self._total_renders = 0\n"
        "\n"
        "        # Cumulative count of detections rejected because solvePnP\n"
        "        # returned a non-finite pose. NOT reset between episodes.\n"
        "        self._total_pnp_rejected = 0\n",
    ),

    # ---- 2. per-episode rejection counter -----------------------------
    (
        "        self._n_steps = 0\n"
        "        self._n_detections = 0\n",

        "        self._n_steps = 0\n"
        "        self._n_detections = 0\n"
        "        self._n_pnp_rejected = 0     # degenerate solves this episode\n",
    ),

    # ---- 3. the actual fix --------------------------------------------
    (
        "        ok, rvec, tvec = cv2.solvePnP(\n"
        "            self._markerCornersObject(), img_pts,\n"
        "            self.CAM_K, self.CAM_DIST,\n"
        "            flags=cv2.SOLVEPNP_IPPE_SQUARE,\n"
        "        )\n"
        "        if not ok:\n"
        "            return False, None\n"
        "\n"
        "        state = self._getDroneStateVector(0)\n",

        "        ok, rvec, tvec = cv2.solvePnP(\n"
        "            self._markerCornersObject(), img_pts,\n"
        "            self.CAM_K, self.CAM_DIST,\n"
        "            flags=cv2.SOLVEPNP_IPPE_SQUARE,\n"
        "        )\n"
        "        if not ok:\n"
        "            return False, None\n"
        "\n"
        "        # DEGENERATE-SOLVE REJECTION.\n"
        "        #\n"
        "        # SOLVEPNP_IPPE_SQUARE can return ok=True with a NaN rvec: it\n"
        "        # is an analytic planar solver, and near-collinear image\n"
        "        # corners make the rotation recovery singular while the\n"
        "        # translation still solves. Observed directly during V3a\n"
        "        # training at an extreme viewing angle:\n"
        "        #     rvec=[nan nan nan]  tvec=[0.037 -0.139 0.432]\n"
        "        #\n"
        "        # rvec is not used downstream, but its being NaN means the\n"
        "        # solve was already degenerate, and on some of those calls\n"
        "        # tvec is non-finite too. That NaN then reaches the actor and\n"
        "        # ultimately DSLPIDControl, where Rotation.from_matrix raises\n"
        "        # 'SVD did not converge'. np.clip does NOT stop it:\n"
        "        # np.clip(np.nan, -1, 1) is still np.nan.\n"
        "        #\n"
        "        # Rejecting is the correct semantics, not a workaround: a\n"
        "        # degenerate solve is a perception failure, and a blind step\n"
        "        # already has well-defined handling in _updateEstimator().\n"
        "        if not (np.all(np.isfinite(np.asarray(rvec, dtype=float)))\n"
        "                and np.all(np.isfinite(np.asarray(tvec, dtype=float)))):\n"
        "            self._n_pnp_rejected += 1\n"
        "            self._total_pnp_rejected += 1\n"
        "            return False, None\n"
        "\n"
        "        state = self._getDroneStateVector(0)\n",
    ),

    # ---- 4. validate the derived quantity too -------------------------
    (
        "        d_meas = (\n"
        "            R_body @ self.CAM_OFFSET_BODY           # camera mount offset\n"
        "            + R_wc @ np.asarray(tvec).reshape(3)    # camera -> world\n"
        "            - np.array([0.0, 0.0, self.MARKER_PLANE_OFFSET])  # marker -> pad\n"
        "        )\n"
        "        return True, d_meas\n",

        "        d_meas = (\n"
        "            R_body @ self.CAM_OFFSET_BODY           # camera mount offset\n"
        "            + R_wc @ np.asarray(tvec).reshape(3)    # camera -> world\n"
        "            - np.array([0.0, 0.0, self.MARKER_PLANE_OFFSET])  # marker -> pad\n"
        "        )\n"
        "\n"
        "        # Belt and braces: the drone's own attitude feeds R_body and\n"
        "        # R_wc, so a non-finite physics state would also poison d_meas\n"
        "        # even with a clean solvePnP result. Same treatment.\n"
        "        if not np.all(np.isfinite(d_meas)):\n"
        "            self._n_pnp_rejected += 1\n"
        "            self._total_pnp_rejected += 1\n"
        "            return False, None\n"
        "\n"
        "        return True, d_meas\n",
    ),

    # ---- 5. expose in info[] ------------------------------------------
    (
        "            'est_detection_rate': (\n"
        "                self._n_detections / self._n_steps if self._n_steps else 0.0\n"
        "            ),\n",

        "            'est_detection_rate': (\n"
        "                self._n_detections / self._n_steps if self._n_steps else 0.0\n"
        "            ),\n"
        "            'est_pnp_rejected': int(self._n_pnp_rejected),\n",
    ),

    # ---- 6. expose in the episode summary -----------------------------
    (
        "            'steps_before_first_detection':\n"
        "                self._steps_before_first_detection,\n"
        "            'ever_detected': self._have_detection,\n",

        "            'steps_before_first_detection':\n"
        "                self._steps_before_first_detection,\n"
        "            'pnp_rejected': int(self._n_pnp_rejected),\n"
        "            'pnp_rejection_rate': (\n"
        "                self._n_pnp_rejected / self._n_steps\n"
        "                if self._n_steps else 0.0\n"
        "            ),\n"
        "            'ever_detected': self._have_detection,\n",
    ),
]


def main():
    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found. Run this from the repo root.")
        sys.exit(1)

    src = open(TARGET).read()

    if '_total_pnp_rejected' in src:
        print("Patch already applied -- nothing to do.")
        sys.exit(0)

    if 'STRICT_NO_PRIVILEGED' not in src:
        print("ERROR: apply_v3a_patch.py has not been applied yet.")
        print("       Run that first -- this patch builds on it.")
        sys.exit(1)

    missing = [i for i, (old, _) in enumerate(EDITS) if old not in src]
    if missing:
        print("ERROR: the file on disk does not match what this patch expects.")
        print(f"       Could not locate edit site(s): {missing}")
        print("       Stopping rather than guessing -- send me the file.")
        sys.exit(1)

    backup = TARGET + '.pre_pnpfix'
    shutil.copy2(TARGET, backup)

    for old, new in EDITS:
        src = src.replace(old, new, 1)

    open(TARGET, 'w').write(src)

    print(f"Patched {TARGET}")
    print(f"Backup written to {backup}")
    print()
    print("Added:")
    print("  - rejection of non-finite solvePnP output (rvec or tvec)")
    print("  - rejection of non-finite derived d_meas")
    print("  - per-episode and cumulative rejection counters")
    print("  - est_pnp_rejected in info[], pnp_rejected + rate in the")
    print("    episode summary")
    print()
    print("NEXT -- confirm V2 is unchanged before trusting anything:")
    print("  cd src/scripts")
    print("  python eval_aruco_v2.py --mode selfcheck \\")
    print("      --model results_lander/moving_025_seed1_final.zip \\")
    print("      --episodes 10")
    print()
    print("  Expected: pos error norm ~0.0129 m, dz mean ~-0.0084,")
    print("  'transform chain looks correct'. If those moved, the patch")
    print("  changed V2 and that has to be reported, not absorbed.")


if __name__ == '__main__':
    main()
