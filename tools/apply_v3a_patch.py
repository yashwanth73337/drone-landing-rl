"""
apply_v3a_patch.py
==================

Applies the ONLY approved modification to the validated V2 file
src/envs/ArucoLanderAviary.py:

  1. a constructor flag  strict_no_privileged=False
  2. a cumulative render counter  _total_renders  (diagnostics only)

Both are ADDITIVE. With strict_no_privileged=False (the default) the file
behaves exactly as it did for V2, so V1 and V2 remain reproducible.

Run from the repo root:

    python tools/apply_v3a_patch.py

Idempotent: running it twice is safe and reports "already applied".
Writes a backup to ArucoLanderAviary.py.pre_v3a before touching anything.
"""

import os
import shutil
import sys

TARGET = os.path.join('src', 'envs', 'ArucoLanderAviary.py')


EDITS = [
    # ---- 1. constructor signature -------------------------------------
    (
        "    def __init__(self,\n"
        "                 *args,\n"
        "                 estimator_mode: str = 'predict',\n"
        "                 **kwargs):\n",

        "    def __init__(self,\n"
        "                 *args,\n"
        "                 estimator_mode: str = 'predict',\n"
        "                 strict_no_privileged: bool = False,\n"
        "                 **kwargs):\n",
    ),

    # ---- 2. store the flag + render counter ---------------------------
    (
        "        self.ESTIMATOR_MODE = estimator_mode\n",

        "        self.ESTIMATOR_MODE = estimator_mode\n"
        "\n"
        "        # V3a ADDITION (additive; default False preserves V2 exactly).\n"
        "        # When True, _computeObs() may NEVER return the parent's\n"
        "        # privileged observation vector. The scientific claim of V3a --\n"
        "        # 'no privileged target-relative state can reach the actor' --\n"
        "        # depends on this being enforced rather than being a property of\n"
        "        # call ordering several frames up the stack.\n"
        "        self.STRICT_NO_PRIVILEGED = bool(strict_no_privileged)\n"
        "\n"
        "        # Cumulative count of actual camera rasterisations, NOT reset\n"
        "        # between episodes. Used by bench_v3a.py to assert that exactly\n"
        "        # one render occurred per control step.\n"
        "        self._total_renders = 0\n",
    ),

    # ---- 3. count renders ---------------------------------------------
    (
        "        ok, d_meas = self._detectRelativePosition()\n"
        "        self._updateEstimator(ok, d_meas)\n",

        "        self._total_renders += 1\n"
        "        ok, d_meas = self._detectRelativePosition()\n"
        "        self._updateEstimator(ok, d_meas)\n",
    ),

    # ---- 4. the strict guard ------------------------------------------
    (
        "    def _computeObs(self):\n"
        "        # During super().reset() the platform may not exist yet; defer to the\n"
        "        # parent, which handles that case and returns the privileged vector.\n"
        "        if self.PLAT_ID is None:\n"
        "            return super()._computeObs()\n",

        "    def _computeObs(self):\n"
        "        # During super().reset() the platform may not exist yet.\n"
        "        #\n"
        "        # V2 behaviour (strict_no_privileged=False): defer to the parent,\n"
        "        # which creates the platform and returns the privileged vector.\n"
        "        # That value is discarded by this class's reset() override, which\n"
        "        # recomputes afterwards -- so it never actually reaches the policy.\n"
        "        # But 'does not leak because of call ordering three frames up' is\n"
        "        # not a property that should be load-bearing during training.\n"
        "        #\n"
        "        # V3a behaviour (strict_no_privileged=True): materialise the\n"
        "        # platform so the camera has a scene, then fall through to the\n"
        "        # ESTIMATED path. The privileged vector is never returned at all.\n"
        "        # Raising outright here was considered and rejected: this branch is\n"
        "        # genuinely reached during reset, before the platform body exists,\n"
        "        # so an unconditional raise would break every episode reset. The\n"
        "        # raise below fires only if the privileged fallback would actually\n"
        "        # have been reached.\n"
        "        if self.PLAT_ID is None:\n"
        "            if self.STRICT_NO_PRIVILEGED and self.ESTIMATOR_MODE != 'off':\n"
        "                self._updatePlatform(step_offset=0)\n"
        "                if self.PLAT_ID is None:\n"
        "                    raise RuntimeError(\n"
        "                        'ArucoLanderAviary: strict_no_privileged=True but the '\n"
        "                        'platform could not be created, so _computeObs() would '\n"
        "                        'have had to fall back to the parent PRIVILEGED '\n"
        "                        'observation. Refusing to do so -- a V3a run must not '\n"
        "                        'feed the actor ground-truth target state.'\n"
        "                    )\n"
        "            else:\n"
        "                return super()._computeObs()\n",
    ),
]


def main():
    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found. Run this from the repo root.")
        sys.exit(1)

    src = open(TARGET).read()

    if 'STRICT_NO_PRIVILEGED' in src:
        print("Patch already applied -- nothing to do.")
        sys.exit(0)

    missing = [i for i, (old, _) in enumerate(EDITS) if old not in src]
    if missing:
        print("ERROR: the file on disk does not match what this patch expects.")
        print(f"       Could not locate edit site(s): {missing}")
        print("       The file has changed since V2. Stopping rather than")
        print("       guessing -- send me the current file.")
        sys.exit(1)

    backup = TARGET + '.pre_v3a'
    shutil.copy2(TARGET, backup)

    for old, new in EDITS:
        src = src.replace(old, new, 1)

    open(TARGET, 'w').write(src)

    print(f"Patched {TARGET}")
    print(f"Backup written to {backup}")
    print()
    print("Added:")
    print("  - strict_no_privileged=False constructor flag (default = V2 behaviour)")
    print("  - _total_renders cumulative counter (diagnostics only)")
    print("  - strict guard in _computeObs()")
    print()
    print("Verify V2 is unchanged:")
    print("  python eval_aruco_v2.py --mode selfcheck \\")
    print("      --model results_lander/moving_025_seed1_final.zip --episodes 3")


if __name__ == '__main__':
    main()
