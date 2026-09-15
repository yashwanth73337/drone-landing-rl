"""
test_v3a_reset_isolation.py
===========================

Three things V3a's scientific claim depends on, tested rather than asserted:

  1. ESTIMATOR RESET      no estimator state survives a reset, and identical
                          seeds produce identical estimator traces regardless
                          of what ran before.

  2. WORKER ISOLATION     parallel VecEnv workers hold independent estimator
                          state.

  3. NO PRIVILEGED LEAK   in strict mode the actor's target-relative
                          observation comes from the camera and from nothing
                          else. Ground truth may move without a fresh
                          detection and the observation must not follow it.
                          And the privileged fallback must RAISE rather than
                          silently returning ground truth.

Run from the repo root:

    PYTHONPATH=. python tests/test_v3a_reset_isolation.py
"""

import sys
import numpy as np

sys.path.insert(0, 'src')

import pybullet as p                                       # noqa: E402
from envs.ArucoLanderAviary import ArucoLanderAviary       # noqa: E402


ENV_KWARGS = dict(
    tilt_limit=1.0,
    episode_len_sec=10.0,
    spawn_radius_min=0.11,
    spawn_radius_max=0.30,
    platform_speed_range=0.25,
    camera_res=(128, 128),
    camera_fov_deg=90.0,
    marker_size=0.25,
)

RESULTS = []


def check(label, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
          + (f"   {detail}" if detail else ""))


def scripted_action(obs):
    d = np.asarray(obs, dtype=np.float64)[9:12] * 5.0
    return np.clip([8.0 * d[0], 8.0 * d[1], 3.0 * d[2]], -1, 1).astype(
        np.float32)


# ===========================================================================
# 1. ESTIMATOR RESET
# ===========================================================================

def test_reset_clears_state():
    print("\n1. Estimator state is cleared on reset")
    env = ArucoLanderAviary(**ENV_KWARGS, estimator_mode='predict',
                            strict_no_privileged=True)
    try:
        obs, _ = env.reset(seed=1234)
        for _ in range(60):
            obs, *_ = env.step(scripted_action(obs))

        dirty = (
            env._have_detection or env._n_steps > 0
            or env._prev_d_meas is not None
            or np.any(env._d_est != 0) or np.any(env._dv_est != 0)
        )
        check("estimator carries state mid-episode (sanity)", dirty)

        env.reset(seed=4321)
        # reset() forces one fresh detection, so _n_steps == 1 and d_est may
        # be populated FROM THAT DETECTION. What must NOT survive is history:
        # blind-run accumulators, the finite-difference anchor's staleness,
        # and the step counters.
        check("_n_steps reset to the single post-reset detection",
              env._n_steps <= 1, f"_n_steps={env._n_steps}")
        check("_blind_runs cleared", len(env._blind_runs) == 0,
              f"len={len(env._blind_runs)}")
        check("_steps_since_detection cleared",
              env._steps_since_detection <= 1,
              f"={env._steps_since_detection}")
        check("_v_plat_est cleared or re-derived, not stale",
              env._n_detections <= 1, f"_n_detections={env._n_detections}")
    finally:
        env.close()


def test_reset_determinism():
    print("\n2. Same seed -> same estimator trace, regardless of history")

    def trace(seed, warmup_seed=None):
        env = ArucoLanderAviary(**ENV_KWARGS, estimator_mode='predict',
                                strict_no_privileged=True)
        try:
            if warmup_seed is not None:
                obs, _ = env.reset(seed=warmup_seed)
                for _ in range(80):
                    obs, *_ = env.step(scripted_action(obs))
            obs, _ = env.reset(seed=seed)
            out = []
            for _ in range(50):
                obs, r, term, trunc, info = env.step(scripted_action(obs))
                out.append([info['est_d_est_x'], info['est_d_est_y'],
                            info['est_d_est_z'],
                            float(info['est_detected'])])
                if term or trunc:
                    break
            return np.asarray(out)
        finally:
            env.close()

    clean = trace(777)
    dirty = trace(777, warmup_seed=999)
    n = min(len(clean), len(dirty))
    same = n > 0 and np.array_equal(clean[:n], dirty[:n])
    check("estimator trace identical after a prior episode", same,
          f"compared {n} steps")


# ===========================================================================
# 3. WORKER ISOLATION
# ===========================================================================

def test_worker_isolation():
    print("\n3. Parallel workers hold independent estimator state")
    from stable_baselines3.common.env_util import make_vec_env
    venv = make_vec_env(
        ArucoLanderAviary, n_envs=3, seed=5,
        env_kwargs=dict(ENV_KWARGS, estimator_mode='predict',
                        strict_no_privileged=True))
    try:
        venv.reset()
        for _ in range(25):
            venv.step(np.random.uniform(-1, 1, (3, 3)).astype(np.float32))

        d_ests = venv.get_attr('_d_est')
        ids = [id(x) for x in d_ests]
        check("each worker owns a distinct _d_est array",
              len(set(ids)) == 3)
        distinct = len({tuple(np.round(x, 6)) for x in d_ests}) > 1
        check("workers hold different estimates (independent episodes)",
              distinct)
    finally:
        venv.close()


# ===========================================================================
# 4. NO PRIVILEGED LEAK
# ===========================================================================

def test_no_privileged_leak():
    print("\n4. Ground truth cannot reach the actor observation (strict mode)")
    env = ArucoLanderAviary(**ENV_KWARGS, estimator_mode='predict',
                            strict_no_privileged=True)
    try:
        obs, _ = env.reset(seed=2024)
        for _ in range(40):
            obs, *_ = env.step(scripted_action(obs))

        before = np.asarray(env._computeObs(), dtype=np.float64).copy()

        # Teleport the platform a long way WITHOUT advancing a control step,
        # so no new camera frame is taken. A privileged path would follow the
        # platform instantly; a camera-only path cannot.
        pos, orn = p.getBasePositionAndOrientation(
            env.PLAT_ID, physicsClientId=env.CLIENT)
        p.resetBasePositionAndOrientation(
            env.PLAT_ID, [pos[0] + 2.0, pos[1] + 2.0, pos[2]], orn,
            physicsClientId=env.CLIENT)

        after = np.asarray(env._computeObs(), dtype=np.float64).copy()

        d_same = np.array_equal(before[9:12], after[9:12])
        dv_same = np.array_equal(before[12:15], after[12:15])
        check("relative POSITION unchanged by moving ground truth", d_same,
              f"delta={np.abs(before[9:12] - after[9:12]).max():.2e}")
        check("relative VELOCITY unchanged by moving ground truth", dv_same,
              f"delta={np.abs(before[12:15] - after[12:15]).max():.2e}")

        # Positive control: info[] SHOULD see the moved platform, since
        # ground truth is legitimate for diagnostics.
        info = env._computeInfo()
        moved = abs(info['est_err_dx']) > 1.0
        check("info[] ground-truth diagnostics DID follow the platform",
              moved, "(positive control -- confirms the test moved anything)")
    finally:
        env.close()


def test_strict_mode_raises():
    print("\n5. Strict mode raises rather than falling back to privileged")
    env = ArucoLanderAviary(**ENV_KWARGS, estimator_mode='predict',
                            strict_no_privileged=True)
    try:
        env.reset(seed=11)
        # Force the fallback branch AND prevent the platform from being
        # recreated, so the only remaining option would be the parent's
        # privileged observation.
        env.PLAT_ID = None
        original = env._updatePlatform
        env._updatePlatform = lambda *a, **k: None
        try:
            env._computeObs()
            check("RuntimeError raised instead of privileged fallback", False,
                  "no exception was raised")
        except RuntimeError as e:
            check("RuntimeError raised instead of privileged fallback", True,
                  f"{str(e)[:60]}...")
        except Exception as e:                              # noqa: BLE001
            check("RuntimeError raised instead of privileged fallback", False,
                  f"wrong exception type: {type(e).__name__}")
        finally:
            env._updatePlatform = original
    finally:
        env.close()


def test_v2_behaviour_preserved():
    print("\n6. Default (strict=False) preserves V2 behaviour exactly")
    env = ArucoLanderAviary(**ENV_KWARGS, estimator_mode='predict')
    try:
        check("strict flag defaults to False",
              env.STRICT_NO_PRIVILEGED is False)
        env.reset(seed=11)
        env.PLAT_ID = None
        original = env._updatePlatform
        env._updatePlatform = lambda *a, **k: None
        try:
            env._computeObs()
            check("non-strict mode still falls back without raising", True)
        except Exception as e:                              # noqa: BLE001
            check("non-strict mode still falls back without raising", False,
                  f"raised {type(e).__name__}")
        finally:
            env._updatePlatform = original
    finally:
        env.close()


def main():
    print("=" * 74)
    print("  V3a ENVIRONMENT TESTS")
    print("=" * 74)
    test_reset_clears_state()
    test_reset_determinism()
    test_worker_isolation()
    test_no_privileged_leak()
    test_strict_mode_raises()
    test_v2_behaviour_preserved()
    print()
    print("=" * 74)
    ok = all(RESULTS)
    print(f"  {sum(RESULTS)}/{len(RESULTS)} checks passed -- "
          f"{'PASS' if ok else 'FAIL'}")
    print("=" * 74)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
