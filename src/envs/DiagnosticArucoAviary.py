"""
DiagnosticArucoAviary
=====================

Instrumentation-only subclass of ArucoLanderAviary, written to find the FIRST
non-finite value in the V3a pipeline rather than the last one.

The observed crash was:

    numpy.linalg.LinAlgError: SVD did not converge
      in Rotation.from_matrix(target_rotation)
      in DSLPIDControl._dslPIDPositionControl

That is the END of the chain, not the start. By the time SciPy fails, a NaN
or Inf has already travelled through the observation, the actor, the action,
and into the controller. This class asserts finiteness at every boundary so
the failure is reported where it originates.

NOTHING IS MASKED. No np.nan_to_num, no clipping of non-finite values, no
substitution of zeros. Every check raises, and dumps full context first.

    np.clip(np.nan, -1, 1) is still np.nan.

This file is DIAGNOSTIC ONLY. It is not used for the real V3a training run,
and it changes no science: no reward, curriculum, camera, estimator, TD3 or
action-limit changes. It exists to answer one question and then get out of
the way.


WHY A SUBCLASS RATHER THAN PATCHES
------------------------------------------------------------------------------
Requirements 4 and 5 name LanderAIAviary._preprocessAction() and
_computeReward(). Neither file needs editing: ArucoLanderAviary already
overrides _preprocessAction(), and _computeReward() can be wrapped the same
way. So LanderAIAviary.py, VisionLanderAviary.py and ArucoLanderAviary.py all
remain byte-identical and V1/V2 stay exactly reproducible.

One duplication is unavoidable and worth flagging: _detectRelativePosition()
is reimplemented here, because the parent computes rvec/tvec internally and
returns only d_meas, and requirement 2 needs rvec/tvec themselves. The copy
is marked below. If the parent's transform chain ever changes, this copy must
change with it -- it is diagnostic scaffolding, not a second implementation
to maintain long-term.
"""

import os
import json
import datetime

import numpy as np
import pybullet as p
import cv2

from envs.ArucoLanderAviary import ArucoLanderAviary


class NonFiniteError(RuntimeError):
    """Raised at the first boundary where a non-finite value is observed."""


class DiagnosticArucoAviary(ArucoLanderAviary):

    _instance_counter = 0

    def __init__(self, *args, dump_dir='nan_dumps', **kwargs):
        DiagnosticArucoAviary._instance_counter += 1
        self.WORKER_ID = DiagnosticArucoAviary._instance_counter
        self.DUMP_DIR = dump_dir
        self._dumped = False

        # Rolling history of the last known-good values, so the dump can show
        # what the pipeline looked like immediately BEFORE it went bad.
        self._last_good_obs = None
        self._last_good_action = None
        self._last_good_state = None
        self._last_good_detection_step = None
        self._last_rvec = None
        self._last_tvec = None
        self._episode_step = 0
        self._episode_index = 0

        super().__init__(*args, **kwargs)

    # ==================================================================
    # DUMP
    # ==================================================================

    def _snapshot(self):
        """Everything worth knowing at the moment of failure."""
        try:
            state = np.asarray(self._getDroneStateVector(0), dtype=float)
        except Exception as e:                              # noqa: BLE001
            state = f"<unavailable: {e}>"

        def arr(x):
            if x is None:
                return None
            a = np.asarray(x, dtype=float)
            return a.tolist()

        try:
            plat_pos = arr(self._getLandingTargetPos())
            plat_vel = arr(self._getPlatformVel())
        except Exception:                                   # noqa: BLE001
            plat_pos = plat_vel = None

        return {
            'worker_id': self.WORKER_ID,
            'episode_index': self._episode_index,
            'episode_step': self._episode_step,
            'step_counter': int(getattr(self, 'step_counter', -1)),
            'det_epoch': int(self._det_epoch),
            'det_cached_epoch': self._det_cached_epoch,

            'drone_state': arr(state) if not isinstance(state, str) else state,
            'drone_pos': arr(state[0:3]) if not isinstance(state, str) else None,
            'drone_quat': arr(state[3:7]) if not isinstance(state, str) else None,
            'drone_rpy': arr(state[7:10]) if not isinstance(state, str) else None,
            'drone_vel': arr(state[10:13]) if not isinstance(state, str) else None,
            'drone_angvel': arr(state[13:16]) if not isinstance(state, str) else None,

            'estimator_mode': self.ESTIMATOR_MODE,
            'd_est': arr(self._d_est),
            'dv_est': arr(self._dv_est),
            'v_plat_est': arr(self._v_plat_est),
            'prev_d_meas': arr(self._prev_d_meas),
            'have_detection': bool(self._have_detection),
            'last_detection_valid': bool(self._last_detection_valid),
            'steps_since_detection': int(self._steps_since_detection),
            'steps_before_first_detection':
                int(self._steps_before_first_detection),
            'blind_run': int(self._blind_run),
            'n_detections': int(self._n_detections),
            'n_steps': int(self._n_steps),
            'last_good_detection_step': self._last_good_detection_step,

            'last_rvec': arr(self._last_rvec),
            'last_tvec': arr(self._last_tvec),
            'det_cache': {
                k: (arr(v) if isinstance(v, np.ndarray) else v)
                for k, v in (self._det_cache or {}).items()
            },

            'ctrl_timestep': float(self.CTRL_TIMESTEP),
            'ctrl_freq': int(self.CTRL_FREQ),
            'pyb_freq': int(self.PYB_FREQ),

            'spawn_radius_min': float(self.SPAWN_RADIUS_MIN),
            'spawn_radius_max': float(self.SPAWN_RADIUS_MAX),
            'platform_speed_range': float(self.PLATFORM_SPEED_RANGE),
            'plat_vel': plat_vel,
            'plat_target_pos': plat_pos,
            'plat_pos0': arr(self.plat_pos0),

            'last_good_obs': arr(self._last_good_obs),
            'last_good_action': arr(self._last_good_action),
            'last_good_state': arr(self._last_good_state),
        }

    def _dump(self, reason, extra=None):
        """Write the first-failure context to disk and return the path."""
        os.makedirs(self.DUMP_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base = os.path.join(
            self.DUMP_DIR, f'nan_w{self.WORKER_ID}_{stamp}')

        payload = {'reason': reason, 'extra': extra or {},
                   **self._snapshot()}

        with open(base + '.json', 'w') as f:
            json.dump(payload, f, indent=2, default=str)

        npz = {}
        for k, v in payload.items():
            if isinstance(v, list):
                try:
                    npz[k] = np.asarray(v, dtype=float)
                except Exception:                           # noqa: BLE001
                    pass
        if npz:
            np.savez(base + '.npz', **npz)

        print('\n' + '=' * 74)
        print(f'  FIRST NON-FINITE VALUE DETECTED: {reason}')
        print('=' * 74)
        print(f'  worker {payload["worker_id"]}  '
              f'episode {payload["episode_index"]}  '
              f'episode_step {payload["episode_step"]}  '
              f'step_counter {payload["step_counter"]}')
        print(f'  estimator: have_detection={payload["have_detection"]}  '
              f'steps_since_detection={payload["steps_since_detection"]}  '
              f'blind_run={payload["blind_run"]}')
        print(f'  d_est      = {payload["d_est"]}')
        print(f'  dv_est     = {payload["dv_est"]}')
        print(f'  v_plat_est = {payload["v_plat_est"]}')
        print(f'  drone_pos  = {payload["drone_pos"]}')
        print(f'  drone_vel  = {payload["drone_vel"]}')
        print(f'  drone_quat = {payload["drone_quat"]}')
        if extra:
            for k, v in extra.items():
                print(f'  {k} = {v}')
        print(f'\n  dump written to {base}.json / .npz')
        print('=' * 74 + '\n')
        return base

    def _require_finite(self, name, value, reason, extra=None):
        a = np.asarray(value, dtype=float)
        if np.all(np.isfinite(a)):
            return
        ex = dict(extra or {})
        ex[f'offending::{name}'] = a.tolist()
        if not self._dumped:
            self._dumped = True
            self._dump(reason, ex)
        raise NonFiniteError(
            f'non-finite {name} ({reason}) at worker {self.WORKER_ID}, '
            f'episode {self._episode_index}, step {self._episode_step}: '
            f'{a.tolist()}')

    # ==================================================================
    # 2. DETECTION / solvePnP  -- COPY of the parent, with rvec/tvec
    #    captured and every intermediate checked.
    # ==================================================================

    def _detectRelativePosition(self):
        gray = self._renderCamera()
        self._require_finite('rendered_image', gray.astype(float),
                             'renderer produced a non-finite image')

        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        if ids is None or self.MARKER_ID not in ids.flatten():
            return False, None

        idx = int(np.where(ids.flatten() == self.MARKER_ID)[0][0])
        img_pts = corners[idx].reshape(4, 2).astype(np.float64)

        if not np.all(np.isfinite(img_pts)):
            print(f'  [reject] worker {self.WORKER_ID}: non-finite ArUco '
                  f'image points, detection discarded')
            return False, None

        ok, rvec, tvec = cv2.solvePnP(
            self._markerCornersObject(), img_pts,
            self.CAM_K, self.CAM_DIST,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return False, None

        self._last_rvec = np.asarray(rvec, dtype=float).reshape(3).copy()
        self._last_tvec = np.asarray(tvec, dtype=float).reshape(3).copy()

        # Requirement 2: reject, do not raise -- a degenerate PnP solve is a
        # legitimate perception outcome, not a pipeline fault. It is reported
        # explicitly so its frequency is visible.
        if not (np.all(np.isfinite(self._last_rvec))
                and np.all(np.isfinite(self._last_tvec))):
            print(f'  [reject] worker {self.WORKER_ID}: non-finite solvePnP '
                  f'output rvec={self._last_rvec} tvec={self._last_tvec}, '
                  f'detection discarded')
            return False, None

        state = self._getDroneStateVector(0)
        self._require_finite('drone_state@detect', state[0:16],
                             'drone state already non-finite at detection')

        R_body = np.array(
            p.getMatrixFromQuaternion(state[3:7]), dtype=float).reshape(3, 3)
        self._require_finite('R_body', R_body,
                             'body rotation matrix non-finite')

        _, R_wc = self._cameraPose()
        self._require_finite('R_wc', R_wc,
                             'camera rotation matrix non-finite')

        d_meas = (
            R_body @ self.CAM_OFFSET_BODY
            + R_wc @ self._last_tvec
            - np.array([0.0, 0.0, self.MARKER_PLANE_OFFSET])
        )

        if not np.all(np.isfinite(d_meas)):
            print(f'  [reject] worker {self.WORKER_ID}: non-finite d_meas '
                  f'{d_meas}, detection discarded')
            return False, None

        self._last_good_detection_step = self._episode_step
        return True, d_meas

    # ==================================================================
    # 3. ESTIMATOR
    # ==================================================================

    def _updateEstimator(self, ok, d_meas):
        dt = float(self.CTRL_TIMESTEP)
        if not np.isfinite(dt) or dt <= 0.0:
            self._require_finite(
                'CTRL_TIMESTEP', [dt],
                f'control timestep invalid (dt={dt})')

        prev_gap = self._steps_since_detection

        super()._updateEstimator(ok, d_meas)

        extra = {'dt': dt, 'gap_steps_before_update': prev_gap,
                 'detection_ok': bool(ok)}
        self._require_finite('_dv_est', self._dv_est,
                             'estimator relative-velocity non-finite', extra)
        self._require_finite('_d_est', self._d_est,
                             'estimator relative-position non-finite', extra)
        self._require_finite('_v_plat_est', self._v_plat_est,
                             'estimated platform velocity non-finite', extra)
        if self._prev_d_meas is not None:
            self._require_finite('_prev_d_meas', self._prev_d_meas,
                                 'finite-difference anchor non-finite', extra)

    # ==================================================================
    # 1. OBSERVATION
    # ==================================================================

    def _computeObs(self):
        obs = super()._computeObs()
        a = np.asarray(obs, dtype=float)
        if not np.all(np.isfinite(a)):
            state = self._getDroneStateVector(0)
            self._require_finite(
                'observation', a, 'non-finite observation',
                {'theta': np.asarray(state[7:10]).tolist(),
                 'v': np.asarray(state[10:13]).tolist(),
                 'omega': np.asarray(state[13:16]).tolist()})
        self._last_good_obs = a.copy()
        return obs

    # ==================================================================
    # 4. ACTION BOUNDARY
    # ==================================================================

    def _preprocessAction(self, action):
        self._episode_step += 1

        a = np.asarray(action, dtype=float).reshape(-1)
        self._require_finite(
            'action', a, 'non-finite ACTION arrived from the policy',
            {'note': 'environment produced finite obs but the actor returned '
                     'a non-finite action -- inspect model parameters and '
                     'replay buffer (requirement 6)'})

        state = self._getDroneStateVector(0)
        self._require_finite('drone_state@action', state[0:16],
                             'drone state non-finite before PID control')

        current_pos = np.asarray(state[0:3], dtype=float)
        next_pos = current_pos + self.ACTION_SCALE * np.clip(a, -1.0, 1.0)
        self._require_finite(
            'next_pos', next_pos,
            'PID target position non-finite -- this is what reaches '
            'DSLPIDControl and produces the SVD failure',
            {'current_pos': current_pos.tolist(),
             'action_scale': float(self.ACTION_SCALE)})

        self._last_good_action = a.copy()
        self._last_good_state = np.asarray(state[0:16], dtype=float).copy()

        return super()._preprocessAction(action)

    # ==================================================================
    # 5. REWARD
    # ==================================================================

    def _computeReward(self):
        r = super()._computeReward()
        if not np.isfinite(r):
            self._require_finite('reward', [r],
                                 'non-finite REWARD -- would poison the '
                                 'TD3 critic even with finite observations')
        return r

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self, seed=None, options=None):
        self._episode_index += 1
        self._episode_step = 0
        self._last_rvec = None
        self._last_tvec = None
        self._last_good_detection_step = None
        return super().reset(seed=seed, options=options)
