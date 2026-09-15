"""
ArucoLanderAviary
=================

Stage V2 of the vision-based landing work.

Subclasses VisionLanderAviary and replaces the PRIVILEGED relative-state
components of the observation with an ArUco-derived estimate. Nothing else
changes: physics, reward, contact/success logic, curriculum hooks and the
PID-mediated action mechanism are all inherited untouched.

    LanderAIAviary          (frozen -- privileged baseline)
        |
    VisionLanderAviary      (frozen -- V1, camera observes but never feeds
        |                    the policy)
        |
    ArucoLanderAviary       (this file -- V2, camera FEEDS the policy)

NO TRAINING HAPPENS IN V2. The point is to take the existing frozen TD3
checkpoints and measure what it costs them to fly on estimated state instead
of ground truth. One variable moves.


WHAT IS REPLACED, AND WHAT IS NOT
------------------------------------------------------------------------------
LanderAIAviary's Eq.3 observation is [theta, v, omega, d, delta_v]:

    theta    attitude            PROPRIOCEPTIVE  -- kept truthful (IMU)
    v        drone velocity      PROPRIOCEPTIVE  -- kept truthful (IMU/VIO)
    omega    angular velocity    PROPRIOCEPTIVE  -- kept truthful (IMU)
    d        relative pad pos    TARGET PROPERTY -- REPLACED by ArUco
    delta_v  relative pad vel    TARGET PROPERTY -- REPLACED by ArUco

The first three describe the drone's own body and a real airframe measures
them onboard, so leaving them truthful is not a privilege leak. The last two
describe the target, and those are what vision has to supply.

The observation SPACE is unchanged: Box(-1, 1, shape=(15,)). It has to be,
or the frozen checkpoints will not load. A direct consequence worth stating
plainly: THERE IS NO VALIDITY FLAG. The policy receives fifteen numbers and
has no way to know whether the relative-state components are a fresh
detection or a stale extrapolation. That is a real limitation of V2 and it
is exactly the gap a learned temporal estimator is meant to close later.


THE TRANSFORM CHAIN (the part most likely to be silently wrong)
------------------------------------------------------------------------------
solvePnP returns the marker's pose in the CAMERA frame. The observation
needs the pad's top-surface centre relative to the drone, in the WORLD
frame. The chain:

    marker_world  = eye + R_wc @ tvec
    target_world  = marker_world - [0, 0, MARKER_PLANE_OFFSET]
    d_est         = target_world - drone_pos

Substituting eye = drone_pos + R_body @ CAM_OFFSET_BODY, the drone position
CANCELS COMPLETELY:

    d_est = R_body @ CAM_OFFSET_BODY
            + R_wc @ tvec
            - [0, 0, MARKER_PLANE_OFFSET]

This matters for more than tidiness. It means the estimate depends only on
(a) the drone's own attitude, which is proprioceptive, (b) two fixed
calibration constants, and (c) solvePnP's output. No world position of
either body is used. The estimate is genuinely non-privileged, and that is
provable from the expression rather than asserted.

MARKER_PLANE_OFFSET is not cosmetic. The marker's drawing surface sits above
the pad's top face, and _getLandingTargetPos() returns the pad top. Omitting
this term puts a constant bias into every observation the policy sees. Two
bugs of exactly this kind were found in V1.


RELATIVE VELOCITY -- DO NOT SUBTRACT DRONE VELOCITY TWICE
------------------------------------------------------------------------------
Since d = p_platform - p_drone,

    d(d)/dt = v_platform - v_drone = delta_v

so finite-differencing the estimated RELATIVE position already yields the
relative velocity the observation wants. Subtracting the drone's own
velocity from that result again would double-count it and produce an
observation that is wrong but entirely plausible-looking. delta_v_est is
therefore taken as the direct finite difference of d_est and nothing else.


BEHAVIOUR DURING VISUAL DROPOUT
------------------------------------------------------------------------------
V1 measured the marker fully visible only ~51% of control steps, with the
blind stretches concentrated in the terminal approach. What the policy is
fed during those steps is the central design choice of V2, and both options
are implemented so the difference between them is itself a measurement.

  estimator_mode='hold'
      Zero-order hold. d_est and delta_v_est are frozen at their last
      detected values. The platform keeps moving at up to 0.25 m/s while
      the estimate does not, so error accumulates linearly with the length
      of the blind stretch, against a 0.1 m success radius.

  estimator_mode='predict'
      Constant-velocity prediction (dead reckoning). NOT an EKF: there is no
      covariance propagation and no measurement update -- constant velocity
      is only the motion model an EKF would coast on during dropout, and
      that model alone is what is implemented here.

      At the last detection the platform's world velocity is recovered as

          v_plat_est = delta_v_est + v_drone        (v_drone proprioceptive)

      and held. On each blind step the relative velocity is re-formed
      against the drone's CURRENT measured velocity,

          delta_v_pred = v_plat_est - v_drone_now

      and the position integrated forward, d_est += delta_v_pred * dt.

      Re-forming delta_v each step rather than holding it matters: the drone
      is actively manoeuvring during the blind stretch, so its own velocity
      changes even though the platform's does not. Holding delta_v constant
      would attribute the drone's own acceleration to the platform.

      In this environment the platform genuinely does move at constant
      velocity (LanderAIAviary._samplePlatformMotion samples one velocity
      per episode), so this model is well matched to the task. That is a
      property of the environment, not a general result, and any comparison
      to the paper has to say so.

  estimator_mode='off'
      Observation stays privileged, exactly as VisionLanderAviary. The
      estimator still runs and logs, so estimation error can be measured
      against a known-good trajectory. This is what --mode selfcheck uses.


NO PRIVILEGED INITIALISATION
------------------------------------------------------------------------------
The estimator is cleared on every reset. Before the first valid detection
there is NOTHING to report, and ground truth is not consulted to fill the
gap -- not even once, not as a seed.

The documented neutral initialisation is d_est = 0, delta_v_est = 0, which
normalises to the centre of the observation range. It is not a good guess:
d = 0 asserts the pad is exactly at the drone's position. It is chosen
because it is neutral and obviously wrong rather than plausibly wrong, so a
policy relying on it will fail visibly instead of subtly. The number of
steps spent in this state is recorded as steps_before_first_detection and
must be reported.

Ground truth is available in info[] for diagnostics only and never routes
into _computeObs().
"""

import numpy as np
import pybullet as p
import cv2

from envs.VisionLanderAviary import VisionLanderAviary


class ArucoLanderAviary(VisionLanderAviary):
    """VisionLanderAviary with the relative-state observation supplied by
    ArUco instead of by the simulator."""

    VALID_MODES = ('off', 'hold', 'predict')

    def __init__(self,
                 *args,
                 estimator_mode: str = 'predict',
                 strict_no_privileged: bool = False,
                 **kwargs):

        if estimator_mode not in self.VALID_MODES:
            raise ValueError(
                f"estimator_mode must be one of {self.VALID_MODES}, "
                f"got {estimator_mode!r}"
            )
        self.ESTIMATOR_MODE = estimator_mode

        # V3a ADDITION (additive; default False preserves V2 exactly).
        # When True, _computeObs() may NEVER return the parent's
        # privileged observation vector. The scientific claim of V3a --
        # 'no privileged target-relative state can reach the actor' --
        # depends on this being enforced rather than being a property of
        # call ordering several frames up the stack.
        self.STRICT_NO_PRIVILEGED = bool(strict_no_privileged)

        # Cumulative count of actual camera rasterisations, NOT reset
        # between episodes. Used by bench_v3a.py to assert that exactly
        # one render occurred per control step.
        self._total_renders = 0

        # Cumulative count of detections rejected because solvePnP
        # returned a non-finite pose. NOT reset between episodes.
        self._total_pnp_rejected = 0

        # The parent is forced to 'geometry'. Its _computeInfo() renders when
        # camera_mode == 'render', and this class does its own rendering
        # inside _refreshDetection(); leaving the parent in 'render' would
        # rasterise the scene twice per step. Requirement: render exactly
        # once per physical step, cached, consumed by both _computeObs() and
        # _computeInfo().
        kwargs['camera_mode'] = 'geometry'

        self._resetEstimatorState()
        # Incremented once per control step in _preprocessAction(). Used as
        # the detection cache key. self.step_counter cannot serve this role:
        # BaseAviary increments it only AFTER _computeObs/_computeReward/
        # _computeInfo have all run, so the first step and the reset would
        # share a key and the first step would silently reuse the reset's
        # detection.
        self._det_epoch = 0
        self._det_cached_epoch = None
        self._det_cache = {}

        super().__init__(*args, **kwargs)

    # ==================================================================
    # ESTIMATOR STATE
    # ==================================================================

    def _resetEstimatorState(self):
        """Clear every estimator variable. Called on reset, no exceptions.

        Nothing here is seeded from the simulator. See the module docstring,
        'NO PRIVILEGED INITIALISATION'.
        """
        self._d_est = np.zeros(3)
        self._dv_est = np.zeros(3)
        self._v_plat_est = np.zeros(3)      # world-frame, for 'predict'
        self._have_detection = False        # any detection yet this episode
        self._last_detection_valid = False  # detection on THIS step
        self._prev_d_meas = None            # for the finite difference
        self._steps_since_detection = 0
        self._steps_before_first_detection = 0
        self._blind_run = 0                 # current blind stretch length
        self._blind_runs = []               # completed blind stretch lengths
        self._n_steps = 0
        self._n_detections = 0
        self._n_pnp_rejected = 0     # degenerate solves this episode

    # ==================================================================
    # DETECTION -- rendered and cached exactly once per control step
    # ==================================================================

    def _preprocessAction(self, action):
        # One control step begins here. Bumping the epoch invalidates the
        # detection cache for the step about to be taken. The parent's
        # behaviour is otherwise untouched.
        self._det_epoch += 1
        return super()._preprocessAction(action)

    def _detectRelativePosition(self):
        """Render, detect, and recover d_meas = pad_top_centre - drone_pos.

        Returns (ok, d_meas). d_meas is in the WORLD frame. Uses only the
        drone's attitude and two calibration constants -- see the module
        docstring for why the drone's position cancels out.
        """
        gray = self._renderCamera()
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)

        if ids is None or self.MARKER_ID not in ids.flatten():
            return False, None

        idx = int(np.where(ids.flatten() == self.MARKER_ID)[0][0])
        img_pts = corners[idx].reshape(4, 2).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            self._markerCornersObject(), img_pts,
            self.CAM_K, self.CAM_DIST,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return False, None

        # DEGENERATE-SOLVE REJECTION.
        #
        # SOLVEPNP_IPPE_SQUARE can return ok=True with a NaN rvec: it
        # is an analytic planar solver, and near-collinear image
        # corners make the rotation recovery singular while the
        # translation still solves. Observed directly during V3a
        # training at an extreme viewing angle:
        #     rvec=[nan nan nan]  tvec=[0.037 -0.139 0.432]
        #
        # rvec is not used downstream, but its being NaN means the
        # solve was already degenerate, and on some of those calls
        # tvec is non-finite too. That NaN then reaches the actor and
        # ultimately DSLPIDControl, where Rotation.from_matrix raises
        # 'SVD did not converge'. np.clip does NOT stop it:
        # np.clip(np.nan, -1, 1) is still np.nan.
        #
        # Rejecting is the correct semantics, not a workaround: a
        # degenerate solve is a perception failure, and a blind step
        # already has well-defined handling in _updateEstimator().
        if not (np.all(np.isfinite(np.asarray(rvec, dtype=float)))
                and np.all(np.isfinite(np.asarray(tvec, dtype=float)))):
            self._n_pnp_rejected += 1
            self._total_pnp_rejected += 1
            return False, None

        state = self._getDroneStateVector(0)
        R_body = np.array(
            p.getMatrixFromQuaternion(state[3:7]), dtype=float
        ).reshape(3, 3)

        # _cameraPose() derives R_wc from attitude alone; eye is discarded
        # here precisely because the drone's world position must not enter.
        _, R_wc = self._cameraPose()

        d_meas = (
            R_body @ self.CAM_OFFSET_BODY           # camera mount offset
            + R_wc @ np.asarray(tvec).reshape(3)    # camera -> world
            - np.array([0.0, 0.0, self.MARKER_PLANE_OFFSET])  # marker -> pad
        )

        # Belt and braces: the drone's own attitude feeds R_body and
        # R_wc, so a non-finite physics state would also poison d_meas
        # even with a clean solvePnP result. Same treatment.
        if not np.all(np.isfinite(d_meas)):
            self._n_pnp_rejected += 1
            self._total_pnp_rejected += 1
            return False, None

        return True, d_meas

    def _refreshDetection(self):
        """Run the detection for this control step, or reuse the cache."""
        if self._det_cached_epoch == self._det_epoch:
            return self._det_cache

        self._total_renders += 1
        ok, d_meas = self._detectRelativePosition()
        self._updateEstimator(ok, d_meas)

        self._det_cache = {
            'detected': bool(ok),
            'd_meas': None if d_meas is None else d_meas.copy(),
            'd_est': self._d_est.copy(),
            'dv_est': self._dv_est.copy(),
            'steps_since_detection': self._steps_since_detection,
            'have_detection': self._have_detection,
        }
        self._det_cached_epoch = self._det_epoch
        return self._det_cache

    def _updateEstimator(self, ok, d_meas):
        """Advance the estimator by one control step."""
        dt = self.CTRL_TIMESTEP
        state = self._getDroneStateVector(0)
        v_drone = np.asarray(state[10:13], dtype=float)

        self._n_steps += 1

        if ok:
            self._n_detections += 1
            self._last_detection_valid = True

            if self._prev_d_meas is not None and self._steps_since_detection == 0:
                # Consecutive detections: finite-difference the relative
                # position. This IS delta_v -- the drone's own velocity must
                # NOT be subtracted again.
                self._dv_est = (d_meas - self._prev_d_meas) / dt
            elif self._prev_d_meas is not None:
                # Detection resumed after a gap. Divide by the true elapsed
                # time, not by dt, or the velocity is inflated by the gap
                # length.
                gap = (self._steps_since_detection + 1) * dt
                self._dv_est = (d_meas - self._prev_d_meas) / gap
            # else: first ever detection -- no basis for a velocity yet, so
            # dv_est stays at its neutral zero rather than being invented.

            self._d_est = d_meas.copy()
            self._prev_d_meas = d_meas.copy()

            # Platform world velocity, for the 'predict' model. v_drone is
            # proprioceptive.
            self._v_plat_est = self._dv_est + v_drone

            if not self._have_detection:
                self._have_detection = True

            if self._blind_run > 0:
                self._blind_runs.append(self._blind_run)
                self._blind_run = 0
            self._steps_since_detection = 0
            return

        # ---- no detection this step ----
        self._last_detection_valid = False
        self._steps_since_detection += 1
        self._blind_run += 1

        if not self._have_detection:
            # Still waiting for the first detection. Neutral state stands;
            # ground truth is not consulted.
            self._steps_before_first_detection += 1
            return

        if self.ESTIMATOR_MODE == 'predict':
            # Platform assumed to continue at its last estimated world
            # velocity. Relative velocity is re-formed against the drone's
            # CURRENT measured velocity, since the drone is manoeuvring
            # even though the platform is not.
            self._dv_est = self._v_plat_est - v_drone
            self._d_est = self._d_est + self._dv_est * dt
        # 'hold' and 'off': d_est and dv_est are left exactly as they were.

    # ==================================================================
    # OBSERVATION
    # ==================================================================

    def _computeObs(self):
        # During super().reset() the platform may not exist yet.
        #
        # V2 behaviour (strict_no_privileged=False): defer to the parent,
        # which creates the platform and returns the privileged vector.
        # That value is discarded by this class's reset() override, which
        # recomputes afterwards -- so it never actually reaches the policy.
        # But 'does not leak because of call ordering three frames up' is
        # not a property that should be load-bearing during training.
        #
        # V3a behaviour (strict_no_privileged=True): materialise the
        # platform so the camera has a scene, then fall through to the
        # ESTIMATED path. The privileged vector is never returned at all.
        # Raising outright here was considered and rejected: this branch is
        # genuinely reached during reset, before the platform body exists,
        # so an unconditional raise would break every episode reset. The
        # raise below fires only if the privileged fallback would actually
        # have been reached.
        if self.PLAT_ID is None:
            if self.STRICT_NO_PRIVILEGED and self.ESTIMATOR_MODE != 'off':
                self._updatePlatform(step_offset=0)
                if self.PLAT_ID is None:
                    raise RuntimeError(
                        'ArucoLanderAviary: strict_no_privileged=True but the '
                        'platform could not be created, so _computeObs() would '
                        'have had to fall back to the parent PRIVILEGED '
                        'observation. Refusing to do so -- a V3a run must not '
                        'feed the actor ground-truth target state.'
                    )
            else:
                return super()._computeObs()

        if self.ESTIMATOR_MODE == 'off':
            # Privileged observation, but the estimator still runs so its
            # error can be logged against a known-good trajectory.
            self._refreshDetection()
            return super()._computeObs()

        det = self._refreshDetection()

        state = self._getDroneStateVector(0)
        theta = state[7:10]      # proprioceptive
        v = state[10:13]         # proprioceptive
        omega = state[13:16]     # proprioceptive

        d = det['d_est']         # ESTIMATED
        delta_v = det['dv_est']  # ESTIMATED

        def clip_norm(x, bound):
            return np.clip(x, -bound, bound) / bound

        theta_n = clip_norm(theta, self.ATTITUDE_BOUND)
        v_n = np.array([
            clip_norm(v[0], self.VEL_XY_BOUND),
            clip_norm(v[1], self.VEL_XY_BOUND),
            clip_norm(v[2], self.VEL_Z_BOUND)])
        omega_n = clip_norm(omega, self.ANG_VEL_BOUND)
        d_n = clip_norm(d, self.REL_POS_BOUND)
        delta_v_n = clip_norm(delta_v, self.REL_VEL_BOUND)

        obs = np.concatenate([theta_n, v_n, omega_n, d_n, delta_v_n])
        return obs.astype('float32')

    # ==================================================================
    # INFO -- diagnostics only. Ground truth appears HERE and nowhere else.
    # ==================================================================

    def _computeInfo(self):
        info = super()._computeInfo()

        if self.PLAT_ID is None:
            return info

        det = self._refreshDetection()

        # Ground truth, for logging and selfcheck ONLY.
        state = self._getDroneStateVector(0)
        d_true = self._getLandingTargetPos() - state[0:3]
        dv_true = self._getPlatformVel() - state[10:13]

        d_est = det['d_est']
        dv_est = det['dv_est']

        info.update({
            'est_mode': self.ESTIMATOR_MODE,
            'est_detected': det['detected'],
            'est_have_detection': det['have_detection'],
            'est_steps_since_detection': det['steps_since_detection'],
            'est_steps_before_first_detection':
                self._steps_before_first_detection,
            'est_blind_run': self._blind_run,
            'est_detection_rate': (
                self._n_detections / self._n_steps if self._n_steps else 0.0
            ),
            'est_pnp_rejected': int(self._n_pnp_rejected),
            # Axis-wise SIGNED errors. Norms hide frame swaps and datum
            # offsets; a constant bias on one axis is the signature of a
            # bad transform and only shows up signed and per-axis.
            'est_err_dx': float(d_est[0] - d_true[0]),
            'est_err_dy': float(d_est[1] - d_true[1]),
            'est_err_dz': float(d_est[2] - d_true[2]),
            'est_err_dvx': float(dv_est[0] - dv_true[0]),
            'est_err_dvy': float(dv_est[1] - dv_true[1]),
            'est_err_dvz': float(dv_est[2] - dv_true[2]),
            'est_err_pos_norm': float(np.linalg.norm(d_est - d_true)),
            'est_err_vel_norm': float(np.linalg.norm(dv_est - dv_true)),
            'est_d_est_x': float(d_est[0]),
            'est_d_est_y': float(d_est[1]),
            'est_d_est_z': float(d_est[2]),
            'est_d_true_x': float(d_true[0]),
            'est_d_true_y': float(d_true[1]),
            'est_d_true_z': float(d_true[2]),
        })
        return info

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self, seed=None, options=None):
        self._resetEstimatorState()
        self._det_cached_epoch = None
        self._det_cache = {}

        obs, info = super().reset(seed=seed, options=options)

        # LanderAIAviary.reset() re-samples the platform motion AFTER the
        # base reset has already produced an observation, then recomputes.
        # The platform therefore moved between those two points, so any
        # detection taken earlier is stale. Clear the estimator again and
        # force one fresh detection against the actual episode geometry.
        self._resetEstimatorState()
        self._det_epoch += 1
        self._det_cached_epoch = None

        obs = self._computeObs()
        info = self._computeInfo()
        return obs, info

    # ==================================================================
    # EPISODE SUMMARY -- for the evaluation script
    # ==================================================================

    def episodeEstimatorSummary(self):
        """Blind-stretch statistics for the episode just finished."""
        runs = list(self._blind_runs)
        if self._blind_run > 0:
            runs.append(self._blind_run)   # stretch open at termination
        return {
            'steps': self._n_steps,
            'detections': self._n_detections,
            'detection_rate': (
                self._n_detections / self._n_steps if self._n_steps else 0.0
            ),
            'blind_fraction': (
                1.0 - self._n_detections / self._n_steps
                if self._n_steps else 0.0
            ),
            'steps_before_first_detection':
                self._steps_before_first_detection,
            'pnp_rejected': int(self._n_pnp_rejected),
            'pnp_rejection_rate': (
                self._n_pnp_rejected / self._n_steps
                if self._n_steps else 0.0
            ),
            'ever_detected': self._have_detection,
            'blind_runs': runs,
            'longest_blind_run': max(runs) if runs else 0,
            'mean_blind_run': float(np.mean(runs)) if runs else 0.0,
        }
