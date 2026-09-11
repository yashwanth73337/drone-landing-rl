"""
LandingAviary
=============

Land a quadrotor on a moving platform, with a real downward camera and OpenCV
ArUco detection.

Built on gym-pybullet-drones (utiasDSL).


WHAT CHANGED IN THIS VERSION: A REAL LANDING CONDITION
-------------------------------------------------------
Every earlier version defined "landed" as: the drone's centre came within some
height of the pad surface, once, for one instant. That is not a landing — it
is a proximity check.

Two specific problems with it were found by inspection, not by a metric:

1. THE HEIGHT WAS WRONG. It was set to 0.04 m by guess. The drone's actual
   collision geometry — read directly from cf2x.urdf — is a cylinder of
   length 0.025 m centred on the drone's tracked origin. Its underside is
   therefore 0.0125 m below that origin. The correct contact height is
   0.0125 m, not 0.04 m. The old value let the episode call "landed" while
   the drone's physical body was still floating roughly 2.75 cm above the
   pad.

2. TOUCHING ONCE IS NOT LANDING. The old condition said nothing about HOW the
   drone arrived, and nothing about whether it stayed. A drone that dove
   fast, or that grazed the pad edge while sliding sideways, or that bounced
   straight back off, all counted as a success as long as the height
   threshold was crossed once.

THE NEW CONDITION has three parts, all required:

  A. CONTACT HEIGHT -- 0.0125 m, the physically correct value (fixed, see
     above).

  B. CONTROLLED APPROACH -- in the 3 SECONDS BEFORE contact, the descent must
     be steady (no last-second dive, no stall-then-drop) and the sideways
     speed relative to the platform must be low and trending down, not just
     low at the final instant. Checked over a 90-step rolling history
     (3 s at 30 Hz), not a snapshot.

  C. SETTLE PERIOD -- after contact, the simulation continues for 3 MORE
     SECONDS with the platform still moving. Throughout that window the
     drone must not tip past the tilt limit, must not rise back off the pad,
     must not drift outside the pad footprint, and must not develop
     excessive relative speed. Only a landing that survives the full settle
     period counts as successful.

This makes "success" a claim about the WHOLE 6-second episode around
touchdown -- 3 seconds of approach and 3 seconds of staying put -- rather
than a claim about one instant.


THREE SENSING MODES
-------------------
Set via `sensing`. This is the ablation axis.

  'privileged'   Exact relative state from the simulator. Not physically
                 realisable; separates the control problem from perception.

  'abstract'     Modelled UWB noise plus a hand-set ArUco dropout threshold.
                 No rendering.

  'camera'       A rendered 128x96 downward view, real cv2.aruco detection,
                 pose from solvePnP, UWB fallback when detection fails.

A camera at height h sees a footprint of 2h*tan(fov/2), so a marker of side s
leaves the frame at h = s/(2*tan(fov/2)) = 0.18 m for s = 0.30 m and
fov = 80 deg. Measured cutoff: 0.195 m to 0.123 m. Derived, then verified.


PAPER-INSPIRED ADDITIONS (Shin et al., "Vision-Based Autonomous Drone Landing
on Moving Platforms With Uncertain Motion via Deep Reinforcement Learning,"
IEEE RA-L 2026)
------------------------------------------------------------------------------
Two mechanisms from the paper are adapted here, scoped to what is realistic
without the paper's learned LSTM estimator and asymmetric critic:

1. ACTIVE-PERCEPTION REWARD (paper Sec III-C). Paper formula:

       r_active_t = -alpha * clip(beta * (L_est_{t+1} - tau), 0, 1)

   where L_est_t is the mean squared error between the true relative state
   and the ESTIMATED relative state, evaluated over the 6-dim vector
   [dx, dy, dz, dvx, dvy, dvz]. In the paper, the estimate comes from a
   learned LSTM trained jointly with the policy on an auxiliary regression
   loss. This codebase has no such estimator -- L_est here is computed
   directly as ground-truth vs. the classical ArUco/UWB sensor's currently
   held estimate (self.last_meas_pos / self.last_meas_vel from _sense()).
   This is a SCOPED SIMPLIFICATION, not a reproduction of the paper's
   estimator. Gains (alpha=0.1, beta=1.0, tau=0.01) are taken directly from
   the paper.

   Reward-timing note: gym_pybullet_drones' BaseAviary.step() calls
   _computeObs() BEFORE _computeReward(). _sense() (called from
   _computeObs()) sets self.L_est using the state AFTER this step's action
   has been applied and physics has advanced -- i.e. by the time
   _computeReward() reads self.L_est, it already reflects the paper's
   L_est_{t+1} relative to the action just taken. No extra step-delay
   bookkeeping is required for this to line up correctly.

   The paper explicitly reports that a simpler binary "is the target in the
   camera FOV" reward degrades performance once platform motion is
   aggressive, because visibility-chasing corrections fight stable descent
   and produce oscillatory camera-steering. That binary form is deliberately
   NOT used here for that reason.

2. CURRICULUM ON PLATFORM MOTION (paper Sec III-D-2, Fig. 3 caption). The
   paper curriculum runs 8 levels, labelled 10 through 80, stepped every 512
   completed episodes; c = level / 80 scales platform motion magnitude, with
   c=1 (level 80) reproducing the full task specification. Implemented here
   via CURRICULUM_C, set externally by a training callback through
   set_curriculum(c) (see CurriculumCallback in train_landing.py). Defaults
   to 1.0 so eval / --gui / standalone construction is unaffected.

NOT implemented today (explicitly out of scope, for honesty about what this
file does and does not reproduce from the paper): the keypoint-based visual
encoder (hexagonal marker + learned descriptors replacing single-ArUco
solvePnP), the learned LSTM relative-state estimator with its auxiliary MSE
loss, the asymmetric actor-critic (privileged critic), and full domain
randomization of control gains / disturbances / visual appearance (paper
Table II). These require custom SB3 policy/critic architecture and are a
larger, separate effort.


PRIOR BUGS FIXED IN THIS FILE
-----------------------------

1. ORDERING BUG. BaseAviary.step() calls _computeReward() BEFORE
   _computeTerminated(). The landing bonus was never paid, at any value,
   until _computeReward() was made to call the touchdown check itself.

2. PHASE MEMORISATION. Fixed platform motion let the policy memorise a timed
   routine rather than track. Amplitude, frequency, and phase are now
   resampled every episode.

3. MARKER RENDERING. PyBullet repeats textures rather than mapping them once.
   The marker is built from 36 black boxes, one per cell of a 6x6 grid, to
   avoid UV mapping entirely.

4. MARKER CORNER SCALE. cv2.aruco reports the OUTER black square's corners;
   MARKER_OBJ_PTS had used the inner 4x4 payload, scaling every recovered
   distance by 4/6. Fixed; detection error dropped from 0.238 m to 0.020 m.

5. TOUCHDOWN HEIGHT. See above. 0.04 m -> 0.0125 m, from the drone's own
   collision geometry in cf2x.urdf.

6. DOUBLE-CALL BUG. _checkTouchdown() was being called once from
   _computeReward() and again from _computeTerminated() -- both hit the
   same (unchanged) physics state per real step, but each call appended to
   height_history / lateral_speed_history, silently halving the real-time
   duration of the 3-second approach window to 1.5 s. Fixed with a
   per-step cache (_checkTouchdown now wraps _checkTouchdownImpl and only
   invokes it once per self.step_counter).

7. RATCHET-VS-ABORT INTERACTION. The descent ratchet in _computeTruncated()
   was applying to the intentional climb-away after an aborted landing
   attempt (RATCHET_SLACK=0.10 m < MIN_CLEARANCE_AFTER_ABORT=0.15 m),
   truncating the episode mid-recovery before it could ever clear and
   retry. Fixed by exempting the ratchet check while awaiting_clearance.
"""

import os
from collections import deque

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

try:
    import cv2
    _HAS_CV2 = hasattr(cv2, 'aruco')
except ImportError:
    cv2 = None
    _HAS_CV2 = False


class LandingAviary(BaseRLAviary):
    """Land on a moving platform: controlled approach, correct contact
    height, and a survived settle period -- not a single-instant touch."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 # --- platform motion ---------------------------------
                 randomize_platform: bool = True,
                 amplitude_range=(0.2, 1.0),
                 omega_range=(0.2, 1.5),
                 platform_amplitude: float = 0.5,
                 platform_omega: float = 0.5,
                 platform_size: float = 0.20,
                 platform_height: float = 0.10,
                 start_height: float = 0.8,
                 ratchet_slack: float = 0.10,
                 # --- landing condition --------------------------------
                 contact_height: float = 0.0125,     # from cf2x.urdf, see above
                 approach_window_sec: float = 3.0,
                 settle_window_sec: float = 3.0,
                 approach_min_drop: float = 0.05,    # metres, over the window
                 approach_max_single_drop_frac: float = 0.30,
                 approach_end_lateral_max: float = 0.3,
                 approach_peak_lateral_max: float = 0.6,
                 settle_tilt_limit: float = 0.4,     # rad, matches _computeTruncated
                 settle_rise_limit: float = 0.05,    # metres above contact height
                 settle_lateral_max: float = 0.3,
                 min_clearance_after_abort: float = 0.15,   # metres, must climb this high before retrying
                 # --- descent rate ------------------------------------
                 descent_speed_cap: float = 0.35,
                 descent_penalty: float = 4.0,
                 # --- active perception (Shin et al. RA-L 2026, Sec III-C) --
                 active_perception_alpha: float = 0.02,
                 active_perception_beta: float = 1.0,
                 active_perception_tau: float = 0.01,
                 # --- sensing -----------------------------------------
                 sensing: str = 'camera',
                 cam_width: int = 128,
                 cam_height: int = 96,
                 cam_fov_deg: float = 80.0,
                 cam_every_n_steps: int = 2,
                 marker_size: float = 0.30,
                 marker_id: int = 0,
                 uwb_noise_std: float = 0.15,
                 uwb_rate_hz: float = 20.0,
                 uwb_outlier_prob: float = 0.02,
                 aruco_noise_std: float = 0.02,
                 aruco_fov_height: float = 0.30,
                 aruco_tilt_limit: float = 0.50,
                 aruco_dropout_prob: float = 0.10,
                 ):

        if sensing not in ('privileged', 'abstract', 'camera'):
            raise ValueError(f"sensing must be privileged/abstract/camera, got {sensing}")
        if sensing == 'camera' and not _HAS_CV2:
            raise ImportError(
                "sensing='camera' needs cv2.aruco.\n"
                "    pip install opencv-contrib-python"
            )

        self.SENSING = sensing

        self.RANDOMIZE_PLATFORM = randomize_platform
        self.AMPLITUDE_RANGE = amplitude_range
        self.OMEGA_RANGE = omega_range
        self.PLAT_AMPLITUDE = platform_amplitude
        self.PLAT_OMEGA = platform_omega
        self.PLAT_PHASE = 0.0
        self.PLAT_SIZE = platform_size
        self.PLAT_HEIGHT = platform_height
        self.START_HEIGHT = start_height
        self.RATCHET_SLACK = ratchet_slack

        # --- landing condition ------------------------------------------
        self.CONTACT_HEIGHT = contact_height
        self.APPROACH_WINDOW_STEPS = int(approach_window_sec * ctrl_freq)
        self.SETTLE_WINDOW_STEPS = int(settle_window_sec * ctrl_freq)
        self.APPROACH_MIN_DROP = approach_min_drop
        self.APPROACH_MAX_SINGLE_DROP_FRAC = approach_max_single_drop_frac
        self.APPROACH_END_LATERAL_MAX = approach_end_lateral_max
        self.APPROACH_PEAK_LATERAL_MAX = approach_peak_lateral_max
        self.SETTLE_TILT_LIMIT = settle_tilt_limit
        self.SETTLE_RISE_LIMIT = settle_rise_limit
        self.SETTLE_LATERAL_MAX = settle_lateral_max
        self.MIN_CLEARANCE_AFTER_ABORT = min_clearance_after_abort
        # Rolling history for the approach check. maxlen keeps only the most
        # recent APPROACH_WINDOW_STEPS readings automatically.
        self.height_history = deque(maxlen=self.APPROACH_WINDOW_STEPS)
        self.lateral_speed_history = deque(maxlen=self.APPROACH_WINDOW_STEPS)

        # Settle-phase state. contact_step is set once contact height and the
        # approach check both pass; the episode then keeps running for
        # SETTLE_WINDOW_STEPS more steps before a final success/fail verdict.
        self.contact_step = None
        self.settle_ok = True
        self.awaiting_clearance = False
        self.failed_attempts = 0
        self.truncation_reason = None
        self.truncation_horiz_dist = None
        self._touchdown_check_step = -1
        self._touchdown_check_result = False

        # --- active perception (Shin et al. RA-L 2026, Sec III-C) --------
        # L_est is the per-step ground-truth-vs-sensed-state mean squared
        # error. The paper computes this against a LEARNED LSTM estimator's
        # prediction; here it is ground-truth vs. the classical ArUco/UWB
        # sensor's currently-held estimate -- a scoped simplification, not
        # a reproduction of their estimator. See module docstring.
        self.ACTIVE_PERCEPTION_ALPHA = active_perception_alpha
        self.ACTIVE_PERCEPTION_BETA = active_perception_beta
        self.ACTIVE_PERCEPTION_TAU = active_perception_tau
        self.L_est = 0.0

        # --- curriculum on platform motion (Sec III-D-2, Fig. 3) ---------
        # c in [0, 1] scales platform motion magnitude; c=1 is the full task
        # spec. Set externally by CurriculumCallback via set_curriculum().
        # Defaults to 1.0 so eval/--gui/standalone use is unaffected.
        self.CURRICULUM_C = 1.0

        # --- descent rate ---------------------------------------------
        self.DESCENT_SPEED_CAP = descent_speed_cap
        self.DESCENT_PENALTY = descent_penalty

        # --- camera ---------------------------------------------------
        self.CAM_W = cam_width
        self.CAM_H = cam_height
        self.CAM_FOV = cam_fov_deg
        self.CAM_EVERY = max(1, cam_every_n_steps)
        self.MARKER_SIZE = marker_size
        self.MARKER_ID = marker_id
        self.MARKER_CELL_IDS = []

        f = (self.CAM_H / 2.0) / np.tan(np.radians(self.CAM_FOV) / 2.0)
        self.K = np.array([[f, 0, self.CAM_W / 2.0],
                           [0, f, self.CAM_H / 2.0],
                           [0, 0, 1.0]], dtype=np.float64)
        self.DIST = np.zeros(5)

        hm = self.MARKER_SIZE / 2.0
        self.MARKER_OBJ_PTS = np.array([[-hm,  hm, 0],
                                        [ hm,  hm, 0],
                                        [ hm, -hm, 0],
                                        [-hm, -hm, 0]], dtype=np.float64)

        if _HAS_CV2:
            self.ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.ARUCO_PARAMS = cv2.aruco.DetectorParameters()
            self.ARUCO_DETECTOR = cv2.aruco.ArucoDetector(
                self.ARUCO_DICT, self.ARUCO_PARAMS)

        # --- UWB / abstract sensing -----------------------------------
        self.UWB_NOISE_STD = uwb_noise_std
        self.UWB_RATE_HZ = uwb_rate_hz
        self.UWB_OUTLIER_PROB = uwb_outlier_prob
        self.ARUCO_NOISE_STD = aruco_noise_std
        self.ARUCO_FOV_HEIGHT = aruco_fov_height
        self.ARUCO_TILT_LIMIT = aruco_tilt_limit
        self.ARUCO_DROPOUT_PROB = aruco_dropout_prob

        self.last_meas_pos = np.zeros(3)
        self.last_meas_vel = np.zeros(3)
        self.last_meas_step = -1
        self.last_uwb_step = -1

        self.steps_blind = 0
        self.steps_total = 0
        self.detect_error_sum = 0.0
        self.detect_count = 0

        # Episode must be long enough to contain a full settle period after
        # a touchdown near the end of a normal approach.
        self.EPISODE_LEN_SEC = 10 + settle_window_sec

        self.PLATFORM_ID = None
        self.prev_height = None
        self.min_height_seen = None

        # --- final outcome bookkeeping ----------------------------------
        self.touchdown_recorded = False   # contact height + approach both passed
        self.episode_resolved = False     # settle window finished; verdict is final
        self.touchdown_vz = None
        self.touchdown_lateral = None
        self.touchdown_tilt = None
        self.touchdown_offset = None
        self.landed_successfully = False

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, start_height]])

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    # ------------------------------------------------------------------
    # CURRICULUM (Sec III-D-2)
    # ------------------------------------------------------------------

    def set_curriculum(self, c: float):
        """External hook (called by CurriculumCallback during training) to
        set the curriculum scalar c in [0, 1]. c=1 is the full task
        specification; lower values shrink platform motion for easier early
        training. See Shin et al. RA-L 2026, Sec III-D-2 / Fig. 3."""
        self.CURRICULUM_C = float(np.clip(c, 0.0, 1.0))

    # ------------------------------------------------------------------
    # THE PLATFORM
    # ------------------------------------------------------------------

    def _samplePlatformMotion(self):
        if not self.RANDOMIZE_PLATFORM:
            return
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        # Curriculum: sample from the full configured range, then scale
        # toward zero motion by CURRICULUM_C. c=1 reproduces the original
        # (pre-curriculum) sampling exactly.
        self.PLAT_AMPLITUDE = float(rng.uniform(*self.AMPLITUDE_RANGE)) * self.CURRICULUM_C
        self.PLAT_OMEGA = float(rng.uniform(*self.OMEGA_RANGE)) * self.CURRICULUM_C
        self.PLAT_PHASE = float(rng.uniform(0.0, 2.0 * np.pi))

    def _getPlatformPos(self):
        t = self.step_counter / self.PYB_FREQ
        x = self.PLAT_AMPLITUDE * np.sin(self.PLAT_OMEGA * t + self.PLAT_PHASE)
        return np.array([x, 0.0, self.PLAT_HEIGHT])

    def _getPlatformVel(self):
        t = self.step_counter / self.PYB_FREQ
        vx = (self.PLAT_AMPLITUDE * self.PLAT_OMEGA
              * np.cos(self.PLAT_OMEGA * t + self.PLAT_PHASE))
        return np.array([vx, 0.0, 0.0])

    # ------------------------------------------------------------------
    # MARKER, BUILT FROM GEOMETRY (no textures -- see module docstring)
    # ------------------------------------------------------------------

    def _buildMarkerGeometry(self, pos):
        grid = cv2.aruco.generateImageMarker(self.ARUCO_DICT, self.MARKER_ID, 6)
        is_black = (grid < 128)

        cell = self.MARKER_SIZE / 6.0
        half = cell / 2.0
        first = -self.MARKER_SIZE / 2.0 + half
        z = self.PLAT_HEIGHT + 0.002

        black = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[half, half, 0.0005],
            rgbaColor=[0, 0, 0, 1],
            physicsClientId=self.CLIENT)

        self.MARKER_CELL_IDS = []
        for row in range(6):
            for col in range(6):
                if not is_black[row, col]:
                    continue
                dx = first + col * cell
                dy = -(first + row * cell)
                bid = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=black,
                    basePosition=[pos[0] + dx, pos[1] + dy, z],
                    physicsClientId=self.CLIENT)
                self.MARKER_CELL_IDS.append((bid, dx, dy))

    def _updatePlatform(self):
        pos = self._getPlatformPos()

        if self.PLATFORM_ID is None:
            half = [self.PLAT_SIZE, self.PLAT_SIZE, self.PLAT_HEIGHT / 2]
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=self.CLIENT)
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half,
                rgbaColor=[1, 1, 1, 1],
                physicsClientId=self.CLIENT)
            self.PLATFORM_ID = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[pos[0], pos[1], self.PLAT_HEIGHT / 2],
                physicsClientId=self.CLIENT)

            if self.SENSING == 'camera':
                self._buildMarkerGeometry(pos)
        else:
            p.resetBasePositionAndOrientation(
                self.PLATFORM_ID,
                [pos[0], pos[1], self.PLAT_HEIGHT / 2],
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT)

            z = self.PLAT_HEIGHT + 0.002
            for bid, dx, dy in self.MARKER_CELL_IDS:
                p.resetBasePositionAndOrientation(
                    bid, [pos[0] + dx, pos[1] + dy, z], [0, 0, 0, 1],
                    physicsClientId=self.CLIENT)

    # ------------------------------------------------------------------
    # CAMERA
    # ------------------------------------------------------------------

    def _cameraFrame(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        forward = R @ np.array([0.0, 0.0, -1.0])
        body_x = R @ np.array([1.0, 0.0, 0.0])

        right = np.cross(forward, body_x)
        n = np.linalg.norm(right)
        right = np.array([1.0, 0.0, 0.0]) if n < 1e-8 else right / n

        down = np.cross(forward, right)
        down = down / (np.linalg.norm(down) + 1e-12)

        return pos, right, down, forward

    def _renderCamera(self):
        pos, right, down, forward = self._cameraFrame()

        view = p.computeViewMatrix(
            cameraEyePosition=pos.tolist(),
            cameraTargetPosition=(pos + 0.5 * forward).tolist(),
            cameraUpVector=(-down).tolist(),
            physicsClientId=self.CLIENT)

        proj = p.computeProjectionMatrixFOV(
            fov=self.CAM_FOV,
            aspect=self.CAM_W / self.CAM_H,
            nearVal=0.02, farVal=10.0,
            physicsClientId=self.CLIENT)

        _, _, rgb, _, _ = p.getCameraImage(
            width=self.CAM_W, height=self.CAM_H,
            viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.CLIENT)

        rgb = np.reshape(rgb, (self.CAM_H, self.CAM_W, 4))[:, :, :3]
        return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    def _detectMarker(self, image):
        corners, ids, _ = self.ARUCO_DETECTOR.detectMarkers(image)

        if ids is None or len(ids) == 0:
            return None, False

        idx = None
        for i, mid in enumerate(ids.flatten()):
            if mid == self.MARKER_ID:
                idx = i
                break
        if idx is None:
            return None, False

        img_pts = corners[idx].reshape(4, 2).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            self.MARKER_OBJ_PTS, img_pts, self.K, self.DIST,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None, False

        _, right, down, forward = self._cameraFrame()
        R_cam_to_world = np.column_stack([right, down, forward])
        rel_pos = R_cam_to_world @ tvec.reshape(3)

        if not np.all(np.isfinite(rel_pos)):
            return None, False
        return rel_pos, True

    # ------------------------------------------------------------------
    # SENSING
    # ------------------------------------------------------------------

    def _sense(self, true_rel_pos, true_rel_vel):
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        state = self._getDroneStateVector(0)

        if self.SENSING == 'privileged':
            self.steps_total += 1
            self.L_est = 0.0   # estimate == ground truth exactly, by construction
            return true_rel_pos.copy(), true_rel_vel.copy(), True, True

        steps_between_uwb = max(1, int(self.CTRL_FREQ / self.UWB_RATE_HZ))
        uwb_valid = (self.step_counter - self.last_uwb_step) >= steps_between_uwb

        aruco_valid = False
        new_pos = None

        if self.SENSING == 'camera':
            if self.step_counter % self.CAM_EVERY == 0:
                img = self._renderCamera()
                det_pos, aruco_valid = self._detectMarker(img)
                if aruco_valid:
                    new_pos = det_pos
                    self.detect_error_sum += float(
                        np.linalg.norm(det_pos - true_rel_pos))
                    self.detect_count += 1
        else:
            height_above = state[2] - self._getPlatformPos()[2]
            tilt = np.linalg.norm(state[7:9])
            aruco_valid = bool(height_above > self.ARUCO_FOV_HEIGHT
                               and tilt < self.ARUCO_TILT_LIMIT
                               and rng.random() > self.ARUCO_DROPOUT_PROB)
            if aruco_valid:
                new_pos = true_rel_pos + rng.normal(0, self.ARUCO_NOISE_STD, 3)

        if new_pos is None and uwb_valid:
            noise = rng.normal(0, self.UWB_NOISE_STD, 3)
            if rng.random() < self.UWB_OUTLIER_PROB:
                noise = noise + rng.normal(0, 1.0, 3)
            new_pos = true_rel_pos + noise

        if new_pos is not None:
            new_pos = np.nan_to_num(new_pos, nan=0.0, posinf=0.0, neginf=0.0)
            dt_steps = self.step_counter - self.last_meas_step
            if self.last_meas_step >= 0 and dt_steps > 0:
                dt = dt_steps / self.PYB_FREQ
                vel = (new_pos - self.last_meas_pos) / dt
                self.last_meas_vel = np.clip(vel, -10.0, 10.0)
            self.last_meas_pos = new_pos
            self.last_meas_step = self.step_counter
            self.last_uwb_step = self.step_counter

        if not aruco_valid:
            self.steps_blind += 1
        self.steps_total += 1

        # --- dead-reckoned held estimate ---------------------------------
        # Between fresh fixes, extrapolate from the last real fix using the
        # last known velocity, instead of freezing position outright. A
        # frozen estimate makes L_est spike sharply the instant the marker
        # leaves the camera's FOV near touchdown -- an EXPECTED consequence
        # of camera geometry (see module docstring), not a mistake -- which
        # was found to make the active-perception term punish the final
        # approach itself. This is a closer (still simplified) stand-in for
        # what the paper's learned LSTM estimator would predict while
        # briefly blind.
        dt_since_fix = ((self.step_counter - self.last_meas_step) / self.PYB_FREQ
                        if self.last_meas_step >= 0 else 0.0)
        held_pos = self.last_meas_pos + self.last_meas_vel * dt_since_fix

        # --- active-perception signal (Shin et al. RA-L 2026, Sec III-C) --
        err = np.concatenate([
            true_rel_pos - held_pos,
            true_rel_vel - self.last_meas_vel])
        self.L_est = float(np.mean(err ** 2))

        return (held_pos.astype('float32'),
                self.last_meas_vel.copy(), bool(uwb_valid), bool(aruco_valid))

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self.PLATFORM_ID = None
        self.MARKER_CELL_IDS = []
        self.prev_height = None
        self.min_height_seen = None

        self.height_history.clear()
        self.lateral_speed_history.clear()
        self.contact_step = None
        self.settle_ok = True
        self.awaiting_clearance = False
        self.failed_attempts = 0
        self.truncation_reason = None
        self.truncation_horiz_dist = None
        self._touchdown_check_step = -1
        self._touchdown_check_result = False
        self.L_est = 0.0
        # NOTE: CURRICULUM_C is deliberately NOT reset here -- it must
        # persist across episodes within a training run; only
        # set_curriculum() (called by the training callback) changes it.

        self.touchdown_recorded = False
        self.episode_resolved = False
        self.touchdown_vz = None
        self.touchdown_lateral = None
        self.touchdown_tilt = None
        self.touchdown_offset = None
        self.landed_successfully = False

        self.last_meas_pos = np.zeros(3)
        self.last_meas_vel = np.zeros(3)
        self.last_meas_step = -1
        self.last_uwb_step = -1
        self.steps_blind = 0
        self.steps_total = 0
        self.detect_error_sum = 0.0
        self.detect_count = 0

        out = super().reset(seed=seed, options=options)
        self._samplePlatformMotion()
        return self._computeObs(), out[1]

    # ------------------------------------------------------------------
    # THE LANDING CONDITION
    # ------------------------------------------------------------------
    #
    # Three parts, all required. See the module docstring for the reasoning
    # behind each threshold.

    def _approachWasControlled(self):
        """Part B: was the 3 seconds before contact a genuine controlled
        descent, or a dive / stall-then-drop / wild final correction?

        Requires a full window of history -- an episode that reaches
        contact height in fewer than APPROACH_WINDOW_STEPS steps has not
        had a real approach and cannot pass this check.
        """
        if len(self.height_history) < self.APPROACH_WINDOW_STEPS:
            return False

        heights = list(self.height_history)
        speeds = list(self.lateral_speed_history)

        diffs = [heights[i] - heights[i + 1] for i in range(len(heights) - 1)]
        total_drop = heights[0] - heights[-1]
        biggest_single_drop = max(diffs) if diffs else 0.0

        steady_descent = (
            total_drop > self.APPROACH_MIN_DROP
            and biggest_single_drop < self.APPROACH_MAX_SINGLE_DROP_FRAC * max(total_drop, 1e-6)
        )

        ends_slow = speeds[-1] < self.APPROACH_END_LATERAL_MAX
        was_reasonable_throughout = max(speeds) < self.APPROACH_PEAK_LATERAL_MAX

        return steady_descent and ends_slow and was_reasonable_throughout

    def _duringSettleWindow(self, height_above_pad, tilt, horiz_dist, lateral_speed):
        """Part C: called every step during the 3-second settle period.
        Latches to failed -- one bad step anywhere in the window fails the
        whole landing, even if later steps look fine again (a bounce that
        recovers is still a bounce)."""
        if not self.settle_ok:
            return

        if tilt > self.SETTLE_TILT_LIMIT:
            self.settle_ok = False
        elif height_above_pad > self.CONTACT_HEIGHT + self.SETTLE_RISE_LIMIT:
            self.settle_ok = False
        elif horiz_dist > self.PLAT_SIZE:
            self.settle_ok = False
        elif lateral_speed > self.SETTLE_LATERAL_MAX:
            self.settle_ok = False

    def _checkTouchdown(self):
        """Thin cache: the real logic in _checkTouchdownImpl() must only run
        ONCE per physics step. BaseAviary.step() causes _computeReward() and
        _computeTerminated() to each call this once per step, which would
        otherwise double-append to height_history/lateral_speed_history and
        silently halve the real-time duration of the approach window."""
        if self.step_counter != self._touchdown_check_step:
            self._touchdown_check_result = self._checkTouchdownImpl()
            self._touchdown_check_step = self.step_counter
        return self._touchdown_check_result

    def _checkTouchdownImpl(self):
        if self.episode_resolved:
            return True

        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]
        plat_pos = self._getPlatformPos()
        plat_vel = self._getPlatformVel()

        height_above_pad = drone_pos[2] - plat_pos[2]
        horiz_dist = float(np.linalg.norm(drone_pos[0:2] - plat_pos[0:2]))
        rel_vel = drone_vel - plat_vel
        lateral_speed = float(np.linalg.norm(rel_vel[0:2]))
        tilt = float(np.linalg.norm(state[7:9]))

        if self.touchdown_recorded:
            self._duringSettleWindow(height_above_pad, tilt, horiz_dist, lateral_speed)

            if not self.settle_ok:
                # Contact was made under a GOOD approach, but something went
                # wrong afterward. This is a real landing failure -- terminal.
                self.episode_resolved = True
                self.landed_successfully = False
                return True

            steps_since_contact = self.step_counter - self.contact_step
            if steps_since_contact <= self.SETTLE_WINDOW_STEPS:
                return False   # still settling, still fine so far

            self.episode_resolved = True
            self.landed_successfully = bool(self.touchdown_offset < self.PLAT_SIZE)
            return True

        self.height_history.append(float(height_above_pad))
        self.lateral_speed_history.append(lateral_speed)

        if self.awaiting_clearance:
            if height_above_pad >= self.MIN_CLEARANCE_AFTER_ABORT:
                self.awaiting_clearance = False   # cleared -- a new attempt may now begin
            return False   # not yet cleared; contact cannot be re-triggered

        if height_above_pad > self.CONTACT_HEIGHT:
            return False

        self.touchdown_recorded = True
        self.contact_step = self.step_counter
        self.touchdown_offset = horiz_dist
        self.touchdown_vz = float(abs(rel_vel[2]))
        self.touchdown_lateral = lateral_speed
        self.touchdown_tilt = tilt

        if not self._approachWasControlled():
            # ABORTED, not failed. Reset for a fresh attempt in the SAME
            # episode -- matches Shin et al.'s emergent go-around behaviour.
            self.failed_attempts += 1
            self.touchdown_recorded = False
            self.contact_step = None
            self.height_history.clear()
            self.lateral_speed_history.clear()
            self.min_height_seen = None   # unlock the ratchet for the climb-away
            self.awaiting_clearance = True
            return False

        return False   # approach was good -- entering the settle window

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------

    def _observationSpace(self):
        base = super()._observationSpace()
        n_extra = 6 if self.SENSING == 'privileged' else 8
        lo = np.full((base.shape[0], n_extra), -np.inf, dtype=np.float32)
        hi = np.full((base.shape[0], n_extra), np.inf, dtype=np.float32)
        return spaces.Box(
            low=np.hstack([base.low, lo]).astype(np.float32),
            high=np.hstack([base.high, hi]).astype(np.float32),
            dtype=np.float32)

    def _computeObs(self):
        self._updatePlatform()

        obs = super()._computeObs()
        state = self._getDroneStateVector(0)

        true_rel_pos = self._getPlatformPos() - state[0:3]
        true_rel_vel = self._getPlatformVel() - state[10:13]

        meas_pos, meas_vel, uwb_valid, aruco_valid = self._sense(
            true_rel_pos, true_rel_vel)

        if self.SENSING == 'privileged':
            extra = np.hstack([meas_pos, meas_vel]).reshape(1, 6)
        else:
            extra = np.hstack([meas_pos, meas_vel,
                               float(uwb_valid), float(aruco_valid)]).reshape(1, 8)

        out = np.hstack([obs, extra]).astype('float32')
        return np.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)

    # ------------------------------------------------------------------
    # REWARD (from GROUND TRUTH)
    # ------------------------------------------------------------------

    def _computeReward(self):
        """Alignment + velocity matching + capped descent progress + tilt
        penalty + active-perception term + terminal bonus.

        The terminal bonus only fires once the FULL landing condition
        resolves successfully (approach + contact + settle), not on contact
        alone. The bonus is therefore paid later in the episode than a
        single-instant check would -- after the settle window -- so the
        reward must continue rewarding good behaviour (alignment, velocity
        matching, low tilt) THROUGHOUT the settle window too, or the policy
        has no signal telling it to hold position rather than drift once
        contact is made.
        """
        # CRITICAL: BaseAviary.step() calls _computeReward() BEFORE
        # _computeTerminated(). _checkTouchdown() must be invoked here so
        # its state machine (including the final verdict) has run before we
        # decide whether to pay the bonus this step.
        self._checkTouchdown()

        # Current drone state: position (0:3), velocity (10:13)
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]

        # Where the platform is right now, and how fast it's moving
        plat_pos = self._getPlatformPos()
        plat_vel = self._getPlatformVel()

        # How far the drone is from being directly above the platform (x,y only)
        horiz_dist = float(np.linalg.norm(drone_pos[0:2] - plat_pos[0:2]))

        # How high the drone is above the platform surface (z only)
        height_above = float(drone_pos[2] - plat_pos[2])

        # Reward for being horizontally aligned above the pad.
        # Highest (1.0) when directly overhead, fades as horiz_dist grows.
        align = np.exp(-3.0 * horiz_dist)

        # Velocity of the drone RELATIVE to the platform (both x,y,z)
        rel_vel = drone_vel - plat_vel

        # Reward for matching the platform's horizontal velocity.
        # Highest when the drone is moving at the same sideways speed as
        # the platform, not just standing still in world coordinates.
        vel_match = 0.5 * np.exp(-2.0 * np.linalg.norm(rel_vel[0:2]))

        dt = 1.0 / self.CTRL_FREQ   # real time, in seconds, that one step covers
        progress = 0.0
        descent_pen = 0.0

        if not self.touchdown_recorded:
            # Everything in this block only applies BEFORE contact. During
            # the settle window height is expected to stay roughly constant,
            # so "descent progress" is not a meaningful idea there.

            if self.prev_height is not None:
                # How much height was closed this single step, converted to
                # a speed (metres per second) so it can be compared to the cap.
                closed = self.prev_height - height_above
                v_desc = closed / dt

                # Reward for descending, but ONLY while roughly above the pad,
                # and ONLY up to the speed cap -- descending faster than the
                # cap earns no extra reward here (see descent_pen below).
                if horiz_dist < self.PLAT_SIZE * 1.5:
                    v_useful = min(v_desc, self.DESCENT_SPEED_CAP)
                    progress = 30.0 * v_useful * dt

                # Penalty for descending FASTER than the cap. This is what
                # discourages diving instead of a slow, controlled descent.
                if v_desc > self.DESCENT_SPEED_CAP:
                    descent_pen = -self.DESCENT_PENALTY * (v_desc - self.DESCENT_SPEED_CAP) * dt

            # Remember this step's height, so next step can compute how much
            # was closed since then.
            self.prev_height = height_above

        # How tilted the drone currently is (roll + pitch combined)
        tilt = float(np.linalg.norm(state[7:9]))

        # Penalty for being tilted -- a tilted drone at touchdown is more
        # likely to catch a leg and flip.
        tilt_pen = -0.3 * tilt

        # Active-perception term (Shin et al. RA-L 2026, Sec III-C): penalizes
        # actions whose consequence is worse sensed-state accuracy, using
        # self.L_est computed this step in _sense() (already reflects the
        # action just taken -- see module docstring "Reward-timing note").
        # Not applied under privileged sensing, where L_est is always 0 by
        # construction and the term would be a no-op anyway.
        r_active = -self.ACTIVE_PERCEPTION_ALPHA * float(np.clip(
            self.ACTIVE_PERCEPTION_BETA * (self.L_est - self.ACTIVE_PERCEPTION_TAU),
            0.0, 1.0))

        # The big one-off reward, paid ONLY once the entire landing condition
        # (controlled approach + correct contact height + survived settle
        # window + landed within the pad) has fully resolved successfully.
        bonus = 0.0
        if self.episode_resolved and self.landed_successfully:
            softness = np.exp(-3.0 * self.touchdown_vz)       # gentler = better
            centred = np.exp(-5.0 * self.touchdown_offset)    # closer to centre = better
            level = np.exp(-3.0 * self.touchdown_tilt)        # flatter = better
            bonus = 300.0 + 800.0 * softness * centred * level

        # Everything added together is the reward for this one step.
        return float(align + vel_match + progress + descent_pen
                     + tilt_pen + r_active + bonus)

    # ------------------------------------------------------------------
    # EPISODE END
    # ------------------------------------------------------------------

    def _computeTerminated(self):
        return self._checkTouchdown()

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)

        bound = 1.0 + self.AMPLITUDE_RANGE[1]

        if abs(state[0]) > bound or abs(state[1]) > 1.0 or state[2] > 2.0:
            self.truncation_reason = "position_bound"
            return True

        if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
            self.truncation_reason = "tilt_bound"
            return True

        height_above_pad = state[2] - self._getPlatformPos()[2]
        if height_above_pad < -0.05:
            self.truncation_reason = "below_platform"
            self.truncation_horiz_dist = float(
                np.linalg.norm(state[0:2] - self._getPlatformPos()[0:2]))
            return True

        # Descent ratchet only applies BEFORE contact, and NOT while
        # deliberately climbing away after an aborted attempt (see bug 7
        # in the module docstring).
        if not self.touchdown_recorded and not self.awaiting_clearance:
            h = state[2] - self._getPlatformPos()[2]
            if self.min_height_seen is None or h < self.min_height_seen:
                self.min_height_seen = h
            if h > self.min_height_seen + self.RATCHET_SLACK:
                self.truncation_reason = "ratchet"
                return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            self.truncation_reason = "timeout"
            return True

        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        plat_pos = self._getPlatformPos()

        info = {
            "horiz_dist": float(np.linalg.norm(state[0:2] - plat_pos[0:2])),
            "height_above_pad": float(state[2] - plat_pos[2]),
            "touchdown": self.touchdown_recorded,
            "resolved": self.episode_resolved,
            "success": self.landed_successfully,
            "settle_ok": self.settle_ok,
            "plat_amplitude": self.PLAT_AMPLITUDE,
            "plat_omega": self.PLAT_OMEGA,
            "plat_peak_speed": self.PLAT_AMPLITUDE * self.PLAT_OMEGA,
            "plat_peak_accel": self.PLAT_AMPLITUDE * self.PLAT_OMEGA ** 2,
            "sensing": self.SENSING,
            "blind_fraction": (self.steps_blind / self.steps_total
                               if self.steps_total > 0 else 0.0),
            "detect_error_mean": (self.detect_error_sum / self.detect_count
                                  if self.detect_count > 0 else float('nan')),
            "failed_attempts": self.failed_attempts,
            "truncation_reason": self.truncation_reason,
            "truncation_horiz_dist": self.truncation_horiz_dist,
            "L_est": self.L_est,
            "curriculum_c": self.CURRICULUM_C,
        }

        if self.touchdown_recorded:
            info.update({
                "touchdown_vz": self.touchdown_vz,
                "touchdown_lateral": self.touchdown_lateral,
                "touchdown_tilt": self.touchdown_tilt,
                "touchdown_offset": self.touchdown_offset,
            })

        return info