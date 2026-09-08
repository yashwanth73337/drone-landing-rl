"""
LandingAviary
=============

Land a quadrotor on a moving platform, with a real downward camera and OpenCV
ArUco detection.

Built on gym-pybullet-drones (utiasDSL).


WHAT CHANGED IN THIS VERSION: THE DESCENT-RATE PENALTY
------------------------------------------------------
Measured over three seeds with camera sensing:

    success                45.7 +/- 0.5 %
    touchdown vz           0.461 +/- 0.114 m/s
    vz > 0.5 m/s           43.5 +/- 30.7 %

The drone was not settling onto the pad — it was dropping onto it. Roughly
four landings in ten exceeded the touchdown-velocity threshold.

THE CAUSE WAS IN THE REWARD. The progress term paid

    30 * (previous_height - current_height)

a flat rate per metre closed, regardless of how fast that metre was closed.
Descending twice as fast earned the same total, but collected it sooner — and
sooner is worth more under discounting (gamma = 0.99). So the reward actively
preferred a dive to a controlled descent.

The touchdown bonus did contain a softness factor,

    softness = exp(-3 * touchdown_vz)

but that is one payment at the end against a per-step incentive to hurry. The
per-step term won.

FIX: cap the descent speed that earns progress reward.

    v_desc     = (previous_height - current_height) / dt
    v_useful   = min(v_desc, DESCENT_SPEED_CAP)
    progress   = 30 * v_useful * dt

Below the cap, behaviour is unchanged. Above it, descending faster earns
nothing extra, so there is no longer a reason to dive. A small explicit
penalty on exceeding the cap makes the gradient point the right way rather
than merely flattening.

This is a HAND-TUNED PATCH, and it is deliberately framed as one. The
principled version is a constraint rather than a penalty — maximise success
subject to touchdown velocity <= 0.5 m/s, with a Lagrange multiplier that
tunes itself. That is the next piece of work, and this version is the baseline
it has to beat. Nine reward iterations in this project were spent
hand-balancing weights; this is the tenth, and the argument for stopping.


THREE SENSING MODES
-------------------
Set via `sensing`. This is the ablation axis.

  'privileged'   Exact relative state from the simulator. Not physically
                 realisable; separates the control problem from perception.

  'abstract'     Modelled UWB noise plus a hand-set ArUco dropout threshold.
                 No rendering.

  'camera'       A rendered 128x96 downward view, real cv2.aruco detection,
                 pose from solvePnP, UWB fallback when detection fails.

In camera mode nothing about the dropout is assumed. A camera at height h sees
a footprint of 2h*tan(fov/2), so a marker of side s leaves the frame at
h = s/(2*tan(fov/2)) = 0.18 m for s = 0.30 m and fov = 80 deg. Measured cutoff:
0.195 m to 0.123 m. Derived, then verified.


PRIOR BUGS FIXED IN THIS FILE
-----------------------------

1. ORDERING BUG. BaseAviary.step() calls _computeReward() BEFORE
   _computeTerminated(), and _checkTouchdown() — which sets
   touchdown_recorded — lived only in the latter. On the touchdown step the
   reward function saw the flag as False, skipped the bonus, and the episode
   then ended. The landing bonus was NEVER paid, at any value. Eight reward
   variants failed for this one reason.

2. PHASE MEMORISATION. With fixed amplitude and frequency, and every episode
   starting at t=0, the platform's position was predictable from the step
   counter alone. The policy memorised a timed routine and scored 100% while
   having learned nothing transferable. A frequency sweep exposed it: success
   was non-monotonic, failing at 0.2 rad/s, succeeding at 0.5, failing at 0.8
   and 1.2, succeeding again at 1.8. Motion is now resampled every episode,
   phase included.

3. MARKER RENDERING. PyBullet repeats textures rather than mapping them once —
   on a GEOM_BOX the texture stretches across all six faces, and plane.obj has
   UVs set to repeat, tiling the marker 5x4. The marker is now built from 36
   black boxes, one per cell of the 6x6 grid. No texture, no UV mapping.

4. MARKER CORNER SCALE. cv2.aruco reports the corners of the marker's outer
   black square. MARKER_OBJ_PTS had been set to the inner 4x4 payload, scaling
   every recovered distance by 4/6. Detection error was 0.238 m; after the fix,
   0.020 m. Caught by verify_camera.py before any training.
"""

import os
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
    """Land on a moving platform, with a real camera and ArUco detection."""

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
                 # --- descent rate ------------------------------------
                 descent_speed_cap: float = 0.35,   # m/s, below the 0.5 limit
                 descent_penalty: float = 4.0,      # per m/s over the cap
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

        # --- descent rate ---------------------------------------------
        # Cap set at 0.35 m/s, comfortably below the 0.5 m/s safety threshold.
        # The margin matters: the cap governs the descent, but the last few
        # centimetres are ballistic once thrust is reduced, so touchdown speed
        # ends up slightly above the commanded rate.
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

        # cv2.aruco reports the corners of the marker's OUTER black square,
        # which in this geometry is the full 6x6 extent.
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

        self.EPISODE_LEN_SEC = 10
        self.PLATFORM_ID = None
        self.prev_height = None
        self.min_height_seen = None

        # --- touchdown bookkeeping ------------------------------------
        self.touchdown_recorded = False
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
    # THE PLATFORM
    # ------------------------------------------------------------------

    def _samplePlatformMotion(self):
        """New motion parameters each episode. Phase matters most: without it
        every episode begins with the platform at x = 0 moving in +x, itself a
        memorisable cue."""
        if not self.RANDOMIZE_PLATFORM:
            return
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        self.PLAT_AMPLITUDE = float(rng.uniform(*self.AMPLITUDE_RANGE))
        self.PLAT_OMEGA = float(rng.uniform(*self.OMEGA_RANGE))
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
    # THE MARKER, BUILT FROM GEOMETRY
    # ------------------------------------------------------------------

    def _buildMarkerGeometry(self, pos):
        """Build the ArUco marker from black boxes, one per cell.

        PyBullet repeats textures rather than mapping them once, so both
        GEOM_BOX and plane.obj tile the marker. Geometry has no UV mapping.

        generateImageMarker with sidePixels=6 returns the 6x6 cell grid
        directly: the 4x4 payload plus a one-cell black border.
        """
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
                dy = -(first + row * cell)     # image row 0 is +y in world
                bid = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=black,
                    basePosition=[pos[0] + dx, pos[1] + dy, z],
                    physicsClientId=self.CLIENT)
                self.MARKER_CELL_IDS.append((bid, dx, dy))

    def _updatePlatform(self):
        """Create the pad (and marker) if needed, then move them.

        The pad is kinematic: mass 0, so physics never pushes it, but it has a
        collision shape so the drone can rest on it. White, for contrast with
        the black marker cells.
        """
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
        """Camera axes in WORLD coordinates from the drone's attitude.

        Body-fixed, pointing along the drone's -z axis. When the drone tilts
        the camera tilts with it, so marker loss from tilt is emergent rather
        than a threshold.

        Returns (position, right, down, forward) in OpenCV convention.
        """
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
        """Render the downward view. Returns grayscale uint8."""
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
        """Detect the marker and recover platform position relative to the
        drone, in WORLD axes. Returns (rel_pos, True) or (None, False)."""
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
        """The measurement the drone actually receives.

        Velocity is differentiated from successive positions, since neither a
        camera nor a UWB anchor reports velocity. That amplifies noise, which
        is realistic — hence the clip, which also prevents a division blow-up
        when two measurements arrive in the same step.
        """
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        state = self._getDroneStateVector(0)

        if self.SENSING == 'privileged':
            self.steps_total += 1
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
                noise = noise + rng.normal(0, 1.0, 3)   # NLOS spike
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

        return (self.last_meas_pos.copy(), self.last_meas_vel.copy(),
                bool(uwb_valid), bool(aruco_valid))

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """BaseAviary wipes the simulation, so all body ids must be cleared
        and the objects recreated."""
        self.PLATFORM_ID = None
        self.MARKER_CELL_IDS = []
        self.prev_height = None
        self.min_height_seen = None
        self.touchdown_recorded = False
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
    # TOUCHDOWN
    # ------------------------------------------------------------------

    def _checkTouchdown(self):
        """Ground-truth contact detection. IDEMPOTENT — both _computeReward()
        and _computeTerminated() call it.

        A height condition rather than a physics-engine contact query, because
        contact reporting depends on collision-mesh detail and is not
        reproducible across runs.
        """
        if self.touchdown_recorded:
            return True

        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]
        plat_pos = self._getPlatformPos()
        plat_vel = self._getPlatformVel()

        if drone_pos[2] - plat_pos[2] > 0.04:
            return False

        self.touchdown_recorded = True

        horizontal_offset = np.linalg.norm(drone_pos[0:2] - plat_pos[0:2])
        rel_vel = drone_vel - plat_vel

        self.touchdown_offset = float(horizontal_offset)
        self.touchdown_vz = float(abs(rel_vel[2]))
        self.touchdown_lateral = float(np.linalg.norm(rel_vel[0:2]))
        self.touchdown_tilt = float(np.linalg.norm(state[7:9]))
        self.landed_successfully = bool(horizontal_offset < self.PLAT_SIZE)

        return True

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------

    def _observationSpace(self):
        """Parent space plus 6 (privileged) or 8 (abstract/camera).

        The two extra values are validity flags. Without them the policy cannot
        distinguish "the platform is right here" from "I cannot see the
        platform" — a stale held measurement looks identical to a fresh one.
        """
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
    #
    # The reward defines the TASK, and the task is to actually land on the pad
    # — not to believe you have. Rewarding a measured landing would let the
    # policy score points for being fooled by sensor noise.

    def _computeReward(self):
        """Alignment + velocity matching + capped descent progress + descent
        penalty + tilt penalty + terminal bonus.

        PROGRESS is a function of CHANGE, not of state: positive descending,
        negative climbing, exactly zero hovering. State-based rewards create
        places to park; this cannot be farmed by standing still.

        The GATE (progress pays only within 1.5 pad-widths) encodes the
        commit-timing decision, which is what makes landing harder than
        tracking: get over the pad first, then descend.

        THE DESCENT CAP is new. Progress previously paid a flat rate per metre
        closed, so descending twice as fast earned the same total sooner —
        and sooner is worth more under discounting. The reward preferred a
        dive. Now only descent up to DESCENT_SPEED_CAP earns progress, and
        exceeding it costs DESCENT_PENALTY per m/s over.
        """
        # CRITICAL: BaseAviary.step() calls _computeReward() BEFORE
        # _computeTerminated(), where _checkTouchdown() otherwise lives.
        # Without this line the bonus never fires on the touchdown step.
        self._checkTouchdown()

        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]
        plat_pos = self._getPlatformPos()
        plat_vel = self._getPlatformVel()

        horiz_dist = float(np.linalg.norm(drone_pos[0:2] - plat_pos[0:2]))
        height_above = float(drone_pos[2] - plat_pos[2])

        align = np.exp(-3.0 * horiz_dist)

        rel_vel = drone_vel - plat_vel
        vel_match = 0.5 * np.exp(-2.0 * np.linalg.norm(rel_vel[0:2]))

        # --- capped progress, gated on alignment ----------------------
        dt = 1.0 / self.CTRL_FREQ
        progress = 0.0
        descent_pen = 0.0

        if self.prev_height is not None:
            closed = self.prev_height - height_above     # metres, this step
            v_desc = closed / dt                          # m/s, signed

            if horiz_dist < self.PLAT_SIZE * 1.5:
                # Only descent up to the cap earns reward. Climbing (negative
                # v_desc) still costs, which keeps the ratchet meaningful.
                v_useful = min(v_desc, self.DESCENT_SPEED_CAP)
                progress = 30.0 * v_useful * dt

            # The penalty applies wherever the drone is, not only over the
            # pad, so it cannot dive fast while off-centre and then drift in.
            if v_desc > self.DESCENT_SPEED_CAP:
                descent_pen = -self.DESCENT_PENALTY * (v_desc - self.DESCENT_SPEED_CAP) * dt

        self.prev_height = height_above

        tilt = float(np.linalg.norm(state[7:9]))
        tilt_pen = -0.3 * tilt

        bonus = 0.0
        if self.touchdown_recorded and self.landed_successfully:
            softness = np.exp(-3.0 * self.touchdown_vz)
            centred = np.exp(-5.0 * self.touchdown_offset)
            level = np.exp(-3.0 * self.touchdown_tilt)
            bonus = 300.0 + 800.0 * softness * centred * level

        return float(align + vel_match + progress + descent_pen
                     + tilt_pen + bonus)

    # ------------------------------------------------------------------
    # EPISODE END
    # ------------------------------------------------------------------

    def _computeTerminated(self):
        return self._checkTouchdown()

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)

        bound = 1.0 + self.AMPLITUDE_RANGE[1]

        if abs(state[0]) > bound or abs(state[1]) > 1.0 or state[2] > 2.0:
            return True

        if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
            return True

        # Descent ratchet: never more than RATCHET_SLACK above the lowest
        # point already reached. Six reward variants all converged on a stable
        # hover; this removes the option rather than trying to out-price it.
        h = state[2] - self._getPlatformPos()[2]
        if self.min_height_seen is None or h < self.min_height_seen:
            self.min_height_seen = h
        if h > self.min_height_seen + self.RATCHET_SLACK:
            return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        plat_pos = self._getPlatformPos()

        info = {
            "horiz_dist": float(np.linalg.norm(state[0:2] - plat_pos[0:2])),
            "height_above_pad": float(state[2] - plat_pos[2]),
            "touchdown": self.touchdown_recorded,
            "success": self.landed_successfully,
            "plat_amplitude": self.PLAT_AMPLITUDE,
            "plat_omega": self.PLAT_OMEGA,
            "plat_peak_speed": self.PLAT_AMPLITUDE * self.PLAT_OMEGA,
            "plat_peak_accel": self.PLAT_AMPLITUDE * self.PLAT_OMEGA ** 2,
            "sensing": self.SENSING,
            "blind_fraction": (self.steps_blind / self.steps_total
                               if self.steps_total > 0 else 0.0),
            "detect_error_mean": (self.detect_error_sum / self.detect_count
                                  if self.detect_count > 0 else float('nan')),
        }

        if self.touchdown_recorded:
            info.update({
                "touchdown_vz": self.touchdown_vz,
                "touchdown_lateral": self.touchdown_lateral,
                "touchdown_tilt": self.touchdown_tilt,
                "touchdown_offset": self.touchdown_offset,
            })

        return info
