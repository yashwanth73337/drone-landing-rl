"""
LandingAviary
=============

Land a quadrotor on a moving platform.

Built on gym-pybullet-drones (utiasDSL).


WHAT THIS VERSION ADDS: A SENSING MODEL
---------------------------------------
Every previous version handed the policy PRIVILEGED STATE — the exact relative
position and velocity of the platform, read straight out of the simulator,
noiseless and always available. A real drone cannot measure that.

This version models the lab's actual sensing stack:

  UWB (ultra-wideband radio ranging)
      Works at any range and any attitude. But noisy (decimetre-level), slow
      to update (~20 Hz), and occasionally throws an NLOS outlier.

  ArUco fiducial marker + downward camera
      Centimetre accurate. But only works while the marker is inside the
      camera's field of view.

The critical detail is geometric. A downward camera at height h sees a ground
footprint of roughly 2h. A marker of side s therefore leaves the frame at about
h = s. For a 20-50 cm marker, the drone is BLIND FOR THE FINAL 20-60 CM OF
EVERY LANDING — precisely when precision matters most.

The marker is also lost at high tilt, and detection drops frames during fast
motion.

Set `use_sensor_model=False` to recover the old privileged-state behaviour for
comparison. That switch is the ablation.

Expect success to FALL relative to privileged state. The current policy is a
plain MLP with no memory: it sees only the present instant, so when the marker
vanishes it has nothing to fall back on. Recovering that loss with a recurrent
policy is the contribution.


PRIOR BUGS FIXED IN THIS FILE
-----------------------------

1. ORDERING BUG (v6). gym_pybullet_drones.BaseAviary.step() calls
   _computeReward() BEFORE _computeTerminated(), and _checkTouchdown() — which
   sets touchdown_recorded — lived only in the latter. So on the touchdown step
   the reward function saw touchdown_recorded == False, skipped the bonus, and
   the episode then ended. The landing bonus was NEVER paid, at any value.
   Eight reward variants failed for this one reason. _computeReward() now calls
   _checkTouchdown() itself.

2. PHASE MEMORISATION (v7). With fixed amplitude and frequency, and every
   episode starting at t=0, the platform's position was fully predictable from
   the step counter. The policy memorised a timed routine instead of using its
   observation, and scored 100% while having learned nothing transferable. A
   frequency sweep exposed it: success was NON-MONOTONIC, failing at 0.2 rad/s,
   succeeding at 0.5, failing at 0.8 and 1.2, succeeding again at 1.8. Motion
   parameters are now resampled every episode.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class LandingAviary(BaseRLAviary):
    """Single agent RL problem: land on a moving platform, with modelled sensing."""

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
                 # --- sensing model -----------------------------------
                 use_sensor_model: bool = True,
                 uwb_noise_std: float = 0.15,      # metres, per axis
                 uwb_rate_hz: float = 20.0,        # update rate
                 uwb_outlier_prob: float = 0.02,   # NLOS spikes
                 aruco_noise_std: float = 0.02,    # metres, per axis
                 aruco_fov_height: float = 0.30,   # below this, marker leaves frame
                 aruco_tilt_limit: float = 0.50,   # rad, marker lost past this
                 aruco_dropout_prob: float = 0.10, # random frame drops
                 ):

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

        # --- sensing parameters ---------------------------------------
        self.USE_SENSOR_MODEL = use_sensor_model
        self.UWB_NOISE_STD = uwb_noise_std
        self.UWB_RATE_HZ = uwb_rate_hz
        self.UWB_OUTLIER_PROB = uwb_outlier_prob
        self.ARUCO_NOISE_STD = aruco_noise_std
        self.ARUCO_FOV_HEIGHT = aruco_fov_height
        self.ARUCO_TILT_LIMIT = aruco_tilt_limit
        self.ARUCO_DROPOUT_PROB = aruco_dropout_prob

        # Last measurement the drone actually received. Held between UWB
        # updates and used when nothing is currently visible.
        self.last_meas_pos = np.zeros(3)
        self.last_meas_vel = np.zeros(3)
        self.last_uwb_step = -1

        # Diagnostics
        self.steps_blind = 0
        self.steps_total = 0

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
    # THE PLATFORM (ground truth — used for physics, reward, and metrics,
    # but NOT given directly to the policy when the sensor model is on)
    # ------------------------------------------------------------------

    def _samplePlatformMotion(self):
        """Draw new motion parameters for this episode.

        Phase matters most. Without it, every episode begins with the platform
        at x = 0 moving in +x, which is itself a memorisable cue.
        """
        if not self.RANDOMIZE_PLATFORM:
            return
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        self.PLAT_AMPLITUDE = float(rng.uniform(*self.AMPLITUDE_RANGE))
        self.PLAT_OMEGA = float(rng.uniform(*self.OMEGA_RANGE))
        self.PLAT_PHASE = float(rng.uniform(0.0, 2.0 * np.pi))

    def _getPlatformPos(self):
        """TRUE centre of the pad surface. Ground truth."""
        t = self.step_counter / self.PYB_FREQ
        x = self.PLAT_AMPLITUDE * np.sin(self.PLAT_OMEGA * t + self.PLAT_PHASE)
        return np.array([x, 0.0, self.PLAT_HEIGHT])

    def _getPlatformVel(self):
        """TRUE platform velocity. Ground truth."""
        t = self.step_counter / self.PYB_FREQ
        vx = (self.PLAT_AMPLITUDE * self.PLAT_OMEGA
              * np.cos(self.PLAT_OMEGA * t + self.PLAT_PHASE))
        return np.array([vx, 0.0, 0.0])

    def _updatePlatform(self):
        """Create the pad if needed, then teleport it along its trajectory.

        Mass 0 makes it kinematic: physics never pushes it, but it has a
        collision shape so the drone can rest on it.
        """
        pos = self._getPlatformPos()

        if self.PLATFORM_ID is None:
            half = [self.PLAT_SIZE, self.PLAT_SIZE, self.PLAT_HEIGHT / 2]
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=self.CLIENT)
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half,
                rgbaColor=[0.9, 0.2, 0.2, 1.0],
                physicsClientId=self.CLIENT)
            self.PLATFORM_ID = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[pos[0], pos[1], self.PLAT_HEIGHT / 2],
                physicsClientId=self.CLIENT
            )
        else:
            p.resetBasePositionAndOrientation(
                self.PLATFORM_ID,
                [pos[0], pos[1], self.PLAT_HEIGHT / 2],
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT
            )

    # ------------------------------------------------------------------
    # THE SENSING MODEL
    # ------------------------------------------------------------------

    def _senseRelativeState(self, true_rel_pos, true_rel_vel, drone_state):
        """Turn ground truth into what the drone would actually measure.

        Returns (measured_pos, measured_vel, uwb_valid, aruco_valid).

        ARUCO — accurate but conditional
            Available only when ALL of:
              - the drone is high enough that the marker is still in frame
              - the drone is not tilted too far
              - the detector did not drop this frame

            The height condition is the important one. A downward camera at
            height h sees a footprint of about 2h, so a marker of side s leaves
            the frame at roughly h = s. The drone goes blind for the last
            ARUCO_FOV_HEIGHT metres of every descent.

        UWB — always available, but coarse
            Decimetre noise, ~20 Hz updates (so the same reading is held for
            several control steps), and occasional NLOS outliers of 0.5-2 m.

        FUSION — deliberately naive
            When ArUco is available it is used, because it is far more
            accurate. Otherwise UWB. When neither updated this step, the last
            received measurement is held.

            No Kalman filter. That is intentional: Shin et al. (RA-L 2026)
            showed an EKF DIVERGES during marker dropout because it falls back
            on a constant-velocity model, whereas a learned temporal estimator
            degrades gracefully. Handing the policy raw measurements plus
            validity flags leaves that estimation problem for the network,
            which is exactly what the recurrent policy is meant to solve.
        """
        rng = self.np_random if hasattr(self, 'np_random') else np.random

        height_above = drone_state[2] - self._getPlatformPos()[2]
        tilt = np.linalg.norm(drone_state[7:9])

        # --- ArUco availability ---------------------------------------
        aruco_in_frame = height_above > self.ARUCO_FOV_HEIGHT
        aruco_level_enough = tilt < self.ARUCO_TILT_LIMIT
        aruco_frame_ok = rng.random() > self.ARUCO_DROPOUT_PROB
        aruco_valid = bool(aruco_in_frame and aruco_level_enough and aruco_frame_ok)

        # --- UWB availability (rate-limited) --------------------------
        steps_between_uwb = max(1, int(self.CTRL_FREQ / self.UWB_RATE_HZ))
        uwb_valid = (self.step_counter - self.last_uwb_step) >= steps_between_uwb

        # --- take a measurement ---------------------------------------
        if aruco_valid:
            noise = rng.normal(0, self.ARUCO_NOISE_STD, 3)
            self.last_meas_pos = true_rel_pos + noise
            self.last_meas_vel = true_rel_vel + rng.normal(0, self.ARUCO_NOISE_STD * 2, 3)
            self.last_uwb_step = self.step_counter

        elif uwb_valid:
            noise = rng.normal(0, self.UWB_NOISE_STD, 3)
            if rng.random() < self.UWB_OUTLIER_PROB:
                # NLOS spike: a large, wrong reading
                noise = noise + rng.normal(0, 1.0, 3)
            self.last_meas_pos = true_rel_pos + noise
            self.last_meas_vel = true_rel_vel + rng.normal(0, self.UWB_NOISE_STD * 3, 3)
            self.last_uwb_step = self.step_counter

        # else: no new information this step. Hold the last measurement.

        if not aruco_valid:
            self.steps_blind += 1
        self.steps_total += 1

        return (self.last_meas_pos.copy(), self.last_meas_vel.copy(),
                uwb_valid, aruco_valid)

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self.PLATFORM_ID = None
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
        self.last_uwb_step = -1
        self.steps_blind = 0
        self.steps_total = 0

        out = super().reset(seed=seed, options=options)

        # Sample AFTER super().reset(), where np_random is seeded.
        self._samplePlatformMotion()
        obs = self._computeObs()
        return obs, out[1]

    # ------------------------------------------------------------------
    # TOUCHDOWN DETECTION (uses ground truth — this is measurement, not
    # something the drone is told)
    # ------------------------------------------------------------------

    def _checkTouchdown(self):
        """Has the drone reached the pad surface? If so, record how.

        IDEMPOTENT — safe to call more than once per step. Both
        _computeReward() and _computeTerminated() call it.
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
        """Parent space plus 8 when the sensor model is on, 6 when off.

        With the sensor model:
            3  measured relative position
            3  measured relative velocity
            1  uwb_valid   (0 or 1)
            1  aruco_valid (0 or 1)

        The two flags matter. Without them the policy cannot distinguish
        "the platform is right here" from "I cannot see the platform" — a
        stale measurement looks identical to a fresh one.
        """
        base = super()._observationSpace()
        n_extra = 8 if self.USE_SENSOR_MODEL else 6
        lo = np.full((base.shape[0], n_extra), -np.inf, dtype=np.float32)
        hi = np.full((base.shape[0], n_extra), np.inf, dtype=np.float32)
        return spaces.Box(
            low=np.hstack([base.low, lo]).astype(np.float32),
            high=np.hstack([base.high, hi]).astype(np.float32),
            dtype=np.float32
        )

    def _computeObs(self):
        """Parent observation, plus what the drone can sense of the platform."""
        self._updatePlatform()

        obs = super()._computeObs()
        state = self._getDroneStateVector(0)

        true_rel_pos = self._getPlatformPos() - state[0:3]
        true_rel_vel = self._getPlatformVel() - state[10:13]

        if self.USE_SENSOR_MODEL:
            meas_pos, meas_vel, uwb_valid, aruco_valid = self._senseRelativeState(
                true_rel_pos, true_rel_vel, state)
            extra = np.hstack([
                meas_pos, meas_vel,
                float(uwb_valid), float(aruco_valid)
            ]).reshape(1, 8)
        else:
            extra = np.hstack([true_rel_pos, true_rel_vel]).reshape(1, 6)

        return np.hstack([obs, extra]).astype('float32')

    # ------------------------------------------------------------------
    # REWARD (computed from GROUND TRUTH)
    # ------------------------------------------------------------------
    #
    # The reward uses true positions, not measured ones. That is correct: the
    # reward defines the TASK, and the task is to actually land on the pad,
    # not to think you have. Rewarding a measured landing would let the policy
    # score points for being fooled by sensor noise.
    #
    # The policy never sees the reward's inputs — only its value.

    def _computeReward(self):
        """Approach, align, match velocity, descend, touch down softly.

        PROGRESS — 30 * (previous height - current height), gated on alignment
            A function of CHANGE, not of state. Zero when hovering, so it
            cannot be farmed by standing still. Gated within 1.5 pad-widths:
            the drone must get over the pad FIRST, then descend. That gate is
            the commit-timing decision, and it is what makes landing harder
            than tracking.

        ALIGNMENT — exp(-3 * horizontal distance)
        VELOCITY MATCHING — 0.5 * exp(-2 * lateral relative speed)
        TILT PENALTY — -0.3 * tilt
        TOUCHDOWN BONUS — 300 flat, plus up to 800 scaled by landing quality
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

        if self.prev_height is None:
            progress = 0.0
        elif horiz_dist < self.PLAT_SIZE * 1.5:
            progress = 30.0 * (self.prev_height - height_above)
        else:
            progress = 0.0
        self.prev_height = height_above

        tilt = float(np.linalg.norm(state[7:9]))
        tilt_pen = -0.3 * tilt

        bonus = 0.0
        if self.touchdown_recorded and self.landed_successfully:
            softness = np.exp(-3.0 * self.touchdown_vz)
            centred = np.exp(-5.0 * self.touchdown_offset)
            level = np.exp(-3.0 * self.touchdown_tilt)
            bonus = 300.0 + 800.0 * softness * centred * level

        return float(align + vel_match + progress + tilt_pen + bonus)

    # ------------------------------------------------------------------
    # EPISODE END CONDITIONS
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
        # point already reached. Hovering earns zero progress and times out.
        h = state[2] - self._getPlatformPos()[2]
        if self.min_height_seen is None or h < self.min_height_seen:
            self.min_height_seen = h
        if h > self.min_height_seen + self.RATCHET_SLACK:
            return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    def _computeInfo(self):
        """Per-step diagnostics, touchdown metrics, and sensing statistics."""
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
            "blind_fraction": (self.steps_blind / self.steps_total
                               if self.steps_total > 0 else 0.0),
        }

        if self.touchdown_recorded:
            info.update({
                "touchdown_vz": self.touchdown_vz,
                "touchdown_lateral": self.touchdown_lateral,
                "touchdown_tilt": self.touchdown_tilt,
                "touchdown_offset": self.touchdown_offset,
            })

        return info
