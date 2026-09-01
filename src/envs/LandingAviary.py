"""
LandingAviary
=============

Land a quadrotor on a moving platform.

Built on gym-pybullet-drones (utiasDSL).

Touchdown metrics
-----------------
Goldschmid & Ahmad (2024), TornadoDrone (2024), and Shin et al. (2026) all
report binary success rates and nothing about HOW HARD the drone hit. This
environment records vertical speed, lateral relative speed, tilt, and offset
at the moment of contact. Those are the headline metrics of this thesis.


TWO BUGS THAT SHAPED THIS FILE
------------------------------

1. THE ORDERING BUG (fixed in v6)

   gym_pybullet_drones.BaseAviary.step() evaluates in this order:

       obs        = self._computeObs()
       reward     = self._computeReward()        <-- FIRST
       terminated = self._computeTerminated()    <-- calls _checkTouchdown()

   _checkTouchdown() sets self.touchdown_recorded = True, but it only ran
   inside _computeTerminated(), which is called AFTER _computeReward().

   So on the step where the drone reached the pad, _computeReward() still saw
   touchdown_recorded == False, skipped the bonus block, and returned an
   ordinary shaping reward. The episode then ended, so there was no later step
   in which the bonus could be paid.

   The landing bonus was NEVER awarded, at any value. Eight consecutive reward
   variants failed for this single reason. Fix: _computeReward() now calls
   _checkTouchdown() itself, before evaluating the bonus. The method is
   idempotent, so calling it from both places is safe.

2. PHASE MEMORISATION (fixed in v7, this version)

   With fixed A and omega, and every episode starting at t = 0, the platform
   is at a fully predictable position at every timestep. Nothing forces the
   policy to use its relative-position observation — memorising a timed
   routine ("at step 60, move here") scores just as well and is easier to
   learn.

   That is exactly what happened. A sweep over platform frequency produced
   NON-MONOTONIC success:

       omega   accel      success
       0.200   0.050        0%      <- GENTLER than trained, still fails
       0.500   0.125      100%      <- trained value
       0.800   0.200        0%
       1.200   0.300        0%
       1.800   0.450      100%      <- near a harmonic of the trained rhythm
       2.800   0.700        0%

   A physical limit produces a clean threshold. Success that comes and goes
   with frequency is a timing artefact, not a capability limit. The 100%
   success reported earlier was measuring memorisation, not tracking.

   Fix: platform motion parameters are RESAMPLED EVERY EPISODE. Amplitude,
   frequency, and phase offset are all randomised, so no fixed schedule can
   succeed and the only way to score is to actually use the observation.

   Expect success rate to DROP relative to the fixed-motion version. That drop
   is the honest number. Shin et al. (RA-L 2026) randomise platform motion
   during training for precisely this reason.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class LandingAviary(BaseRLAviary):
    """Single agent RL problem: land on a moving platform."""

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
                 # Ranges, resampled every episode. Set randomize_platform
                 # to False and pass fixed values to evaluate one setting.
                 randomize_platform: bool = True,
                 amplitude_range=(0.2, 1.0),
                 omega_range=(0.2, 1.5),
                 platform_amplitude: float = 0.5,   # used when randomize=False
                 platform_omega: float = 0.5,       # used when randomize=False
                 platform_size: float = 0.20,
                 platform_height: float = 0.10,
                 start_height: float = 0.8,
                 ratchet_slack: float = 0.10,
                 ):

        self.RANDOMIZE_PLATFORM = randomize_platform
        self.AMPLITUDE_RANGE = amplitude_range
        self.OMEGA_RANGE = omega_range

        # Current episode's values. Overwritten each reset when randomising.
        self.PLAT_AMPLITUDE = platform_amplitude
        self.PLAT_OMEGA = platform_omega
        self.PLAT_PHASE = 0.0

        self.PLAT_SIZE = platform_size
        self.PLAT_HEIGHT = platform_height
        self.START_HEIGHT = start_height
        self.RATCHET_SLACK = ratchet_slack

        self.EPISODE_LEN_SEC = 10

        self.PLATFORM_ID = None

        # Height above the pad on the previous step, for the progress term.
        self.prev_height = None

        # Lowest height above the pad reached so far, for the ratchet.
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
        """Draw new motion parameters for this episode.

        Randomising AMPLITUDE, FREQUENCY, and PHASE together means:

          - amplitude   -> the platform travels a different distance
          - frequency   -> it reverses at a different rate
          - phase       -> it is somewhere different at t = 0, and moving in
                           a different direction

        Phase matters most. Without it, every episode begins with the platform
        at x = 0 moving in the +x direction, which is itself a memorisable cue.
        """
        if not self.RANDOMIZE_PLATFORM:
            return

        rng = self.np_random if hasattr(self, 'np_random') else np.random

        self.PLAT_AMPLITUDE = float(rng.uniform(*self.AMPLITUDE_RANGE))
        self.PLAT_OMEGA = float(rng.uniform(*self.OMEGA_RANGE))
        self.PLAT_PHASE = float(rng.uniform(0.0, 2.0 * np.pi))

    def _getPlatformPos(self):
        """Centre of the pad SURFACE right now."""
        t = self.step_counter / self.PYB_FREQ
        x = self.PLAT_AMPLITUDE * np.sin(self.PLAT_OMEGA * t + self.PLAT_PHASE)
        return np.array([x, 0.0, self.PLAT_HEIGHT])

    def _getPlatformVel(self):
        """Platform velocity right now."""
        t = self.step_counter / self.PYB_FREQ
        vx = (self.PLAT_AMPLITUDE * self.PLAT_OMEGA
              * np.cos(self.PLAT_OMEGA * t + self.PLAT_PHASE))
        return np.array([vx, 0.0, 0.0])

    def _updatePlatform(self):
        """Create the pad if needed, then teleport it along its trajectory.

        Mass 0 makes it KINEMATIC: physics never pushes it, but it still has a
        collision shape so the drone can rest on it. Created even without the
        GUI, because the drone must be able to physically touch it.
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

    def reset(self, seed=None, options=None):
        """Reset the episode.

        BaseAviary wipes the PyBullet simulation, so the pad id must be
        cleared. New platform motion is sampled here.
        """
        self.PLATFORM_ID = None
        self.prev_height = None
        self.min_height_seen = None
        self.touchdown_recorded = False
        self.touchdown_vz = None
        self.touchdown_lateral = None
        self.touchdown_tilt = None
        self.touchdown_offset = None
        self.landed_successfully = False

        out = super().reset(seed=seed, options=options)

        # Sample AFTER super().reset(), because that is where np_random is
        # seeded. Then rebuild the observation so it reflects the new motion.
        self._samplePlatformMotion()
        obs = self._computeObs()
        return obs, out[1]

    # ------------------------------------------------------------------
    # TOUCHDOWN DETECTION
    # ------------------------------------------------------------------

    def _checkTouchdown(self):
        """Has the drone reached the pad surface? If so, record how.

        IDEMPOTENT — safe to call more than once per step. Both
        _computeReward() and _computeTerminated() call it.

        Touchdown is defined as a CONDITION on height rather than waiting for
        the physics engine to report contact. Contact detection depends on
        collision-mesh detail and is not reproducible; a height threshold is.
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
        """Parent space plus 6: relative platform position and velocity."""
        base = super()._observationSpace()
        lo = np.full((base.shape[0], 6), -np.inf, dtype=np.float32)
        hi = np.full((base.shape[0], 6), np.inf, dtype=np.float32)
        return spaces.Box(
            low=np.hstack([base.low, lo]).astype(np.float32),
            high=np.hstack([base.high, hi]).astype(np.float32),
            dtype=np.float32
        )

    def _computeObs(self):
        """Parent observation, plus platform position and velocity RELATIVE
        to the drone.

        Relative, because that is what real sensing gives you. UWB reports a
        range to the platform; an ArUco marker reports a pose relative to the
        camera. The drone never receives world coordinates.

        With randomised motion these six numbers are the ONLY way to locate
        the platform. That is the point: it forces the policy to be a feedback
        controller rather than a timed routine.
        """
        self._updatePlatform()

        obs = super()._computeObs()
        state = self._getDroneStateVector(0)

        rel_pos = self._getPlatformPos() - state[0:3]
        rel_vel = self._getPlatformVel() - state[10:13]

        extra = np.hstack([rel_pos, rel_vel]).reshape(1, 6)
        return np.hstack([obs, extra]).astype('float32')

    # ------------------------------------------------------------------
    # REWARD
    # ------------------------------------------------------------------

    def _computeReward(self):
        """Approach, align, match velocity, descend, touch down softly.

        PROGRESS — 30 * (previous height - current height), gated

            A function of CHANGE, not of state. Positive when descending,
            negative when climbing, exactly zero when hovering. State-based
            rewards create places to park; progress rewards cannot be farmed
            by standing still.

            Gated on horizontal alignment: progress only pays while within
            1.5 pad-widths. This is the commit-timing decision, and it is what
            makes landing harder than tracking — the drone must get over the
            pad FIRST, then come down.

        ALIGNMENT — exp(-3 * horizontal distance)

            Get over the pad. Horizontal only, because vertical distance is
            what we WANT the drone to close.

        VELOCITY MATCHING — 0.5 * exp(-2 * |lateral relative velocity|)

            Move WITH the platform. Right place at the wrong speed means a
            sideways hit and a tumble on contact.

        TILT PENALTY — -0.3 * tilt

            A drone that touches down tilted catches a leg and flips.

        TOUCHDOWN BONUS — 300 flat, plus up to 800 scaled by quality

            The flat 300 pays for REACHING the pad at all. Without it a
            mediocre landing scores less than continued hovering, so the policy
            never experiences a good outcome and cannot learn that landing is
            worthwhile.

        Returns
        -------
        float
        """
        # CRITICAL: BaseAviary.step() calls _computeReward() BEFORE
        # _computeTerminated(), and _checkTouchdown() lives in the latter.
        # Without this line the bonus below never fires on the touchdown step,
        # and the episode then ends, so the bonus is never paid at all.
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
        """Episode ends on touchdown, successful or not."""
        return self._checkTouchdown()

    def _computeTruncated(self):
        """Failure, or out of time."""
        state = self._getDroneStateVector(0)

        # Arena scales with the largest amplitude the platform can be given,
        # so the bound does not change between episodes.
        bound = 1.0 + self.AMPLITUDE_RANGE[1]

        if abs(state[0]) > bound or abs(state[1]) > 1.0 or state[2] > 2.0:
            return True

        # tilted past ~23 degrees
        if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
            return True

        # --- DESCENT RATCHET ------------------------------------------
        #
        # The drone may never be more than RATCHET_SLACK above the lowest
        # point it has already reached. Hovering is still legal, but it earns
        # zero progress reward and times out with a low score, while
        # descending reaches the bonus.
        h = state[2] - self._getPlatformPos()[2]
        if self.min_height_seen is None or h < self.min_height_seen:
            self.min_height_seen = h
        if h > self.min_height_seen + self.RATCHET_SLACK:
            return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    def _computeInfo(self):
        """Per-step diagnostics, plus touchdown metrics once contact happens.

        Platform parameters are included so evaluation can be broken down by
        motion difficulty — with randomised motion, every episode is a
        different test case.
        """
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
        }

        if self.touchdown_recorded:
            info.update({
                "touchdown_vz": self.touchdown_vz,
                "touchdown_lateral": self.touchdown_lateral,
                "touchdown_tilt": self.touchdown_tilt,
                "touchdown_offset": self.touchdown_offset,
            })

        return info
