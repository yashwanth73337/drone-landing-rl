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


THE BUG THAT CAUSED EIGHT FAILED REWARD VARIANTS
------------------------------------------------
gym_pybullet_drones.BaseAviary.step() evaluates in this order:

    obs        = self._computeObs()
    reward     = self._computeReward()        <-- FIRST
    terminated = self._computeTerminated()    <-- calls _checkTouchdown()
    truncated  = self._computeTruncated()
    info       = self._computeInfo()

_checkTouchdown() is what sets self.touchdown_recorded = True. But it runs
INSIDE _computeTerminated(), which is called AFTER _computeReward().

So on the step where the drone actually reaches the pad, _computeReward()
still sees touchdown_recorded == False, skips the bonus block entirely, and
returns a normal shaping reward. The episode then terminates, so there is no
next step in which the bonus could be paid.

The landing bonus was NEVER awarded. Not once, at any value.

The agent therefore had no evidence that landing was worth anything, and
correctly learned to hover instead. Eight reward variants (bonus 100 -> 800,
time penalties, descent ratchets) all failed for this single reason.

FIX: _computeReward() now calls _checkTouchdown() itself, before evaluating
the bonus. _checkTouchdown() is idempotent — it returns early if already
recorded — so calling it from both places is safe.


Reward design history
---------------------
v1  bonus 100, time penalty -0.5   -> hovered. (bonus never paid)
v2  bonus 600, descent x3          -> hovered. (bonus never paid)
v3  time penalty -2.0              -> self-terminated in 4 steps.
v4  progress term, bonus 200       -> hovered at 0.26 m, centred to 5 mm.
v4b bonus 800, weights cut         -> hovered at 0.28 m, tracking degraded.
v5  descent ratchet 0.15           -> oscillated 0.17 <-> 0.32 m.
v5b ratchet 0.05, flat +300        -> still hovered.
v6  (this version) ORDERING BUG FIXED.


Difficulty settings
-------------------
The defaults below are EASY MODE, to establish that landing is learnable at
all before adding difficulty back:

    platform_amplitude = 0.0   (stationary platform)
    platform_size      = 0.50  (1 m wide pad)
    start_height       = 0.4   (only 30 cm to descend)

Once this lands reliably, walk the difficulty back up ONE parameter at a
time, retraining from the previous policy each time:

    1. start_height       0.4  -> 0.8
    2. platform_size      0.50 -> 0.20
    3. platform_amplitude 0.0  -> 0.5    (platform starts moving)

That staged progression is the curriculum. Goldschmid & Ahmad's ablation
found sequential curriculum with transfer reached 99% success in 118 min,
while training on the full task directly reached 57% in 520 min.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class LandingAviary(BaseRLAviary):
    """Single agent RL problem: land on a (possibly moving) platform."""

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
                 platform_amplitude: float = 0.0,   # 0.0 = stationary. Raise to 0.5 later.
                 platform_omega: float = 0.5,
                 platform_size: float = 0.50,       # half-width. 0.50 = 1 m pad. Shrink to 0.20 later.
                 platform_height: float = 0.10,
                 start_height: float = 0.4,         # raise to 0.8 later
                 ratchet_slack: float = 0.10,
                 ):

        self.PLAT_AMPLITUDE = platform_amplitude
        self.PLAT_OMEGA = platform_omega
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

    def _getPlatformPos(self):
        """Centre of the pad SURFACE right now."""
        t = self.step_counter / self.PYB_FREQ
        x = self.PLAT_AMPLITUDE * np.sin(self.PLAT_OMEGA * t)
        return np.array([x, 0.0, self.PLAT_HEIGHT])

    def _getPlatformVel(self):
        """Platform velocity right now."""
        t = self.step_counter / self.PYB_FREQ
        vx = self.PLAT_AMPLITUDE * self.PLAT_OMEGA * np.cos(self.PLAT_OMEGA * t)
        return np.array([vx, 0.0, 0.0])

    def _updatePlatform(self):
        """Create the pad if needed, then teleport it along the sine wave.

        Mass 0 makes it KINEMATIC: physics never pushes it, but it still has
        a collision shape so the drone can rest on it. Created even without
        the GUI, because the drone must be able to physically touch it.
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
        """BaseAviary wipes the simulation, so clear the pad id, the ratchet,
        and all touchdown records."""
        self.PLATFORM_ID = None
        self.prev_height = None
        self.min_height_seen = None
        self.touchdown_recorded = False
        self.touchdown_vz = None
        self.touchdown_lateral = None
        self.touchdown_tilt = None
        self.touchdown_offset = None
        self.landed_successfully = False
        return super().reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    # TOUCHDOWN DETECTION
    # ------------------------------------------------------------------

    def _checkTouchdown(self):
        """Has the drone reached the pad surface? If so, record how.

        IDEMPOTENT — safe to call multiple times per step. Returns early once
        touchdown has already been recorded. This matters because both
        _computeReward() and _computeTerminated() now call it.

        Touchdown is defined as a CONDITION on height rather than waiting for
        the physics engine to report contact. Contact detection depends on
        collision-mesh detail and is noisy; a height threshold is reproducible
        and easy to justify in a report.
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
            1.5 pad-widths. This is the commit-timing decision, and it is
            what makes landing harder than tracking — the drone must get
            over the pad FIRST, then come down.

        ALIGNMENT — exp(-3 * horizontal distance)

            Get over the pad. Horizontal only, because vertical distance is
            what we WANT the drone to close.

        VELOCITY MATCHING — 0.5 * exp(-2 * |lateral relative velocity|)

            Move WITH the platform. Right place at the wrong speed means a
            sideways hit and a tumble on contact.

        TILT PENALTY — -0.3 * tilt

            A drone that touches down tilted catches a leg and flips.

        TOUCHDOWN BONUS — 300 flat, plus up to 800 scaled by quality

            The flat 300 pays for REACHING the pad at all. Without it, a
            mediocre landing scores less than continued hovering, so the
            policy never experiences a good outcome and cannot learn that
            landing is worthwhile.

            The scaled part is softness x centred x level, so landing WELL
            is still worth far more than landing badly.

        Returns
        -------
        float
        """
        # CRITICAL: BaseAviary.step() calls _computeReward() BEFORE
        # _computeTerminated(), and _checkTouchdown() lives in the latter.
        # Without this line the bonus block below never fires on the
        # touchdown step, and since the episode then ends, the bonus is
        # never paid at all. This single ordering issue caused eight
        # consecutive reward variants to fail.
        self._checkTouchdown()

        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]
        plat_pos = self._getPlatformPos()
        plat_vel = self._getPlatformVel()

        horiz_dist = float(np.linalg.norm(drone_pos[0:2] - plat_pos[0:2]))
        height_above = float(drone_pos[2] - plat_pos[2])

        # --- alignment ------------------------------------------------
        align = np.exp(-3.0 * horiz_dist)

        # --- velocity matching ----------------------------------------
        rel_vel = drone_vel - plat_vel
        vel_match = 0.5 * np.exp(-2.0 * np.linalg.norm(rel_vel[0:2]))

        # --- progress, gated on alignment -----------------------------
        if self.prev_height is None:
            progress = 0.0
        elif horiz_dist < self.PLAT_SIZE * 1.5:
            progress = 30.0 * (self.prev_height - height_above)
        else:
            progress = 0.0
        self.prev_height = height_above

        # --- tilt penalty ---------------------------------------------
        tilt = float(np.linalg.norm(state[7:9]))
        tilt_pen = -0.3 * tilt

        # --- terminal bonus -------------------------------------------
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

        bound = 1.0 + self.PLAT_AMPLITUDE

        # flew out of the arena
        if abs(state[0]) > bound or abs(state[1]) > 1.0 or state[2] > 2.0:
            return True

        # tilted past ~23 degrees
        if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
            return True

        # --- DESCENT RATCHET ------------------------------------------
        #
        # The drone may never be more than RATCHET_SLACK above the lowest
        # point it has already reached. Reach 0.5 m and the ceiling becomes
        # 0.60 m; reach 0.3 m and it becomes 0.40 m.
        #
        # Hovering is still legal, but it earns zero progress reward and
        # times out with a low score, while descending reaches the bonus.
        h = state[2] - self._getPlatformPos()[2]
        if self.min_height_seen is None or h < self.min_height_seen:
            self.min_height_seen = h
        if h > self.min_height_seen + self.RATCHET_SLACK:
            return True

        # ran out of time
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    def _computeInfo(self):
        """Per-step diagnostics, plus touchdown metrics once contact happens."""
        state = self._getDroneStateVector(0)
        plat_pos = self._getPlatformPos()

        info = {
            "horiz_dist": float(np.linalg.norm(state[0:2] - plat_pos[0:2])),
            "height_above_pad": float(state[2] - plat_pos[2]),
            "touchdown": self.touchdown_recorded,
            "success": self.landed_successfully,
        }

        if self.touchdown_recorded:
            info.update({
                "touchdown_vz": self.touchdown_vz,
                "touchdown_lateral": self.touchdown_lateral,
                "touchdown_tilt": self.touchdown_tilt,
                "touchdown_offset": self.touchdown_offset,
            })

        return info