"""
TrackingAviary
==============

First custom environment for the MTP project.

Task: the drone must follow a target point that slides back and forth
horizontally, while holding altitude.

This is HoverAviary with one change: the target moves. That makes it a
tracking problem instead of a hover problem — which is landing without
the descent.

Built on gym-pybullet-drones (utiasDSL).

Reward history (kept as a record of what was tried and why)
-----------------------------------------------------------
v1  max(0, 2 - dist**4)          -> too flat. 0.618 m error while
                                    collecting 92% of max reward.
v2  exp(-2.0 * dist)             -> too sharp. Near-zero reward where
                                    learning starts, so the policy gave
                                    up and sat on the ground.
v3  exp(-0.7 * dist)             -> still sat on the ground: 0.50/step
                                    for doing nothing beat the risk of
                                    trying to fly.
v4  + unconditional alive bonus  -> flew, but drifted away collecting the
                                    bonus. 0.840 m error.
v5  + conditional alive bonus    -> 0.388 m. Tracks, then drifts out at
                                    ~5.4 s because nothing penalises speed.
v6  + velocity matching          -> this version.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class TrackingAviary(BaseRLAviary):
    """Single agent RL problem: follow a horizontally moving target."""

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
                 target_amplitude: float = 0.5,   # metres the target swings either side of centre
                 target_omega: float = 0.5,       # radians/sec — how fast it swings
                 target_height: float = 1.0,      # metres — target altitude, held constant
                 ):

        # --- target motion parameters ---------------------------------
        # Set BEFORE super().__init__() because the parent calls
        # _computeObs() during setup, which needs them to exist.
        self.TARGET_AMPLITUDE = target_amplitude
        self.TARGET_OMEGA = target_omega
        self.TARGET_HEIGHT = target_height

        self.EPISODE_LEN_SEC = 8

        # PyBullet id of the red marker sphere. None means "not created yet".
        self.TARGET_MARKER_ID = None

        # --- START THE DRONE IN THE AIR -------------------------------
        #
        # Taking off is a DIFFERENT SKILL from tracking. If the drone has
        # to learn both at once, it never gets far enough to see any
        # tracking reward, and the safest policy becomes "sit on the
        # ground and don't crash". That is exactly what happened in v2
        # and v3 above.
        #
        # On real hardware ArduPilot handles takeoff anyway — the RL
        # policy only takes over once the drone is already flying.
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, self.TARGET_HEIGHT]])

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
    # THE TARGET
    # ------------------------------------------------------------------

    def _getTargetPos(self):
        """Where the target is right now.

        This replaces HoverAviary's fixed self.TARGET_POS.
        The target slides back and forth along x as a sine wave.

        Returns
        -------
        ndarray
            (3,)-shaped array: target [x, y, z] in world coordinates.
        """
        t = self.step_counter / self.PYB_FREQ          # simulated seconds elapsed
        x = self.TARGET_AMPLITUDE * np.sin(self.TARGET_OMEGA * t)
        return np.array([x, 0.0, self.TARGET_HEIGHT])

    def _getTargetVel(self):
        """How fast the target is moving right now.

        The time derivative of _getTargetPos(). Matching this velocity is
        what makes a landing gentle rather than a collision.

        Returns
        -------
        ndarray
            (3,)-shaped array: target velocity [vx, vy, vz] in m/s.
        """
        t = self.step_counter / self.PYB_FREQ
        vx = self.TARGET_AMPLITUDE * self.TARGET_OMEGA * np.cos(self.TARGET_OMEGA * t)
        return np.array([vx, 0.0, 0.0])

    # ------------------------------------------------------------------
    # DRAWING THE TARGET
    # ------------------------------------------------------------------
    #
    # The target is pure maths — there is no object in the physics
    # simulation. That makes it invisible, which makes debugging hard.
    #
    # So when the GUI is on, we draw a small red sphere at the target
    # position and move it every step. It has NO mass and NO collision
    # shape, so it cannot affect the drone. It is decoration only.
    #
    # Skipped entirely when gui=False, so training stays fast.

    def _updateTargetMarker(self):
        """Create the marker if needed, then move it to the target position."""
        if not self.GUI:
            return

        pos = self._getTargetPos()

        if self.TARGET_MARKER_ID is None:
            visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.04,
                rgbaColor=[1, 0, 0, 0.8],      # red, slightly transparent
                physicsClientId=self.CLIENT
            )
            self.TARGET_MARKER_ID = p.createMultiBody(
                baseMass=0,                     # massless — gravity ignores it
                baseCollisionShapeIndex=-1,     # no collision — drone flies through it
                baseVisualShapeIndex=visual,
                basePosition=pos,
                physicsClientId=self.CLIENT
            )
        else:
            p.resetBasePositionAndOrientation(
                self.TARGET_MARKER_ID,
                pos,
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT
            )

    def reset(self, seed=None, options=None):
        """Reset the episode.

        BaseAviary.reset() wipes the whole PyBullet simulation, which
        destroys our marker. Setting the id to None here means it gets
        recreated on the next observation.
        """
        self.TARGET_MARKER_ID = None
        return super().reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------
    #
    # HoverAviary never tells the drone where the target is. It doesn't
    # need to — the target never moves, so the network just memorises
    # "fly to [0,0,1]".
    #
    # A MOVING target cannot be memorised. If the drone can't see where
    # the target is, the task is unsolvable. So we extend the observation
    # with the target's position and velocity RELATIVE to the drone.
    #
    # Relative, not absolute — because that's what real hardware gives
    # you. UWB reports a distance to the platform; an ArUco marker
    # reports a pose relative to the camera. The drone never receives the
    # platform's world coordinates.

    def _observationSpace(self):
        """Extend the parent's observation space by 6 numbers."""
        base = super()._observationSpace()
        extra_low = np.full((base.shape[0], 6), -np.inf, dtype=np.float32)
        extra_high = np.full((base.shape[0], 6), np.inf, dtype=np.float32)
        return spaces.Box(
            low=np.hstack([base.low, extra_low]).astype(np.float32),
            high=np.hstack([base.high, extra_high]).astype(np.float32),
            dtype=np.float32
        )

    def _computeObs(self):
        """Parent observation, plus relative target position and velocity."""
        self._updateTargetMarker()

        obs = super()._computeObs()
        state = self._getDroneStateVector(0)

        rel_pos = self._getTargetPos() - state[0:3]     # where the target is, from the drone
        rel_vel = self._getTargetVel() - state[10:13]   # how fast it's moving, relative

        extra = np.hstack([rel_pos, rel_vel]).reshape(1, 6)
        return np.hstack([obs, extra]).astype('float32')

    # ------------------------------------------------------------------
    # REWARD
    # ------------------------------------------------------------------

    def _computeReward(self):
        """Three terms: be close, stay alive near the target, move with it.

        TRACKING — exp(-1.5 * dist)

            Rewards being near the target. An exponential rather than a
            polynomial, because it keeps a real gradient at every
            distance instead of going flat.

                dist = 0.0 m  ->  1.00
                dist = 0.3 m  ->  0.64
                dist = 0.6 m  ->  0.41
                dist = 1.0 m  ->  0.22

        ALIVE — 0.5, but ONLY while within 0.5 m of the target

            Without a survival bonus, crashing and sitting still score
            about the same, so the policy learns to do nothing.

            But an UNCONDITIONAL bonus is worse: the drone learns to
            drift slowly away, collecting 0.5/step for simply existing
            until it exits the arena. That is exactly what v4 did.

            Making it conditional means survival only pays while the
            drone is doing its job.

        VELOCITY MATCHING — 0.3 * exp(-|v_drone - v_target|)

            Reward that depends only on POSITION cannot distinguish
            between a drone hovering steadily 0.3 m from the target and
            one flying past it at 1 m/s. Both get the same score. So the
            policy never learns to slow down, drifts through the target
            region, and exits — which is why v5 was truncated at 5.4 s.

            This term rewards moving WITH the target.

            It is also the seed of the real project. When landing on a
            moving platform, being in the right place is not enough: if
            the platform moves at 1 m/s and you are stationary, you hit
            it sideways and tumble. Lateral relative speed at touchdown
            is one of this thesis's headline metrics, and this is where
            it starts.

        Returns
        -------
        float
            Between 0.0 and 1.8. Higher is better.
        """
        state = self._getDroneStateVector(0)

        dist = np.linalg.norm(self._getTargetPos() - state[0:3])
        tracking = np.exp(-1.5 * dist)

        alive = 0.5 if dist < 0.5 else 0.0

        rel_vel = self._getTargetVel() - state[10:13]
        vel_match = 0.3 * np.exp(-1.0 * np.linalg.norm(rel_vel))
        tilt = np.linalg.norm(state[7:9])
        smooth = -0.5 * tilt

        return float(tracking + alive + vel_match + smooth)

    # ------------------------------------------------------------------
    # EPISODE END CONDITIONS
    # ------------------------------------------------------------------

    def _computeTerminated(self):
        """Success condition. There isn't one for a tracking task —
        the drone just keeps following until time runs out."""
        return False

    def _computeTruncated(self):
        """Failure, or out of time.

        The x bound is wider than the y bound because the target itself
        swings up to TARGET_AMPLITUDE along x and the drone needs room
        to chase it.
        """
        state = self._getDroneStateVector(0)

        x_bound = 1.0 + self.TARGET_AMPLITUDE

        # flew out of the arena
        if abs(state[0]) > x_bound or abs(state[1]) > 1.0 or state[2] > 2.0:
            return True

        # hit the ground — sitting on the floor is not a valid strategy
        if state[2] < 0.05:
            return True

        # tilted past ~23 degrees (state[7] = roll, state[8] = pitch, radians)
        if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
            return True

        # ran out of time
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    def _computeInfo(self):
        """Diagnostic values. Not used for learning, but this is where your
        evaluation metrics will come from later."""
        state = self._getDroneStateVector(0)
        rel_vel = self._getTargetVel() - state[10:13]
        return {
            "tracking_error": float(np.linalg.norm(self._getTargetPos() - state[0:3])),
            "velocity_error": float(np.linalg.norm(rel_vel)),
            "target_pos": self._getTargetPos().tolist(),
        }