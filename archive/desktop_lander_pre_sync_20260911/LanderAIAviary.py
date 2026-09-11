"""
LanderAIAviary
==============

A faithful rebuild of the environment described in:

    Peter, Ratnabala, Aschu, Fedoseev, Tsetserukou. "Lander.AI: DRL-based
    Autonomous Drone Landing on Moving 3D Surface in the Presence of
    Aerodynamic Disturbances." ICUAS 2024.

Built on gym-pybullet-drones, same as the paper's own setup (Fig. 2 of the
paper: "Simulation environment based on gym-pybullet-drones framework").

SCOPE FOR TODAY: platform motion is LINEAR only (constant velocity, sampled
per episode) -- matching the paper's own staged evaluation (SPL -> LMPL ->
CMPL -> CTL). No wind-force disturbance (Eqs. 1-2 of the paper) and no
camera/vision pipeline -- both are explicitly deferred to future work.
Observation is privileged ground-truth state throughout, matching the
paper's own Eq. 3 (the paper does not use vision for its core DRL agent;
vision only appears via the marker-based real-world validation rig, not the
simulated training loop).


OBSERVATION -- Eq. 3 of the paper (PAPER FACT)
------------------------------------------------------------------------------
    o_t = [theta, v, omega, d, delta_v]

    theta    attitude of the drone (roll, pitch, yaw)              3 dims
    v        drone linear velocity                                 3 dims
    omega    drone angular velocity                                3 dims
    d        relative position of the landing pad                  3 dims
    delta_v  relative velocity of the landing pad                  3 dims
                                                                    ------
                                                                    15 dims

"These inputs are first clipped and normalized to a range of -1 to 1"
(paper, Sec III-B-1). Per-component clip bounds used here:

    theta   (+/- pi rad)                       PAPER FACT (Sec III-A)
    v       (+/-3 m/s XY, +/-2 m/s Z)           PAPER FACT (Sec III-A)
    omega   (+/- 2*pi rad/s)                    NOT stated in the paper --
                                                 assumed, documented here
    d       (+/- 5 m)                           NOT stated in the paper --
                                                 assumed, documented here
    delta_v (+/- 4 m/s)                         NOT stated in the paper --
                                                 assumed (drone max speed +
                                                 platform max speed, rounded)

Observation SPACE is therefore simply Box(-1, 1, shape=(15,)) -- clipping and
normalization both happen inside _computeObs(), so nothing outside [-1,1]
is ever returned.


ACTION -- Eq. 4-5 of the paper (PAPER FACT, "Option B" from our discussion)
------------------------------------------------------------------------------
Network output (paper, Eq. 4 -- this shapes the policy net in train_lander.py,
not this file): ReLU(FC512 x2) -> ReLU(FC256) -> ReLU(FC128) -> 3-neuron
tanh output c_t in [-1,1]^3.

Position update (paper, Eq. 5):

    delta_p_t = 0.1 * c_t

"applied to the current drone pose through the position controller of the
UAV" (paper, Sec III-B-1).

IMPLEMENTATION NOTE: gym-pybullet-drones' built-in ActionType.PID does NOT
implement this. Its _preprocessAction() treats the raw action as an
ABSOLUTE destination point and moves up to 1 metre toward it per step
(_calculateNextStep(), step_size=1) -- a much larger, differently-shaped
motion than the paper's small 0.1 m nudge. This file does NOT use
BaseRLAviary's default PID action handling. Instead, _preprocessAction() is
overridden here to compute next_pos = current_position + 0.1*clip(action,-1,1)
directly and feed that straight to DSLPIDControl.computeControl() -- the
same controller class BaseRLAviary itself uses internally, just wired up to
match the paper's exact formula rather than the library's default semantics.


REWARD -- Eq. 6-8 of the paper, WITH TWO DOCUMENTED INTERPRETATIONS
------------------------------------------------------------------------------
Paper's Eq. 6 (as published):

    Reward = tanh(gamma)                        if d_target > 2
           = tanh(alpha * (d_target - R))        if d_target in (0.1, 2)
           = tanh(-U - beta + Delta)             if d_target < 0.1
           = tanh(-U + Delta)                    otherwise

INTERPRETATION 1 -- the mid-range term. The paper defines R as "the current
distance to the target," identical to d_target's own definition. Taken
literally, (d_target - R) = 0 always, making this branch a constant zero --
inconsistent with the branch's own stated purpose ("reward scaling factor
for proximity to the target") and with the paper's framing that closer
should mean higher reward (Fig. 4's description). Implemented here as a
progress/shaping term instead: R -> distance at the PREVIOUS step, with the
subtraction order corrected so that closing the distance is rewarded:

    reward_mid = tanh(alpha * (R_prev - d_target))

This is an interpretation, not a verbatim reproduction of Eq. 6.

INTERPRETATION 2 -- the "otherwise" branch. The first three branches
(>2, (0.1,2), <0.1) already partition the entire positive real line except
the two boundary points {0.1, 2} (measure zero). There is no case left for
a fourth branch to catch. This file implements only the three well-defined
branches; the "otherwise" branch is dropped as structurally unreachable.

U = U_attractive + U_repulsive (Eq. 7-8, PAPER FACT for the formulas):

    U_attractive = 0.5 * zeta * d_target^2
    U_repulsive  = 0.5 * eta * (1/sigma - 1/Q_max)^2   if sigma < Q_max
                 = 0                                    otherwise

sigma is "the distance to the nearest obstacle" (paper). This environment
has no general obstacles; the paper itself states the repulsive term's
purpose directly: "a safety feature that penalizes low-altitude flights,
ensuring the drone remains above a safe height relative to the landing pad"
(Sec III-B-2). sigma is therefore implemented as height_above_pad, and
Q_max as a configurable minimum safe height -- this mapping is well-grounded
in the paper's own text, not a free interpretation.

beta ("adjusts for edge proximity penalties and below the landing pad
altitude") is implemented as a flat penalty applied only when the drone is
below the safe-height threshold while also within the <0.1 m close-range
branch -- i.e. specifically during the final approach, matching the paper's
description of this term applying near touchdown.

Delta ("discourages excessive speed, allowing descending relative velocity
while approaching the landing pad") is implemented as a penalty on
HORIZONTAL relative speed only, deliberately excluding the vertical
(descending) component, per the paper's explicit "allowing descending
relative velocity" clause.

None of gamma, alpha, beta, eta, zeta, Q_max are given numeric values
anywhere in the paper. All are exposed as constructor parameters with
documented default values chosen here (not derived from the paper).


TERMINATION / SUCCESS CRITERION -- NOT SPECIFIED IN THE PAPER
------------------------------------------------------------------------------
The paper reports success rates (Table I: 100% SPL, 93.33% LMPL, etc.) but
does not state the exact episode-ending rule used to produce them, beyond
the 20-second episode duration (Table I). The following is BORROWED from
Amendola et al., "Drone Landing on Moving UGV Platform with Reinforcement
Learning Based Offsets" (a different paper in the same project's reading
set), which explicitly specifies: success requires remaining within the
landing criterion for 2 consecutive control steps, avoiding single-instant
flukes. Applied here as: d_target < 0.1 m (the paper's own "close" threshold)
AND height_above_pad below a safe contact height, both held for 2 consecutive
steps.

Failure / truncation conditions (this file's own additions, not from either
paper): tilt exceeding a safety limit, height below the platform surface
(crash-equivalent), drifting outside a generous position boundary, and the
paper's own 20-second episode timeout (Table I, PAPER FACT).
"""

import os
from collections import deque

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class LanderAIAviary(BaseRLAviary):
    """Land a quadrotor on a linearly-moving platform, rebuilt to match
    Lander.AI (Peter et al., ICUAS 2024) -- see module docstring for exactly
    which parts are paper fact vs. documented interpretation vs. borrowed
    from a different paper vs. this file's own addition."""

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
                 # --- platform (paper: 0.5m cube, +/-0.46 m/s, XYZ) -------
                 # SCOPE DEVIATION: platform_speed_range is reduced from the
                 # paper's 0.46 to 0.25 m/s, and episode_len_sec from 20 to
                 # 10 s, for this session only. Rationale: at the paper's
                 # values the pad drifts up to ~9 m from the drone's spawn
                 # point within one episode, which -- against a training
                 # budget orders of magnitude smaller than the paper's own
                 # 5M-35M steps -- leaves the policy chasing a target it
                 # has not yet learned to track. Both values are exposed
                 # here as parameters: set platform_speed_range=0.46 and
                 # episode_len_sec=20.0 to restore the paper's own setup
                 # for a full-length training run.
                 platform_size: float = 0.5,
                 platform_speed_range: float = 0.25,
                 start_height: float = 1.0,
                 # --- action (PAPER FACT: Eq. 5, delta_p_t = 0.1 * c_t) ---
                 action_scale: float = 0.1,
                 # --- observation clip bounds ------------------------
                 # theta, v are PAPER FACT (Sec III-A); omega, d, delta_v
                 # are NOT stated in the paper -- documented assumptions.
                 attitude_bound: float = np.pi,               # PAPER FACT
                 vel_xy_bound: float = 3.0,                   # PAPER FACT
                 vel_z_bound: float = 2.0,                    # PAPER FACT
                 ang_vel_bound: float = 2.0 * np.pi,          # ASSUMED
                 rel_pos_bound: float = 5.0,                  # ASSUMED
                 rel_vel_bound: float = 4.0,                  # ASSUMED
                 # --- reward (Eq. 6-8): NO numeric values given in the ---
                 # --- paper. All defaults below are this file's choice. --
                 gamma: float = -1.0,        # far-away flat penalty
                 alpha: float = 2.0,         # mid-range progress scaling
                 zeta: float = 1.0,          # attractive potential strength
                 eta: float = 1.0,           # repulsive potential strength
                 beta: float = 1.0,          # close-range altitude penalty
                 min_safe_height: float = 0.15,   # Q_max: repulsive range
                 delta_scale: float = 2.0,   # horizontal-speed penalty scale
                 delta_allowed_speed: float = 0.3,  # m/s, unpenalized
                 # --- termination (BORROWED from Amendola et al. for the --
                 # --- hold-time idea; thresholds are this file's choice) --
                 success_distance: float = 0.1,     # PAPER FACT (the "<0.1" cutoff)
                 success_hold_steps: int = 2,       # BORROWED (Amendola et al.)
                 tilt_limit: float = 0.4,           # rad, this file's own choice
                 position_bound: float = 10.0,      # metres, this file's own choice
                 episode_len_sec: float = 10.0,     # paper: 20.0 (Table I) --
                                                    # halved for this session,
                                                    # see platform_speed_range
                                                    # note above
                 ):

        self.PLAT_SIZE = platform_size
        self.PLATFORM_SPEED_RANGE = platform_speed_range
        self.START_HEIGHT = start_height
        self.PLAT_ID = None
        self.plat_vel = np.zeros(3)
        self.plat_pos0 = np.zeros(3)

        self.ACTION_SCALE = action_scale

        self.ATTITUDE_BOUND = attitude_bound
        self.VEL_XY_BOUND = vel_xy_bound
        self.VEL_Z_BOUND = vel_z_bound
        self.ANG_VEL_BOUND = ang_vel_bound
        self.REL_POS_BOUND = rel_pos_bound
        self.REL_VEL_BOUND = rel_vel_bound

        self.GAMMA = gamma
        self.ALPHA = alpha
        self.ZETA = zeta
        self.ETA = eta
        self.BETA = beta
        self.MIN_SAFE_HEIGHT = min_safe_height   # Q_max
        self.DELTA_SCALE = delta_scale
        self.DELTA_ALLOWED_SPEED = delta_allowed_speed
        self.prev_dist = None   # for the progress-based mid-range term

        self.SUCCESS_DISTANCE = success_distance
        self.SUCCESS_HOLD_STEPS = success_hold_steps
        self.TILT_LIMIT = tilt_limit
        self.POSITION_BOUND = position_bound
        self.EPISODE_LEN_SEC = episode_len_sec
        self.success_hold_counter = 0
        self.landed_successfully = False
        self.episode_resolved = False
        self.truncation_reason = None

        # DSLPIDControl: the same low-level position controller class
        # BaseRLAviary itself uses internally for ActionType.PID -- we
        # instantiate and drive it directly (see _preprocessAction below)
        # rather than going through BaseRLAviary's default PID action path,
        # which has different (larger-step) semantics than the paper's
        # Eq. 5. act=ActionType.PID is still passed to super().__init__ so
        # the action SPACE (3-dim, [-1,1]) matches what we need; only the
        # per-step interpretation of that action is overridden.
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X)]

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
                         act=ActionType.PID,
                         )

    # ------------------------------------------------------------------
    # PLATFORM -- linear motion only (this session's scope; paper's own
    # LMPL scenario). Constant per-episode velocity, sampled within the
    # paper's stated +/-0.46 m/s range (PAPER FACT), each axis independent.
    # ------------------------------------------------------------------

    def _samplePlatformMotion(self):
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        self.plat_vel = rng.uniform(-self.PLATFORM_SPEED_RANGE,
                                    self.PLATFORM_SPEED_RANGE, 3)
        self.plat_vel[2] *= 0.3   # keep vertical drift gentle; horizontal
                                  # linear motion is the paper's LMPL case
        # Random start position under/near the drone's spawn point.
        self.plat_pos0 = np.array([
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.5, 0.5),
            self.PLAT_SIZE / 2])

    def _getPlatformPos(self):
        t = self.step_counter / self.PYB_FREQ
        return self.plat_pos0 + self.plat_vel * t

    def _getPlatformVel(self):
        return self.plat_vel.copy()

    def _updatePlatform(self):
        pos = self._getPlatformPos()
        if self.PLAT_ID is None:
            half = [self.PLAT_SIZE / 2] * 3
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=self.CLIENT)
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half, rgbaColor=[1, 1, 1, 1],
                physicsClientId=self.CLIENT)
            self.PLAT_ID = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual, basePosition=pos.tolist(),
                physicsClientId=self.CLIENT)
        else:
            p.resetBasePositionAndOrientation(
                self.PLAT_ID, pos.tolist(), [0, 0, 0, 1],
                physicsClientId=self.CLIENT)

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self.PLAT_ID = None
        self.prev_dist = None
        self.success_hold_counter = 0
        self.landed_successfully = False
        self.episode_resolved = False
        self.truncation_reason = None

        out = super().reset(seed=seed, options=options)
        self._samplePlatformMotion()
        # NOTE: out[1] is the info dict from BaseRLAviary's own reset(),
        # computed BEFORE _samplePlatformMotion() has run -- it reflects a
        # stale, pre-randomization platform position. Recompute obs/info
        # here so both reflect the actual sampled platform motion.
        obs = self._computeObs()
        return obs, self._computeInfo()

    # ------------------------------------------------------------------
    # ACTION -- Eq. 5 of the paper (PAPER FACT), "Option B": computed
    # directly rather than via BaseRLAviary's default PID handling. See
    # module docstring, "ACTION" section, for why.
    # ------------------------------------------------------------------

    def _actionSpace(self):
        # BaseRLAviary's default action space for ActionType.PID has shape
        # (NUM_DRONES, 3) = (1, 3) here -- fine for PPO, but confirmed via
        # diagnostic to break TD3's replay buffer reshape logic (produced
        # a 4x size mismatch: 48 vs the expected 12 for n_envs=4,
        # action_dim=3). Overridden here to the standard flat Gymnasium
        # single-agent shape (3,) instead.
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _preprocessAction(self, action):
        self.action_buffer.append(action)

        state = self._getDroneStateVector(0)
        current_pos = state[0:3]

        # Defensive flatten: works whether SB3/VecEnv hands us (3,), (1,3),
        # or anything else that reshapes to 3 elements -- avoids re-baking
        # in an assumption about exact incoming shape.
        c_t = np.clip(np.asarray(action).flatten(), -1.0, 1.0)
        delta_p = self.ACTION_SCALE * c_t          # Eq. 5: delta_p_t = 0.1*c_t
        next_pos = current_pos + delta_p

        rpm_k, _, _ = self.ctrl[0].computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=next_pos)
        return rpm_k.reshape(1, 4)

    # ------------------------------------------------------------------
    # OBSERVATION -- Eq. 3 of the paper (PAPER FACT for the composition;
    # see module docstring for which clip bounds are paper fact vs.
    # assumed). Overrides BaseRLAviary's default KIN observation entirely
    # -- we want exactly the 15 components Eq. 3 specifies, nothing more.
    # ------------------------------------------------------------------

    def _observationSpace(self):
        # Flat (15,) to match the flat action-space convention above --
        # same reasoning: standard single-agent Gymnasium shape rather than
        # BaseRLAviary's (NUM_DRONES, 15) default.
        return spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32)

    def _computeObs(self):
        self._updatePlatform()

        state = self._getDroneStateVector(0)
        theta = state[7:10]     # roll, pitch, yaw
        v = state[10:13]        # linear velocity
        omega = state[13:16]    # angular velocity

        plat_pos = self._getPlatformPos()
        plat_vel = self._getPlatformVel()
        d = plat_pos - state[0:3]           # relative landing pad position
        delta_v = plat_vel - v              # relative landing pad velocity

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

    # ------------------------------------------------------------------
    # REWARD -- Eq. 6-8. See module docstring "REWARD" section for the
    # two documented interpretations (mid-range term, dropped "otherwise"
    # branch) and which constants are paper fact vs. this file's choice.
    # ------------------------------------------------------------------

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]
        plat_pos = self._getPlatformPos()
        plat_vel = self._getPlatformVel()

        d_target = float(np.linalg.norm(plat_pos - drone_pos))
        height_above_pad = float(drone_pos[2] - plat_pos[2])
        rel_vel = drone_vel - plat_vel
        horiz_rel_speed = float(np.linalg.norm(rel_vel[0:2]))

        if self.prev_dist is None:
            self.prev_dist = d_target

        # Delta: penalize horizontal relative speed only, NEVER the
        # descending vertical component (paper: "allowing descending
        # relative velocity while approaching the landing pad").
        excess_speed = max(0.0, horiz_rel_speed - self.DELTA_ALLOWED_SPEED)
        delta_term = -self.DELTA_SCALE * excess_speed

        if d_target > 2.0:
            # DEVIATION FROM Eq. 6 -- DOCUMENTED AND DELIBERATE.
            #
            # The paper's far-field branch is tanh(gamma): a FLAT penalty,
            # identical at 2.1 m and at 10 m. It carries no gradient, so a
            # policy that has fallen behind the drifting platform receives
            # no signal that closing distance is an improvement. With this
            # environment's platform drift (up to 0.46 m/s x 20 s ~ 9 m),
            # episodes spend substantial time in that dead zone -- observed
            # directly: 55/100 timeouts and 43/100 below-platform failures
            # at 1.5M steps, with zero successful landings.
            #
            # Replaced with a distance-graded penalty PLUS the same
            # progress term used in the mid-range branch, so there is a
            # continuous incentive to close distance from any range. The
            # tanh envelope and the sign convention (far = negative) are
            # kept from the paper.
            progress = self.prev_dist - d_target
            reward = np.tanh(self.GAMMA * min(d_target / 5.0, 1.0)
                             + self.ALPHA * progress)

        elif d_target > 0.1:
            # INTERPRETATION (see module docstring): progress-based, using
            # the PREVIOUS step's distance in place of the paper's
            # self-contradictory R, with sign corrected so closing the
            # distance is rewarded.
            reward = np.tanh(self.ALPHA * (self.prev_dist - d_target))

        else:
            # d_target < 0.1: close-range branch (PAPER FACT for U/Delta
            # inclusion; beta interpretation per module docstring).
            u_attractive = 0.5 * self.ZETA * d_target ** 2
            sigma = height_above_pad
            if 0 < sigma < self.MIN_SAFE_HEIGHT:
                u_repulsive = 0.5 * self.ETA * (1.0 / sigma - 1.0 / self.MIN_SAFE_HEIGHT) ** 2
            else:
                u_repulsive = 0.0
            U = u_attractive + u_repulsive

            beta_term = self.BETA if height_above_pad < self.MIN_SAFE_HEIGHT else 0.0
            reward = np.tanh(-U - beta_term + delta_term)

        self.prev_dist = d_target
        return float(reward)

    # ------------------------------------------------------------------
    # EPISODE END
    # ------------------------------------------------------------------

    def _computeTerminated(self):
        if self.episode_resolved:
            return True

        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        plat_pos = self._getPlatformPos()
        d_target = float(np.linalg.norm(plat_pos - drone_pos))
        height_above_pad = float(drone_pos[2] - plat_pos[2])

        # BORROWED (Amendola et al.): require the success condition to
        # hold for consecutive steps, not a single instant.
        close_enough = d_target < self.SUCCESS_DISTANCE
        safe_height = 0 <= height_above_pad < self.MIN_SAFE_HEIGHT * 2

        if close_enough and safe_height:
            self.success_hold_counter += 1
        else:
            self.success_hold_counter = 0

        if self.success_hold_counter >= self.SUCCESS_HOLD_STEPS:
            self.episode_resolved = True
            self.landed_successfully = True
            return True

        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        plat_pos = self._getPlatformPos()

        tilt = float(np.linalg.norm(state[7:9]))
        if tilt > self.TILT_LIMIT:
            self.truncation_reason = "tilt_bound"
            return True

        height_above_pad = drone_pos[2] - plat_pos[2]
        if height_above_pad < -0.05:
            self.truncation_reason = "below_platform"
            return True

        if (abs(drone_pos[0]) > self.POSITION_BOUND
                or abs(drone_pos[1]) > self.POSITION_BOUND
                or drone_pos[2] > self.POSITION_BOUND
                or drone_pos[2] < 0):
            self.truncation_reason = "position_bound"
            return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            self.truncation_reason = "timeout"
            return True

        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        plat_pos = self._getPlatformPos()
        d_target = float(np.linalg.norm(plat_pos - drone_pos))

        return {
            "d_target": d_target,
            "height_above_pad": float(drone_pos[2] - plat_pos[2]),
            "success": self.landed_successfully,
            "resolved": self.episode_resolved,
            "truncation_reason": self.truncation_reason,
            "plat_speed": float(np.linalg.norm(self.plat_vel)),
        }
