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

sigma is "the distance to the nearest obstacle" (paper). This current
environment contains no separate obstacles, so U_repulsive is set to zero.
In particular, height_above_pad is NOT used as sigma: the landing surface is
the target, not an obstacle, and treating it as one creates an artificial
near-singular penalty at touchdown. When explicit obstacles are added later,
their true nearest-obstacle distance should be used for sigma.

beta ("adjusts for edge proximity penalties and below the landing pad
altitude") has no exact functional form in the paper. In this current
minimal environment it is applied only when the drone is below the landing
surface. A separate edge-proximity model can be added later once its exact
geometry/penalty is defined; normal low-altitude touchdown is not penalized.

Delta ("discourages excessive speed, allowing descending relative velocity
while approaching the landing pad") is implemented as a penalty on excessive
horizontal relative speed AND excessive downward relative speed. Gentle
descent is explicitly allowed through a configurable threshold. The paper
does not publish the numerical threshold used here.

None of gamma, alpha, beta, eta, zeta, Q_max are given numeric values
anywhere in the paper. All are exposed as constructor parameters with
documented default values chosen here (not derived from the paper).


TERMINATION / SUCCESS CRITERION -- NOT SPECIFIED IN THE PAPER
------------------------------------------------------------------------------
The paper reports success rates (Table I: 100% SPL, 93.33% LMPL, etc.) but
does not state the exact episode-ending rule used to produce them, beyond
the 20-second episode duration (Table I). The following hold-time idea is BORROWED from
Amendola et al., "Drone Landing on Moving UGV Platform with Reinforcement
Learning Based Offsets" (a different paper in the same project's reading
set): success requires remaining within the landing criterion for 2
consecutive control steps, avoiding single-instant flukes. In this
implementation the criterion is actual PyBullet drone-platform contact plus
horizontal error < 0.1 m, held for 2 consecutive control steps.

The current implementation uses actual PyBullet drone-platform contact,
plus horizontal error < success_distance, held for 2 consecutive control
steps. The Crazyflie's 0.025 m collision-cylinder height is therefore handled
by the physics engine instead of by a loose centre-height threshold.

Failure conditions (this file's own additions, not from either paper): tilt
exceeding a safety limit, missing below the platform, and leaving a generous
position boundary are true terminations and receive an explicit failure
penalty. The episode deadline is a Gymnasium truncation and receives an
explicit timeout penalty so stalling is not reward-neutral.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
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
                 # Multiplier on the sampled vertical platform velocity.
                 # 0.0 = horizontal-only linear motion (the paper's LMPL
                 # scenario). Any nonzero value drifts the pad without bound
                 # -- see the note in _samplePlatformMotion before changing.
                 vertical_speed_factor: float = 0.0,
                 start_height: float = 1.0,
                 # --- spawn annulus (curriculum) ------------------------
                 # Platform XY spawn is sampled on an annulus of this radius
                 # range. Previously hardcoded inside _samplePlatformMotion();
                 # exposed here so a training callback can widen it via
                 # set_spawn_radius() INSIDE one uninterrupted TD3 run,
                 # instead of the save/reload fine-tuning that was observed
                 # to destroy good policies.
                 spawn_radius_min: float = 0.11,
                 spawn_radius_max: float = 0.14,
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
                 delta_scale: float = 2.0,   # speed-penalty scale
                 delta_allowed_speed: float = 0.3,  # horizontal rel. speed, m/s
                 delta_allowed_descent_speed: float = 0.2,  # downward rel. speed, m/s
                 # --- termination/reward-end signals --------------------
                 # The paper's 0.1 m value is a REWARD-region cutoff, not a
                 # published landing-success tolerance. The 0.1 m XY landing
                 # tolerance below is therefore an implementation choice.
                 success_distance: float = 0.1,     # XY contact tolerance, IMPLEMENTATION CHOICE
                 success_hold_steps: int = 2,       # BORROWED (Amendola et al.)
                 # REBALANCED -- diagnosis of the 11-30 cm collapse.
                 # With a terminal-only timeout penalty, gamma^300 ~ 0.05
                 # made stalling nearly free while an immediate crash cost
                 # full price. Hovering was therefore the value-maximising
                 # policy whenever success probability fell below ~45%,
                 # which is exactly why 11-14 cm trained fine (exploration
                 # finds contact often) and 11-30 cm collapsed to a
                 # 300-step stall with Q ~ -3.8. See time_cost below.
                 success_bonus: float = 20.0,             # was 1.0
                 terminal_failure_penalty: float = -5.0,  # was -1.0
                 timeout_penalty: float = -1.0,           # unchanged
                 # Per-step cost. Unlike a terminal penalty this CANNOT be
                 # evaded by discounting: 300 steps of -0.02 discounted at
                 # gamma=0.99 is about -1.9, so hovering now has a real
                 # price. This is the term that removes the stall attractor.
                 time_cost: float = 0.02,
                 tilt_limit: float = 0.4,           # rad, this file's own choice
                 position_bound: float = 10.0,      # metres, this file's own choice
                 episode_len_sec: float = 10.0,     # paper: 20.0 (Table I) --
                                                    # halved for this session,
                                                    # see platform_speed_range
                                                    # note above
                 ):

        self.PLAT_SIZE = platform_size
        self.PLATFORM_SPEED_RANGE = platform_speed_range
        self.VERTICAL_SPEED_FACTOR = vertical_speed_factor
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
        self.HORIZONTAL_ALPHA = 5.0   # IMPLEMENTATION CHOICE
        self.ZETA = zeta
        self.ETA = eta
        self.BETA = beta
        self.MIN_SAFE_HEIGHT = min_safe_height   # Q_max
        self.DELTA_SCALE = delta_scale
        self.DELTA_ALLOWED_SPEED = delta_allowed_speed
        self.DELTA_ALLOWED_DESCENT_SPEED = delta_allowed_descent_speed
        self.CLOSE_REWARD_DISTANCE = 0.1  # PAPER FACT: Eq. 6 branch cutoff
        self.prev_dist = None   # for the progress-based shaping term
        self.prev_horizontal_error = None

        self.SUCCESS_DISTANCE = success_distance
        self.SUCCESS_HOLD_STEPS = success_hold_steps
        self.SUCCESS_BONUS = success_bonus
        self.TERMINAL_FAILURE_PENALTY = terminal_failure_penalty
        self.TIMEOUT_PENALTY = timeout_penalty
        self.TIME_COST = time_cost
        self.SPAWN_RADIUS_MIN = spawn_radius_min
        self.SPAWN_RADIUS_MAX = spawn_radius_max
        self.TILT_LIMIT = tilt_limit
        self.POSITION_BOUND = position_bound
        self.EPISODE_LEN_SEC = episode_len_sec
        self.success_hold_counter = 0
        self.landed_successfully = False
        self.episode_resolved = False
        self.truncation_reason = None

        if self.SUCCESS_HOLD_STEPS < 1:
            raise ValueError("success_hold_steps must be >= 1")
        if self.SUCCESS_DISTANCE <= 0:
            raise ValueError("success_distance must be > 0")
        if self.DELTA_ALLOWED_SPEED < 0 or self.DELTA_ALLOWED_DESCENT_SPEED < 0:
            raise ValueError("allowed speed thresholds must be >= 0")

        # BaseRLAviary creates the integrated DSLPIDControl for ActionType.PID.
        # We still override _preprocessAction() so ActionType.PID follows
        # the paper's delta_p_t = 0.1*c_t semantics instead of the base
        # class's absolute-destination interpretation.

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

    def set_spawn_radius(self, radius_min, radius_max):
        """Curriculum hook, called via VecEnv.env_method() during training.

        Widening the spawn annulus inside ONE uninterrupted TD3 run keeps the
        replay buffer, optimizer moments and target networks continuous --
        removing the discontinuity that save/reload fine-tuning introduced
        (which degraded every good checkpoint it was applied to).
        """
        self.SPAWN_RADIUS_MIN = float(radius_min)
        self.SPAWN_RADIUS_MAX = float(radius_max)

    def set_platform_speed(self, speed_range):
        """Curriculum hook for platform motion, called via env_method().

        Sets the half-range from which each episode's constant platform
        velocity is drawn. 0.0 reproduces the stationary task exactly.
        """
        self.PLATFORM_SPEED_RANGE = float(speed_range)

    def _samplePlatformMotion(self):
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        self.plat_vel = rng.uniform(-self.PLATFORM_SPEED_RANGE,
                                    self.PLATFORM_SPEED_RANGE, 3)
        # Vertical drift defaults to ZERO. _commandedPlatformPos integrates
        # plat_vel linearly with no bound, so any nonzero vertical component
        # walks the pad out of the world within one episode: at the previous
        # 0.3 factor and a 0.25 m/s range the pad moved +/-0.75 m over 10 s,
        # from a start of z=0.25. Downward that buries it under the floor;
        # upward it puts the pad's top surface at 1.25 m, ABOVE the drone's
        # 1.0 m spawn, so height_above_pad goes negative and
        # _getFailureReason returns "below_platform" -- an unavoidable
        # failure. The paper's LMPL scenario is horizontal linear motion;
        # 3D surface motion belongs to its later CMPL/CTL scenarios and
        # needs a bounded trajectory, not unbounded linear drift.
        self.plat_vel[2] *= self.VERTICAL_SPEED_FACTOR
        # Random start position on the curriculum annulus. Every episode
        # starts strictly outside SUCCESS_DISTANCE so no episode can begin
        # already inside the success region.
        theta = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(self.SPAWN_RADIUS_MIN, self.SPAWN_RADIUS_MAX)

        self.plat_pos0 = np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            self.PLAT_SIZE / 2.0
        ])

    def _commandedPlatformPos(self, step_offset=0):
        """Platform trajectory evaluated at a simulation-step offset."""
        t = (self.step_counter + step_offset) / self.PYB_FREQ
        return self.plat_pos0 + self.plat_vel * t

    def _getPlatformPos(self):
        """Return the platform's ACTUAL current PyBullet base position."""
        if self.PLAT_ID is not None:
            pos, _ = p.getBasePositionAndOrientation(
                self.PLAT_ID, physicsClientId=self.CLIENT
            )
            return np.asarray(pos, dtype=float)
        return self._commandedPlatformPos(step_offset=0)

    def _getLandingTargetPos(self):
        """Return the center of the TOP SURFACE of the actual platform."""
        target = self._getPlatformPos().copy()
        target[2] += self.PLAT_SIZE / 2.0
        return target

    def _getPlatformVel(self):
        return self.plat_vel.copy()

    def _hasValidLandingContact(self):
        """Return True when the drone is physically touching the platform
        within the allowed horizontal landing region.

        PyBullet contact detection handles the Crazyflie's collision geometry
        directly (COLLISION_H=0.025 m, so an upright touchdown naturally
        occurs with the drone centre about 0.0125 m above the platform top).
        """
        if self.PLAT_ID is None:
            return False

        contacts = p.getContactPoints(
            bodyA=int(self.DRONE_IDS[0]),
            bodyB=int(self.PLAT_ID),
            physicsClientId=self.CLIENT,
        )

        if len(contacts) == 0:
            return False

        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        target = self._getLandingTargetPos()

        horizontal_error = float(
            np.linalg.norm(drone_pos[:2] - target[:2])
        )

        return horizontal_error < self.SUCCESS_DISTANCE

    def _updatePlatform(self, step_offset=0):
        """Place the kinematic platform on its commanded trajectory.

        BaseAviary computes reward/termination before incrementing
        step_counter. Updating to the end of the upcoming control interval
        before physics avoids the previous one-control-step platform lag.

        resetBasePositionAndOrientation() zeros base velocity, so we restore
        the commanded linear velocity as well; otherwise moving-platform
        contacts would behave like contacts with a stationary surface.
        """
        pos = self._commandedPlatformPos(step_offset=step_offset)

        if self.PLAT_ID is None:
            half = [self.PLAT_SIZE / 2] * 3
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=self.CLIENT)
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half, rgbaColor=[1, 1, 1, 1],
                physicsClientId=self.CLIENT)
            self.PLAT_ID = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=pos.tolist(),
                physicsClientId=self.CLIENT,
            )
        else:
            p.resetBasePositionAndOrientation(
                self.PLAT_ID,
                pos.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT,
            )

        p.resetBaseVelocity(
            self.PLAT_ID,
            linearVelocity=self.plat_vel.tolist(),
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.CLIENT,
        )

    def _getFailureReason(self, state=None, plat_pos=None):
        """Return a true task-failure reason, or None.

        Centralizing this logic prevents reward and termination from silently
        using different failure definitions.
        """
        if state is None:
            state = self._getDroneStateVector(0)
        if plat_pos is None:
            plat_pos = self._getLandingTargetPos()

        drone_pos = state[0:3]
        tilt = float(np.linalg.norm(state[7:9]))
        if tilt > self.TILT_LIMIT:
            return "tilt_bound"

        height_above_pad = float(drone_pos[2] - plat_pos[2])
        if height_above_pad < -0.05:
            return "below_platform"

        if (abs(drone_pos[0]) > self.POSITION_BOUND
                or abs(drone_pos[1]) > self.POSITION_BOUND
                or drone_pos[2] > self.POSITION_BOUND
                or drone_pos[2] < 0):
            return "position_bound"

        return None

    def _willTimeoutAfterCurrentStep(self):
        """True when the just-executed control step reaches the time limit.

        BaseAviary increments step_counter only AFTER calling this environment's
        reward/termination/truncation methods. Using the prospective counter
        removes the previous off-by-one episode-length error.
        """
        elapsed_after_step = (
            self.step_counter + self.PYB_STEPS_PER_CTRL
        ) / self.PYB_FREQ
        return elapsed_after_step >= self.EPISODE_LEN_SEC

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self.PLAT_ID = None
        self.prev_dist = None
        self.prev_horizontal_error = None
        self.success_hold_counter = 0
        self.landed_successfully = False
        self.episode_resolved = False
        self.truncation_reason = None

        out = super().reset(seed=seed, options=options)

        # IMPORTANT: BaseRLAviary does not reset the DSLPIDControl's integral
        # and previous-error state when the Gym environment resets. Carrying
        # controller state across episodes makes identical observations behave
        # differently depending on the previous episode.
        for controller in self.ctrl:
            controller.reset()

        # Keep BaseRLAviary's action history internally well-defined even
        # though this environment's 15-D observation does not expose it.
        self.action_buffer.clear()
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(
                np.zeros((1, 3), dtype=np.float32)
            )

        self._samplePlatformMotion()
        self._updatePlatform(step_offset=0)

        # super().reset() computed obs/info before the new platform motion was
        # sampled. Recompute both against the actual episode platform state.
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
        # Accept either (3,) or (1,3), but fail loudly on any other size.
        c_t = np.asarray(action, dtype=np.float32).reshape(-1)
        if c_t.size != 3:
            raise ValueError(
                f"LanderAIAviary expects exactly 3 action values, got shape "
                f"{np.asarray(action).shape}"
            )
        c_t = np.clip(c_t, -1.0, 1.0)
        self.action_buffer.append(c_t.reshape(1, 3))

        # Place the platform at its current commanded position.
        # PyBullet advances it during the upcoming control interval using
        # the commanded base velocity.
        self._updatePlatform(step_offset=0)

        state = self._getDroneStateVector(0)
        current_pos = state[0:3]

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
        # During super().reset() the platform may not exist yet.
        if self.PLAT_ID is None:
            self._updatePlatform(step_offset=0)

        state = self._getDroneStateVector(0)
        theta = state[7:10]     # roll, pitch, yaw
        v = state[10:13]        # linear velocity
        omega = state[13:16]    # angular velocity

        plat_pos = self._getLandingTargetPos()
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
        plat_pos = self._getLandingTargetPos()
        plat_vel = self._getPlatformVel()

        d_target = float(np.linalg.norm(plat_pos - drone_pos))
        horizontal_error = float(
            np.linalg.norm(plat_pos[0:2] - drone_pos[0:2])
        )
        height_above_pad = float(drone_pos[2] - plat_pos[2])
        rel_vel = drone_vel - plat_vel

        horiz_rel_speed = float(np.linalg.norm(rel_vel[0:2]))
        descent_rel_speed = max(0.0, float(-rel_vel[2]))

        if self.prev_dist is None:
            self.prev_dist = d_target

        if self.prev_horizontal_error is None:
            self.prev_horizontal_error = horizontal_error

        progress = self.prev_dist - d_target
        horizontal_progress = (
            self.prev_horizontal_error - horizontal_error
        )

        # Delta: the paper says excessive speed is discouraged while descent
        # remains allowed. The earlier implementation incorrectly interpreted
        # that as "never penalize vertical speed". We now allow a configurable
        # gentle descent rate and penalize only EXCESS downward relative speed.
        horizontal_excess = max(
            0.0, horiz_rel_speed - self.DELTA_ALLOWED_SPEED
        )
        descent_excess = max(
            0.0, descent_rel_speed - self.DELTA_ALLOWED_DESCENT_SPEED
        )
        delta_term = -self.DELTA_SCALE * (
            horizontal_excess + descent_excess
        )

        if d_target > 2.0:
            # DOCUMENTED DEVIATION from the paper's flat tanh(gamma) branch:
            # retain a distance-graded far penalty plus progress information
            # so the policy has a learning signal when far from the target.
            reward = np.tanh(
                self.GAMMA * min(d_target / 5.0, 1.0)
                + self.ALPHA * progress
            )

        elif d_target > self.CLOSE_REWARD_DISTANCE:
            # DOCUMENTED INTERPRETATION of the paper's ambiguous
            # tanh(alpha*(d_target-R)) term: reward progress toward the pad.
            # Additional IMPLEMENTATION CHOICE: explicitly reward horizontal
            # alignment progress so vertical descent alone is not sufficient.
            reward = np.tanh(
                self.ALPHA * progress
                + self.HORIZONTAL_ALPHA * horizontal_progress
                + delta_term
            )

        else:
            # Close-range potential-field branch.
            u_attractive = 0.5 * self.ZETA * d_target ** 2
            u_repulsive = 0.0  # no separate obstacles in this environment
            U = u_attractive + u_repulsive

            # The paper does not give beta's exact functional form.
            beta_term = self.BETA if height_above_pad < 0.0 else 0.0

            # IMPORTANT STABILIZATION:
            # Keep the same progress signal through the <0.1 m boundary.
            # Previously reward jumped from a positive progress reward just
            # outside 0.1 m to a slightly negative potential reward just
            # inside 0.1 m, creating an artificial incentive to avoid the
            # final approach region.
            reward = np.tanh(
                self.ALPHA * progress
                + self.HORIZONTAL_ALPHA * horizontal_progress
                - U
                - beta_term
                + delta_term
            )

        # Per-step time cost (see __init__). A terminal timeout penalty is
        # discounted by gamma^T and is therefore nearly invisible to the
        # critic; a per-step cost is not. This is what makes "hover until
        # the deadline" strictly worse than attempting the landing.
        reward -= self.TIME_COST

        # Event rewards/penalties are ordered by priority:
        # true failure > successful landing > timeout > shaped reward.
        failure_reason = self._getFailureReason(
            state=state, plat_pos=plat_pos
        )

        valid_contact = self._hasValidLandingContact()
        completes_success_hold = (
            valid_contact
            and self.success_hold_counter >= self.SUCCESS_HOLD_STEPS - 1
        )

        if failure_reason is not None:
            reward = self.TERMINAL_FAILURE_PENALTY

        elif completes_success_hold:
            reward += self.SUCCESS_BONUS

        elif self._willTimeoutAfterCurrentStep():
            # The earlier reward allowed a hover/stall policy to finish with
            # approximately zero return. An explicit timeout penalty makes
            # "did not land before the deadline" distinguishable from success.
            reward += self.TIMEOUT_PENALTY

        self.prev_dist = d_target
        self.prev_horizontal_error = horizontal_error
        return float(reward)

    # ------------------------------------------------------------------
    # EPISODE END
    # ------------------------------------------------------------------

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        plat_pos = self._getLandingTargetPos()

        failure_reason = self._getFailureReason(
            state=state, plat_pos=plat_pos
        )
        if failure_reason is not None:
            self.truncation_reason = failure_reason
            self.episode_resolved = True
            return True

        valid_contact = self._hasValidLandingContact()

        if valid_contact:
            self.success_hold_counter += 1
        else:
            self.success_hold_counter = 0

        if self.success_hold_counter >= self.SUCCESS_HOLD_STEPS:
            self.episode_resolved = True
            self.landed_successfully = True
            self.truncation_reason = None
            return True

        return False

    def _computeTruncated(self):
        # Never report both a true terminal outcome and a timeout.
        if self.landed_successfully:
            return False

        if self._getFailureReason() is not None:
            return False

        if self._willTimeoutAfterCurrentStep():
            self.truncation_reason = "timeout"
            return True

        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]
        plat_pos = self._getLandingTargetPos()
        plat_vel = self._getPlatformVel()

        d_target = float(np.linalg.norm(plat_pos - drone_pos))
        horizontal_error = float(
            np.linalg.norm(drone_pos[:2] - plat_pos[:2])
        )
        height_above_pad = float(drone_pos[2] - plat_pos[2])
        bottom_clearance = (
            self.COLLISION_H / 2.0 - self.COLLISION_Z_OFFSET
        )
        bottom_gap = height_above_pad - bottom_clearance
        rel_vel = drone_vel - plat_vel

        return {
            "d_target": d_target,
            "horizontal_error": horizontal_error,
            "height_above_pad": height_above_pad,
            "bottom_gap": float(bottom_gap),
            "contact": bool(self._hasValidLandingContact()),
            "relative_vz": float(rel_vel[2]),
            "horizontal_relative_speed": float(
                np.linalg.norm(rel_vel[:2])
            ),
            "success": self.landed_successfully,
            "resolved": self.episode_resolved,
            "truncation_reason": self.truncation_reason,
            "plat_speed": float(np.linalg.norm(self.plat_vel)),
        }

