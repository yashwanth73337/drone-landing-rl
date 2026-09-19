"""
ShinLanderAviary
================

V4 environment. The first stage in this project that attempts the FULL method
of Shin et al. (RA-L 2026) rather than one component of it.

    LanderAIAviary        (frozen -- privileged baseline)
        |
    VisionLanderAviary    (frozen -- V1, camera observes but never feeds)
        |
    ArucoLanderAviary     (frozen -- V2/V3a, constant-velocity prediction)
        |
    LSTMLanderAviary      (frozen -- V3b, frozen offline LSTM estimator)
        |
    ShinLanderAviary      (this file -- V4, joint estimation + control)


WHAT CHANGES, AND WHY IT IS MORE THAN ONE VARIABLE
------------------------------------------------------------------------------
Every stage before this one moved exactly one variable. V4 deliberately does
not, and that has to be stated plainly rather than glossed:

    1. the estimator is trained JOINTLY with the policy, not offline+frozen
    2. the algorithm becomes PPO (recurrent), not TD3
    3. the critic becomes ASYMMETRIC and privileged
    4. an ACTIVE-PERCEPTION reward couples control to estimation error

These four are not separable. A recurrent estimator inside the policy cannot
be trained by TD3 without sequence-based replay, which is what forced PPO in
the first place (V3b report, section 2). An auxiliary estimation loss has
nowhere to attach unless the estimator is inside the policy. The
active-perception reward is defined in terms of that auxiliary loss, so it
cannot exist without it. The privileged critic is what makes value estimation
tractable once the actor is partially observed.

So V4 is a SYSTEM, and the ablations that separate its parts come afterwards
-- which is the structure the paper itself uses (Table IV). The V3-series
provides two of those ablation points already:

    V3a   no learned estimator at all        (constant velocity)
    V3b   learned estimator, frozen, no AP   (offline supervised)
    V4    learned estimator, joint, with AP  (this stage)


WHAT THE ENVIRONMENT NO LONGER DOES
------------------------------------------------------------------------------
It no longer estimates. In V3b the LSTM lived here, in the observation
pipeline, and the actor received a finished 15-D estimate. In V4 the LSTM
lives INSIDE the policy, so this environment reports only what the camera
actually saw and leaves temporal integration to the network.

That is not a simplification -- it is the paper's architecture (Fig. 4). The
hidden state, the state-estimation head and the policy head all hang off one
trunk, so the representation that produces the action is the same one that is
supervised to predict the relative state.

The constant-velocity estimator inherited from ArucoLanderAviary is still
advanced every step, purely as a passive logging baseline, exactly as in
V3b. It never reaches the actor.


OBSERVATION -- Dict, three keys
------------------------------------------------------------------------------
    'actor'   13-D  what the deployed policy sees. NO ground truth.
    'critic'  15-D  privileged, o_priv = [u_t, s_rel_t]. Removed at deployment.
    'target'   6-D  s_rel_t in physical units, the auxiliary loss label.

'actor' is, component for component, the 13-D vector the V3b estimator was
trained on:

    d_meas (3), valid (1), velocity (3), rpy (3), angular velocity (3)

During blindness d_meas is [0,0,0] and valid is 0. The previous measurement
is NOT held -- carrying temporal information is the recurrent state's job,
and that is the capability under test.

'critic' is [rpy, velocity, angular velocity, d_true, dv_true] = 9 + 6. The
paper's o_priv = [u_t, s_rel_t]. This is privileged by construction and must
never be routed to the actor. The policy in
`policies/asymmetric_recurrent.py` reads 'actor' for the action and 'critic'
for the value, and that separation is the only thing keeping this honest.

'target' carries s_rel_t in PHYSICAL UNITS, unnormalised, because the paper's
auxiliary loss (Eq. 1) is a plain MSE over the six components and the
project reports all estimator error in metres and m/s. Scaling it is a
training-side decision and belongs to the trainer's --aux-scale, not here.


NORMALISATION -- a documented difference from V3b
------------------------------------------------------------------------------
V3b's estimator standardised its input per-component using statistics from
the offline training split. V4 has no offline split -- the estimator is
trained on whatever the policy visits -- so 'actor' and 'critic' are
normalised with the project's existing observation bounds
(ATTITUDE_BOUND, VEL_*_BOUND, ANG_VEL_BOUND, REL_POS_BOUND, REL_VEL_BOUND)
instead.

This is a real difference, not a cosmetic one: the V3b estimator saw inputs
scaled by dataset statistics and V4's sees inputs scaled by clipping bounds
far wider than the data spans. If V4's estimation accuracy comes out worse
than V3b's held-out numbers, this is the first thing to check, before
concluding anything about joint training.


FRAME -- world by default, body available
------------------------------------------------------------------------------
The paper estimates in the BODY frame and argues that doing so avoids
accumulating drift. Every stage of this project so far has worked in the
WORLD frame, and the finite-difference identity the ArUco estimator relies on
(d(d)/dt = delta_v, see ArucoLanderAviary) holds exactly there.

`frame='world'` is therefore the default, so that V4's estimation error is
directly comparable with the V3a and V3b numbers already reported. Passing
`frame='body'` switches both the measurement and the target into body axes by
a single rotation, which is the paper's configuration.

That rotation is the simple interpretation -- vectors expressed in body axes,
R_body.T @ x. It does NOT add the -omega x r term that a full kinematic
body-frame velocity would carry. The paper does not specify which it means;
this is recorded as an assumption, not a derivation.

Do not change the frame in the same run as anything else. It is its own
variable.


REWARD -- unchanged here on purpose
------------------------------------------------------------------------------
This environment emits the inherited shaping reward, untouched. The
active-perception term

    r_active_t = -alpha * clip(beta * (L_est_{t+1} - tau), 0, 1)

is NOT computed here, and cannot be: L_est is produced by the estimation head
inside the policy, which the environment has no access to. It is added to the
rollout rewards in train_shin_v4.py, before advantage computation, using
L_est at t+1 for the reward at t -- which is what makes it a statement about
the causal effect of the action.

Putting an approximation of it here using L_est_t instead would be a
different reward, silently.


ACTION -- unchanged, 3-D
------------------------------------------------------------------------------
The paper's action is [vx, vy, vz, omega_z]. This project's is the 3-D
PID-mediated dp = 0.1 * c_t inherited from the Lander.AI rebuild, with no yaw
control -- which is also why the camera is nadir rather than pitched 60
degrees forward as in the paper.

Changing the action space would move the low-level control interface at the
same time as the estimator and the algorithm, and would invalidate every
comparison back to Strand B. It stays 3-D. Documented deviation.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

from envs.ArucoLanderAviary import ArucoLanderAviary


ACTOR_DIM = 13      # d_meas(3) valid(1) vel(3) rpy(3) angvel(3)
CRITIC_DIM = 15     # rpy(3) vel(3) angvel(3) d_true(3) dv_true(3)
TARGET_DIM = 6      # d_true(3) dv_true(3), physical units


class ShinLanderAviary(ArucoLanderAviary):
    """ArucoLanderAviary with a Dict observation for asymmetric,
    jointly-trained recurrent policies."""

    VALID_FRAMES = ('world', 'body')

    def __init__(self, *args, frame: str = 'world',
                 verbose_source: bool = True, **kwargs):

        if frame not in self.VALID_FRAMES:
            raise ValueError(
                f'frame must be one of {self.VALID_FRAMES}, got {frame!r}')
        self.FRAME = frame

        # The parent's constant-velocity estimator is kept live as a passive
        # baseline only -- it is logged in info[] and never reaches 'actor'.
        # Forcing it here rather than accepting it as an argument removes the
        # ambiguity that LSTMLanderAviary's docstring records: what the actor
        # consumes must not be inferable from estimator_mode.
        if kwargs.get('estimator_mode') not in (None, 'predict'):
            raise ValueError(
                "ShinLanderAviary manages estimator_mode internally (always "
                "'predict', so the CV baseline stays live for logging). The "
                "actor observation is the raw measurement plus validity and "
                "is not selectable.")
        kwargs['estimator_mode'] = 'predict'

        # The actor cannot receive privileged state under any code path.
        # Unlike V2/V3a this is not optional: there is no configuration of
        # V4 in which the privileged vector would be a legitimate actor
        # observation, because the privileged state is already provided to
        # the critic through its own key.
        kwargs['strict_no_privileged'] = True

        super().__init__(*args, **kwargs)

        if verbose_source:
            print(f'[ShinLanderAviary] Dict obs '
                  f'(actor {ACTOR_DIM}-D | critic {CRITIC_DIM}-D privileged | '
                  f'target {TARGET_DIM}-D)  frame={self.FRAME}  '
                  f'| CV baseline live (passive)')

    # ==================================================================
    # SPACES
    # ==================================================================

    def _observationSpace(self):
        return spaces.Dict({
            'actor': spaces.Box(low=-1.0, high=1.0,
                                shape=(ACTOR_DIM,), dtype=np.float32),
            'critic': spaces.Box(low=-1.0, high=1.0,
                                 shape=(CRITIC_DIM,), dtype=np.float32),
            # Physical units, so deliberately unbounded. This key is a
            # LABEL, not an input -- the policy must never read it to act.
            'target': spaces.Box(low=-np.inf, high=np.inf,
                                 shape=(TARGET_DIM,), dtype=np.float32),
        })

    # ==================================================================
    # FRAME HELPER
    # ==================================================================

    def _toFrame(self, vec_world, R_body):
        """Express a world-frame vector in the configured frame.

        Simple rotation into body axes; see the module docstring for what
        this does and does not mean.
        """
        if self.FRAME == 'world':
            return vec_world
        return R_body.T @ vec_world

    # ==================================================================
    # OBSERVATION
    # ==================================================================

    def _trueRelativeState(self, state, R_body):
        """Ground truth [d, delta_v]. Critic and auxiliary label ONLY.

        This is the one place in this class where ground truth is read for
        anything other than info[]. It must never be reachable from the
        'actor' key.
        """
        d_true = self._getLandingTargetPos() - state[0:3]
        dv_true = self._getPlatformVel() - state[10:13]
        return (self._toFrame(d_true, R_body),
                self._toFrame(dv_true, R_body))

    def _computeObs(self):
        # The platform may not exist yet during super().reset(). Materialise
        # it rather than ever falling back to the parent's privileged vector
        # -- which would also be the wrong SHAPE here and would fail loudly,
        # but relying on a shape mismatch to catch a privilege leak is not a
        # safeguard.
        if self.PLAT_ID is None:
            self._updatePlatform(step_offset=0)
            if self.PLAT_ID is None:
                raise RuntimeError(
                    'ShinLanderAviary: platform could not be created, so '
                    '_computeObs() cannot build the observation. Refusing.')

        det = self._refreshDetection()

        state = self._getDroneStateVector(0)
        R_body = np.array(
            p.getMatrixFromQuaternion(state[3:7]), dtype=float).reshape(3, 3)

        theta = np.asarray(state[7:10], dtype=np.float64)    # rpy
        v = np.asarray(state[10:13], dtype=np.float64)       # velocity
        omega = np.asarray(state[13:16], dtype=np.float64)   # angular vel

        def clip_norm(x, bound):
            return np.clip(x, -bound, bound) / bound

        # ---- actor: raw measurement + validity + proprioception --------
        # d_meas is None when nothing was detected this step. Zero is the
        # neutral, obviously-wrong value, exactly as V3b's estimator input
        # was built -- the validity bit is what distinguishes it from a real
        # measurement that happens to be near zero.
        detected = bool(det['detected'])
        d_meas_w = (np.zeros(3) if (not detected or det['d_meas'] is None)
                    else np.asarray(det['d_meas'], dtype=np.float64))
        d_meas = self._toFrame(d_meas_w, R_body) if detected else np.zeros(3)

        theta_n = clip_norm(theta, self.ATTITUDE_BOUND)
        v_n = np.array([clip_norm(v[0], self.VEL_XY_BOUND),
                        clip_norm(v[1], self.VEL_XY_BOUND),
                        clip_norm(v[2], self.VEL_Z_BOUND)])
        omega_n = clip_norm(omega, self.ANG_VEL_BOUND)

        actor = np.concatenate([
            clip_norm(d_meas, self.REL_POS_BOUND),   # 3
            [1.0 if detected else 0.0],              # 1
            v_n,                                     # 3
            theta_n,                                 # 3
            omega_n,                                 # 3
        ]).astype('float32')

        # ---- critic: o_priv = [u_t, s_rel_t] ---------------------------
        d_true, dv_true = self._trueRelativeState(state, R_body)
        critic = np.concatenate([
            theta_n,                                        # 3
            v_n,                                            # 3
            omega_n,                                        # 3
            clip_norm(d_true, self.REL_POS_BOUND),          # 3
            clip_norm(dv_true, self.REL_VEL_BOUND),         # 3
        ]).astype('float32')

        # ---- target: physical units, unnormalised ----------------------
        target = np.concatenate([d_true, dv_true]).astype('float32')

        assert actor.shape == (ACTOR_DIM,)
        assert critic.shape == (CRITIC_DIM,)
        assert target.shape == (TARGET_DIM,)

        return {'actor': actor, 'critic': critic, 'target': target}

    # ==================================================================
    # INFO
    # ==================================================================

    def _computeInfo(self):
        info = super()._computeInfo()
        if self.PLAT_ID is None:
            return info

        state = self._getDroneStateVector(0)
        R_body = np.array(
            p.getMatrixFromQuaternion(state[3:7]), dtype=float).reshape(3, 3)
        d_true, dv_true = self._trueRelativeState(state, R_body)

        info.update({
            'obs_frame': self.FRAME,
            'v4_d_true_norm': float(np.linalg.norm(d_true)),
            'v4_dv_true_norm': float(np.linalg.norm(dv_true)),
            # The inherited CV estimator's error on the SAME frames, so V4
            # can be compared against V3a's predictor without a second run.
            # Named cv_* to match compare_error_distributions.py.
            'cv_err_pos_norm': info.get('est_err_pos_norm'),
            'cv_err_vel_norm': info.get('est_err_vel_norm'),
        })
        return info

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # The parent already clears the estimator and forces one fresh
        # detection after the platform is re-sampled; its returned obs is
        # built by THIS class's _computeObs, so it is already the Dict. No
        # extra recompute is needed here, and adding one would advance the
        # detection epoch a second time for no reason.
        return obs, info
