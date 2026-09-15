"""
LSTMLanderAviary
================

V3b environment. Replaces the hand-coded constant-velocity predictor with the
FROZEN LSTM temporal estimator trained offline by train_estimator.py.

    LanderAIAviary        (frozen -- privileged baseline)
        |
    VisionLanderAviary    (frozen -- V1, camera observes but never feeds)
        |
    ArucoLanderAviary     (frozen -- V2/V3a, constant-velocity prediction)
        |
    LSTMLanderAviary      (this file -- V3b, learned temporal estimation)

ONE VARIABLE CHANGES relative to V3a: how the relative state is propagated
between detections. Constant-velocity dead reckoning becomes a learned
recurrent estimator. Physics, reward, contact logic, action mechanism,
curriculum, camera, marker and the ArUco/solvePnP front-end are all
inherited untouched.


ACTOR OBSERVATION SOURCE IS EXPLICIT
------------------------------------------------------------------------------
`actor_obs_source` selects what the 15-D actor observation is built from, and
nothing else in this class changes it:

    'privileged'  [theta, v, omega, d_TRUE,    delta_v_TRUE]
    'predict'     [theta, v, omega, d_CV,      delta_v_CV]      (V3a)
    'lstm'        [theta, v, omega, d_LSTM,    delta_v_LSTM]    (V3b)

It is printed at construction and published in info['actor_obs_source'] every
step. It must never have to be inferred from which estimator_mode happened to
be set -- that ambiguity is how a diagnostic ends up measuring something other
than what it claims.

BOTH ESTIMATORS ALWAYS RUN, on the same camera stream, every step, regardless
of which one the actor consumes. That is what makes a PASSIVE side-by-side
comparison possible: the unused estimator observes without ever altering the
trajectory.

The shape is 15-D in every case. No validity bit, no estimate age, no hidden
state, no extra dimension reaches the actor -- which is what lets TD3 stay
unchanged and keeps the V3a-vs-V3b comparison clean.

The estimator is FROZEN. It is part of the observation pipeline, not part of
the learned policy. No gradients flow into it during RL. Joint recurrent
training would require sequence-based replay, which TD3 fights, and that is
what eventually forces PPO -- a later stage on purpose.


WHAT THE ESTIMATOR SEES, AND WHAT IT DOES NOT
------------------------------------------------------------------------------
13-D input per control step, exactly as trained (temporal_estimator.py):

    d_meas (3), valid (1), drone velocity (3), rpy (3), angular velocity (3)

During blindness d_meas is [0,0,0] and valid is 0. The previous measurement is
NOT held at the input -- carrying temporal information is the hidden state's
job, and that is precisely the capability being tested.

The validity bit lives INSIDE the estimator and never reaches the actor.

Ground truth is used only for reward, termination, success detection,
curriculum scoring and info[] diagnostics -- never in _computeObs().


HIDDEN STATE
------------------------------------------------------------------------------
Zeroed at every real episode reset, exactly as during training. Never shared
between VecEnv workers. Resetting or sharing it incorrectly would deploy a
different estimator from the one validated offline, and the failure would be
silent.
"""

import numpy as np
import torch

from envs.ArucoLanderAviary import ArucoLanderAviary
from models.temporal_estimator import TemporalEstimator, INPUT_DIM


class LSTMLanderAviary(ArucoLanderAviary):

    ACTOR_OBS_SOURCES = ('privileged', 'predict', 'lstm')

    def __init__(self, *args, estimator_path=None, device='cpu',
                 actor_obs_source='lstm', verbose_source=True, **kwargs):
        if estimator_path is None:
            raise ValueError(
                'LSTMLanderAviary requires estimator_path -- the frozen '
                'estimator checkpoint from train_estimator.py.')
        if actor_obs_source not in self.ACTOR_OBS_SOURCES:
            raise ValueError(
                f'actor_obs_source must be one of '
                f'{self.ACTOR_OBS_SOURCES}, got {actor_obs_source!r}')

        self.ACTOR_OBS_SOURCE = actor_obs_source
        self.ESTIMATOR_PATH = estimator_path
        self._device = torch.device(device)

        ckpt = torch.load(estimator_path, map_location=self._device)
        self._lstm = TemporalEstimator(
            hidden_size=ckpt.get('hidden_size', 128),
            num_layers=ckpt.get('num_layers', 1)).to(self._device)
        self._lstm.load_state_dict(ckpt['state_dict'])
        self._lstm.eval()
        for p in self._lstm.parameters():
            p.requires_grad_(False)      # frozen: no RL gradients, ever

        self._hidden = None
        self._lstm_d = np.zeros(3)
        self._lstm_dv = np.zeros(3)

        # The parent is ALWAYS forced to 'predict' so the constant-velocity
        # estimator propagates during blind steps and is available as a live
        # baseline, no matter what the actor consumes. The parent's
        # estimator_mode therefore no longer controls the actor observation
        # -- actor_obs_source does, and only that.
        #
        # This decoupling is the fix for a real ambiguity: previously,
        # passing estimator_mode='predict' to get a CV baseline also looked
        # like it selected the CV observation for the actor, when in fact
        # _computeObs() used the LSTM regardless.
        if kwargs.get('estimator_mode') not in (None, 'predict'):
            raise ValueError(
                'LSTMLanderAviary manages estimator_mode internally (always '
                "'predict', so the CV baseline stays live). Use "
                'actor_obs_source to choose what the actor sees.')
        kwargs['estimator_mode'] = 'predict'

        super().__init__(*args, **kwargs)

        if verbose_source:
            print(f'[LSTMLanderAviary] actor_obs_source = '
                  f'{self.ACTOR_OBS_SOURCE.upper()}  |  CV and LSTM '
                  f'estimators both live (passive unless selected)')

    # ==================================================================
    # ESTIMATOR INPUT / STEP
    # ==================================================================

    def _buildEstimatorInput(self, detected, d_meas):
        """The same 13-D vector the estimator was trained on."""
        state = self._getDroneStateVector(0)
        d = (np.zeros(3) if (not detected or d_meas is None)
             else np.asarray(d_meas, dtype=np.float64))
        return np.concatenate([
            d,                                              # 3
            [1.0 if detected else 0.0],                     # 1
            np.asarray(state[10:13], dtype=np.float64),     # 3  velocity
            np.asarray(state[7:10], dtype=np.float64),      # 3  rpy
            np.asarray(state[13:16], dtype=np.float64),     # 3  angular vel
        ]).astype(np.float64)

    def _refreshDetection(self):
        """Parent's cached detection, then one LSTM step on a cache miss.

        Piggybacking on the parent's cache keeps the invariant established in
        V1/V3a: exactly one render and one detection per control step, with
        _computeObs() and _computeInfo() consuming the same result. The LSTM
        advances exactly once per control step for the same reason -- a
        recurrent estimator stepped twice per step would silently run at
        double the rate it was trained at.
        """
        stale = (self._det_cached_epoch != self._det_epoch)
        det = super()._refreshDetection()

        if stale:
            x = self._buildEstimatorInput(det['detected'], det['d_meas'])
            assert x.shape == (INPUT_DIM,)
            pred, self._hidden = self._lstm.step(x, self._hidden)
            self._lstm_d = np.asarray(pred[:3], dtype=np.float64)
            self._lstm_dv = np.asarray(pred[3:], dtype=np.float64)

        det['lstm_d'] = self._lstm_d.copy()
        det['lstm_dv'] = self._lstm_dv.copy()
        return det

    # ==================================================================
    # OBSERVATION -- 15-D, identical shape to V3a
    # ==================================================================

    def _computeObs(self):
        if self.PLAT_ID is None:
            # Same strict handling as the parent: materialise the platform
            # rather than ever returning the privileged vector by accident.
            if (getattr(self, 'STRICT_NO_PRIVILEGED', False)
                    and self.ACTOR_OBS_SOURCE != 'privileged'):
                self._updatePlatform(step_offset=0)
                if self.PLAT_ID is None:
                    raise RuntimeError(
                        'LSTMLanderAviary: platform could not be created, '
                        'so _computeObs() would have to fall back to the '
                        'PRIVILEGED observation. Refusing.')
            else:
                return super()._computeObs()

        det = self._refreshDetection()

        # Explicit dispatch. The actor's input source is this line and
        # nothing else.
        if self.ACTOR_OBS_SOURCE == 'privileged':
            # Grandparent's privileged vector. Both estimators have already
            # been advanced above, so they stay in lockstep with the camera
            # stream while observing passively.
            return super(ArucoLanderAviary, self)._computeObs()

        if self.ACTOR_OBS_SOURCE == 'predict':
            d, dv = det['d_est'], det['dv_est']          # constant velocity
        else:
            d, dv = self._lstm_d, self._lstm_dv          # learned estimator

        state = self._getDroneStateVector(0)
        theta = state[7:10]      # proprioceptive
        v = state[10:13]         # proprioceptive
        omega = state[13:16]     # proprioceptive

        def clip_norm(x, bound):
            return np.clip(x, -bound, bound) / bound

        obs = np.concatenate([
            clip_norm(theta, self.ATTITUDE_BOUND),
            np.array([clip_norm(v[0], self.VEL_XY_BOUND),
                      clip_norm(v[1], self.VEL_XY_BOUND),
                      clip_norm(v[2], self.VEL_Z_BOUND)]),
            clip_norm(omega, self.ANG_VEL_BOUND),
            clip_norm(np.asarray(d, dtype=np.float64), self.REL_POS_BOUND),
            clip_norm(np.asarray(dv, dtype=np.float64), self.REL_VEL_BOUND),
        ])
        return obs.astype('float32')

    # ==================================================================
    # INFO -- diagnostics only, both estimators logged side by side
    # ==================================================================

    def _computeInfo(self):
        info = super()._computeInfo()
        if self.PLAT_ID is None:
            return info

        state = self._getDroneStateVector(0)
        d_true = self._getLandingTargetPos() - state[0:3]
        dv_true = self._getPlatformVel() - state[10:13]

        info.update({
            'actor_obs_source': self.ACTOR_OBS_SOURCE,
            'gt_d_norm': float(np.linalg.norm(d_true)),
            'gt_dv_norm': float(np.linalg.norm(dv_true)),
            'gt_dv_x': float(dv_true[0]),
            'gt_dv_y': float(dv_true[1]),
            'gt_dv_z': float(dv_true[2]),
            'lstm_d_x': float(self._lstm_d[0]),
            'lstm_d_y': float(self._lstm_d[1]),
            'lstm_d_z': float(self._lstm_d[2]),
            'lstm_err_dx': float(self._lstm_d[0] - d_true[0]),
            'lstm_err_dy': float(self._lstm_d[1] - d_true[1]),
            'lstm_err_dz': float(self._lstm_d[2] - d_true[2]),
            'lstm_err_pos_norm': float(
                np.linalg.norm(self._lstm_d - d_true)),
            'lstm_err_vel_norm': float(
                np.linalg.norm(self._lstm_dv - dv_true)),
            # The parent's constant-velocity estimate on the SAME frames,
            # so the two can be compared within one episode.
            'cv_err_pos_norm': info.get('est_err_pos_norm'),
            'cv_err_vel_norm': info.get('est_err_vel_norm'),
        })
        return info

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self, seed=None, options=None):
        # Hidden state cleared at the real episode boundary, exactly as in
        # training. Cleared again after the parent's reset because the parent
        # forces one extra detection after re-sampling the platform, and that
        # detection must start from a zero hidden state too.
        self._hidden = None
        self._lstm_d = np.zeros(3)
        self._lstm_dv = np.zeros(3)

        obs, info = super().reset(seed=seed, options=options)

        self._hidden = None
        self._lstm_d = np.zeros(3)
        self._lstm_dv = np.zeros(3)
        self._det_epoch += 1
        self._det_cached_epoch = None

        obs = self._computeObs()
        info = self._computeInfo()
        return obs, info
