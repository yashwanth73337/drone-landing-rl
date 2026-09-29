"""
temporal_estimator.py
=====================

LSTM temporal estimator for V3b. Replaces the hand-coded constant-velocity
predictor with a learned recurrent estimator, trained OFFLINE by supervised
regression and then FROZEN.

Keeping it frozen is what lets TD3 stay completely unchanged: the estimator
becomes part of the observation pipeline, exactly where the constant-velocity
predictor sat, and the actor still receives the same 15-D vector. Joint
recurrent RL would require sequence-based replay, which TD3 fights and which
is what eventually forces PPO -- deferred to a later stage on purpose.


THIS IS AN ADAPTATION OF SHIN ET AL., NOT A REPRODUCTION
------------------------------------------------------------------------------
Shin et al. (IEEE RA-L 2026):
    learned visual / keypoint embedding as the perception front-end
    LSTM temporal estimator
    BODY-frame relative position and velocity
    estimator trained JOINTLY with PPO via an auxiliary loss
    asymmetric privileged critic

This implementation:
    ArUco metric measurement as the perception front-end
    LSTM temporal estimator
    validity signal supplied INSIDE the estimator, never to the actor
    WORLD-frame output, so the existing 15-D TD3 interface is unchanged
    OFFLINE supervised training, estimator FROZEN during RL
    no asymmetric critic, no active-perception reward

Only the recurrent-temporal-estimation idea is shared. Every other component
differs, and the two should not be described as the same system.


INPUT -- 13 dims, per timestep
------------------------------------------------------------------------------
    d_meas              3   raw per-frame ArUco measurement
    valid               1   1.0 if this frame produced a detection
    drone velocity      3   proprioceptive
    attitude (rpy)      3   proprioceptive
    angular velocity    3   proprioceptive

`previous_action` is deliberately EXCLUDED from the primary model. It would
give the LSTM a cue the V3a constant-velocity predictor never received,
weakening the estimator-only comparison. Realised velocity, attitude and body
rates already describe ego-motion after the PID dynamics have acted. The
collection script stores previous_action anyway so it can be tested as a
later ablation without re-collecting.

DURING BLINDNESS: d_meas = [0, 0, 0] and valid = 0. The previous measurement
is NOT held at the input. Carrying temporal information is the hidden state's
job, and handing it a stale measurement would let it avoid learning that.


OUTPUT
------------------------------------------------------------------------------
    s_rel_hat = [d_x, d_y, d_z, dv_x, dv_y, dv_z]   world frame, 6 dims

World frame rather than Shin's body frame so the output drops straight into
the existing observation assembly with no change to the actor interface.


STANDARDISATION
------------------------------------------------------------------------------
Targets are standardised per-component using mean and std computed from the
TRAINING SPLIT ONLY:

    z = (target - train_mean) / train_std

The actor's clipping bounds (REL_POS_BOUND = 5, REL_VEL_BOUND = 4) are NOT
used as loss scales. They are observation-space clipping limits chosen for a
different purpose, and they are far wider than the data actually spans, which
would leave the loss dominated by whichever component happens to have the
larger raw magnitude. Position spans roughly +/-0.5 m and velocity roughly
+/-0.35 m/s here, so equal weighting after standardisation is what gives the
velocity head a fair share of the gradient.

Inputs are standardised the same way, from training statistics only.

At inference the prediction is un-standardised back to physical metres and
m/s, and only then does the existing actor observation normalisation apply --
exactly as in V3a. All reported errors are in physical units.
"""

import numpy as np
import torch
import torch.nn as nn


INPUT_DIM = 13
OUTPUT_DIM = 6

INPUT_FIELDS = ['d_meas_x', 'd_meas_y', 'd_meas_z', 'valid',
                'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw',
                'wx', 'wy', 'wz']
OUTPUT_FIELDS = ['d_x', 'd_y', 'd_z', 'dv_x', 'dv_y', 'dv_z']


class TemporalEstimator(nn.Module):
    """LSTM + MLP head. Small by design -- the input is 13 metric numbers,
    not a 512-dim image embedding, so Shin's 512-unit hidden state would be
    heavily over-parameterised here. Sizes below are OUR choice and are
    documented as such, not taken from the paper."""

    def __init__(self, hidden_size=128, num_layers=1, head_hidden=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(INPUT_DIM, hidden_size, num_layers,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, OUTPUT_DIM),
        )

        # Standardisation statistics. Registered as buffers so they travel
        # with the checkpoint -- an estimator loaded without its statistics
        # would silently produce garbage in physical units.
        self.register_buffer('in_mean', torch.zeros(INPUT_DIM))
        self.register_buffer('in_std', torch.ones(INPUT_DIM))
        self.register_buffer('out_mean', torch.zeros(OUTPUT_DIM))
        self.register_buffer('out_std', torch.ones(OUTPUT_DIM))
        self.register_buffer('stats_set', torch.zeros(1))

    # ---- statistics -------------------------------------------------
    def set_stats(self, in_mean, in_std, out_mean, out_std):
        """Install standardisation statistics. TRAIN SPLIT ONLY."""
        self.in_mean.copy_(torch.as_tensor(in_mean, dtype=torch.float32))
        self.in_std.copy_(torch.as_tensor(in_std, dtype=torch.float32))
        self.out_mean.copy_(torch.as_tensor(out_mean, dtype=torch.float32))
        self.out_std.copy_(torch.as_tensor(out_std, dtype=torch.float32))
        self.stats_set.fill_(1.0)

    def _check_stats(self):
        if float(self.stats_set.item()) != 1.0:
            raise RuntimeError(
                'TemporalEstimator has no standardisation statistics. '
                'Call set_stats() with TRAINING-split values before use.')

    def standardise_in(self, x):
        return (x - self.in_mean) / self.in_std

    def unstandardise_out(self, z):
        return z * self.out_std + self.out_mean

    # ---- forward ----------------------------------------------------
    def forward(self, x_raw, hidden=None):
        """x_raw: (B, T, 13) in PHYSICAL units. Returns standardised
        predictions (B, T, 6) and the final hidden state."""
        self._check_stats()
        z = self.standardise_in(x_raw)
        out, hidden = self.lstm(z, hidden)
        return self.head(out), hidden

    def predict_physical(self, x_raw, hidden=None):
        """Same, but returns predictions in metres and m/s."""
        z_pred, hidden = self.forward(x_raw, hidden)
        return self.unstandardise_out(z_pred), hidden

    @torch.no_grad()
    def step(self, obs13, hidden=None):
        """Single online timestep, for LSTMLanderAviary.

        obs13: (13,) numpy array in physical units.
        Returns (s_rel_hat (6,) numpy in physical units, hidden).

        Hidden state must be reset to None at every real episode reset, and
        must never be shared between VecEnv workers.
        """
        x = torch.as_tensor(obs13, dtype=torch.float32).view(1, 1, INPUT_DIM)
        pred, hidden = self.predict_physical(x, hidden)
        return pred.view(OUTPUT_DIM).cpu().numpy(), hidden

    def init_hidden(self, batch=1, device=None):
        device = device or next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch, self.hidden_size,
                        device=device)
        return (h, h.clone())


# ===========================================================================
# CONSTANT-VELOCITY BASELINE -- offline reimplementation
# ===========================================================================

def constant_velocity_baseline(d_meas, valid, v_drone, dt):
    """Replays ArucoLanderAviary._updateEstimator's 'predict' mode offline.

    Must match the deployed logic exactly, or the V3a-vs-V3b comparison is
    against a strawman. The three behaviours that matter:

      1. On consecutive detections, delta_v is the finite difference of the
         RELATIVE position. Since d = p_plat - p_drone, that derivative IS
         delta_v -- the drone's own velocity must NOT be subtracted again.
      2. When detection resumes after a gap, the difference is divided by the
         true elapsed time, not by dt, or the velocity is inflated by the gap.
      3. On blind steps, the platform's world velocity is held and delta_v is
         RE-FORMED against the drone's CURRENT measured velocity, because the
         drone is manoeuvring even though the platform is not.

    Returns (d_est (T,3), dv_est (T,3), blind_age (T,)).
    """
    T = len(valid)
    d_est = np.zeros((T, 3))
    dv_est = np.zeros((T, 3))
    blind_age = np.zeros(T, dtype=int)

    cur_d = np.zeros(3)
    cur_dv = np.zeros(3)
    v_plat = np.zeros(3)
    prev_meas = None
    since = 0
    have = False

    for t in range(T):
        if valid[t]:
            if prev_meas is not None:
                gap = (since + 1) * dt
                cur_dv = (d_meas[t] - prev_meas) / gap
            cur_d = d_meas[t].copy()
            prev_meas = d_meas[t].copy()
            v_plat = cur_dv + v_drone[t]
            have = True
            since = 0
        else:
            since += 1
            if have:
                cur_dv = v_plat - v_drone[t]
                cur_d = cur_d + cur_dv * dt
            # before the first detection: neutral zeros, no ground truth

        d_est[t] = cur_d
        dv_est[t] = cur_dv
        blind_age[t] = since

    return d_est, dv_est, blind_age


BLIND_AGE_BINS = [
    ('visible', 0, 0),
    ('blind 1-5', 1, 5),
    ('blind 6-15', 6, 15),
    ('blind 16-30', 16, 30),
    ('blind 31-60', 31, 60),
    ('blind 61-120', 61, 120),
    ('blind >120', 121, 10 ** 9),
]
