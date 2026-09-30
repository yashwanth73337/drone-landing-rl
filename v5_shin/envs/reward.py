"""Shaping + terminal reward (V5 Block 9; Shin et al. Sec. III-D4, Table III; SPEC §6).

  r_t = +10                          success
        -10                          crash (ground / platform side / tilt) or drift
        sum_i w_i r_i  (+ r_active)  otherwise, including the truncated (timeout) step

Table III terms, [x]_a^b = clip(x, a, b):
  lateral progress   r1 = [d_{t-1} - d_t]_{-1}^{1}                       w 1.0
  vertical progress  r2 = [|dz_{t-1}| - |dz_t|]_{-1}^{1} / max(d_t, 1)    w 1.0
  vertical speed     r3 = -[ v_z + 0.5]_0^inf   (D3 'literal', DEFAULT: the equation as printed;
                                                 penalises anything slower than a 0.5 m/s descent)
                          -[-v_z - 0.5]_0^inf   ('prose': penalise descent > 0.5 m/s; ablation)
                                                                         w 0.5
  undershoot         r4 = -1[dz_t > 0] dz_t                              w 1.0
  yaw rate           r5 = -|w_z|, w_z = COMMANDED yaw rate (rad/s)       w 2.0
d = horizontal CoM-to-pad-centre distance; dz = z_pad_top - z_CoM (D4: negative while
above); v_z = drone world vertical velocity (true state).
r_active (Sec. III-C) is added by the trainer in Block 13 (it needs L_est at t+1).
"""
import numpy as np

W = dict(lateral=1.0, vertical=1.0, vz=0.5, undershoot=1.0, yaw=2.0)
R_SUCCESS, R_CRASH = 10.0, -10.0
CRASH_OUTCOMES = ("crash_ground", "crash_platform", "tilt", "drift")


def geometry(sim):
    """(d, dz, v_z): horizontal distance, z_pad_top - z_CoM, drone vertical velocity."""
    s = sim.quad.state()
    rel = sim.plat.pad_center() - s["pos"]
    return float(np.hypot(rel[0], rel[1])), float(rel[2]), float(s["v"][2])


def shaping_terms(d_prev, d, adz_prev, adz, dz, vz, yaw_cmd, vz_penalty="literal"):
    """WEIGHTED Table III terms (their sum is the shaping reward).
    adz_prev = |dz_{t-1}|, adz = |dz_t|; dz is signed (D4) for the undershoot term."""
    if vz_penalty == "prose":
        r3 = -max(0.0, -vz - 0.5)
    elif vz_penalty == "literal":
        r3 = -max(0.0, vz + 0.5)
    else:
        raise ValueError(vz_penalty)
    r = dict(
        lateral=float(np.clip(d_prev - d, -1.0, 1.0)),
        vertical=float(np.clip(adz_prev - adz, -1.0, 1.0)) / max(d, 1.0),
        vz=float(r3),
        undershoot=-float(dz) if dz > 0 else 0.0,
        yaw=-abs(float(yaw_cmd)),
    )
    return {k: W[k] * v for k, v in r.items()}


class ShinReward:
    """reward_fn(env, info) for ShinLandingEnv, with reset(env). info['reward_terms'] holds
    the weighted shaping terms; on non-terminal steps they sum to the reward."""

    def __init__(self, vz_penalty="literal"):     # D3, decided 30 Sep (second decision)
        assert vz_penalty in ("prose", "literal")
        self.vz_penalty = vz_penalty

    def reset(self, env):
        d, dz, _ = geometry(env.sim)
        self.d_prev, self.adz_prev = d, abs(dz)

    def __call__(self, env, info):
        d, dz, vz = geometry(env.sim)
        terms = shaping_terms(self.d_prev, d, self.adz_prev, abs(dz), dz, vz,
                              info["action_cmd"][3], self.vz_penalty)
        self.d_prev, self.adz_prev = d, abs(dz)
        info["reward_terms"] = terms
        outcome = info.get("outcome")
        if outcome == "success":
            return R_SUCCESS
        if outcome in CRASH_OUTCOMES:
            return R_CRASH
        return sum(terms.values())         # includes the timeout (truncated) step
