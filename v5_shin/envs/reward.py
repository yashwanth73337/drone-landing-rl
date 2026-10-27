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

TouchdownReward (SPEC D21, agreed 25 Oct 2026; needs the D20 hold, hold_steps > 0):
  flight            Table III terms as above (+ flare term in v3)
  soft touchdown    0 (the step where the hold starts)
  each hold step    -min(1, |v_drone - v_pad_surface|)   (Table III terms OFF: D3 would
                                                           charge 0.25 per step for resting)
  hold success      +10
  hard_landing      v1, v3: -10;  v2: -min(10, 5 (|impact v_z| - 1))
  bounce / slide_off / tip_over / not_still / crash / tilt / drift   -10
  v3 flare (flight, horizontal distance <= 1.5 m):
    -min(1, max(0, -v_z - v_ref)),  v_ref = 0.8 + 1.0 * max(0, h),  h = CoM height above the
    pad top (= -dz). The allowed descent speed shrinks linearly with height (bio-inspired:
    descent speed proportional to height, as in landing honeybees and pigeons).

  v4 (SPEC D22): v2 flight terms (no flare) with graded hold terms
    soft touchdown    +2 * max(0, 1 - e / 0.75),  e = max(|x|, |y|) of the CoM in the pad frame
    each hold step    +0.5 * (1 - min(1, |v_rel|))           (never negative)
    hard_landing      -min(10, 5 + 5 (|impact v_z| - 1))      (always worse than a failed hold)
    bounce / slide_off / tip_over / not_still   -5
    hold success +10; crash / tilt / drift -10 (as before)

  v5 (SPEC D27): v4 with a gentler hard-landing slope so the penalty keeps a gradient up to 3 m/s
    (v4 reached its -10 floor at 2 m/s, where most noisy training landings are):
    hard_landing      -min(10, 5 + 2.5 (|impact v_z| - 1))

  v6 (SPEC D30): v5 + the v3 flare term in flight (dense signal before contact, no floor), because
    the vision policy lands at 3.3-3.7 m/s, beyond v5's -10 floor at 3 m/s.
"""
import numpy as np

from .landing_sim import V_IMP
from .platform import PAD_SIZE

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


# ---- SPEC D21: touchdown-aware reward ------------------------------------------------
HOLD_FAIL_OUTCOMES = ("bounce", "slide_off", "tip_over", "not_still")
FLARE_V0 = 0.8          # m/s, allowed descent speed at the pad top
FLARE_K = 1.0           # 1/s, + 1 m/s of allowed descent per metre of height
FLARE_D = 1.5           # m, horizontal distance within which the flare term applies
W_FLARE = 1.0
HARD_K = 5.0            # v2: penalty per m/s of impact speed above V_IMP
VARIANTS = ("v1", "v2", "v3", "v4", "v5", "v6")
HOLD_GRADED = ("v4", "v5", "v6")    # variants with the graded hold terms (D22)
FLARE_VARIANTS = ("v3", "v6")       # variants with the flare term in flight
HARD_K_V5 = 2.5               # v5: penalty per m/s above V_IMP (floor -10 reached at 3 m/s)
# v4 (D22)
CENTRE_BONUS = 2.0      # at the pad centre, falling linearly to 0 at the pad edge
HOLD_STEP_GAIN = 0.5    # per hold step when still
R_HOLD_FAIL_V4 = -5.0


def flare_term(d, dz, vz):
    """Unweighted v3 term (<= 0). d: horizontal distance (m), dz = z_pad_top - z_CoM (D4),
    vz: drone vertical velocity (m/s, + up)."""
    if d > FLARE_D:
        return 0.0
    v_ref = FLARE_V0 + FLARE_K * max(0.0, -dz)
    return -min(1.0, max(0.0, -vz - v_ref))


def hard_landing_reward(variant, impact_vz):
    if variant == "v2":
        return -min(10.0, HARD_K * (abs(float(impact_vz)) - V_IMP))
    if variant == "v4":
        return -min(10.0, -R_HOLD_FAIL_V4 + HARD_K * (abs(float(impact_vz)) - V_IMP))
    if variant in ("v5", "v6"):
        return -min(10.0, -R_HOLD_FAIL_V4 + HARD_K_V5 * (abs(float(impact_vz)) - V_IMP))
    return R_CRASH


def centre_bonus(rel_pos_pad):
    """v4 touchdown bonus from the CoM's pad-frame offset (x, y) at touchdown."""
    e = max(abs(float(rel_pos_pad[0])), abs(float(rel_pos_pad[1])))
    return CENTRE_BONUS * max(0.0, 1.0 - e / (PAD_SIZE / 2))


def hold_step_reward(variant, vrel):
    if variant in HOLD_GRADED:
        return HOLD_STEP_GAIN * (1.0 - min(1.0, vrel))
    return -min(1.0, vrel)


class TouchdownReward(ShinReward):
    """Reward for the D20 hold task. variant: 'v1' (terminal changes only), 'v2' (+ graded
    hard landing), 'v3' (+ flare term before contact), 'v4' (v2 + graded hold terms, D22)."""

    def __init__(self, variant, vz_penalty="literal"):
        super().__init__(vz_penalty)
        assert variant in VARIANTS, variant
        self.variant = variant

    def __call__(self, env, info):
        d, dz, vz = geometry(env.sim)
        terms = shaping_terms(self.d_prev, d, self.adz_prev, abs(dz), dz, vz,
                              info["action_cmd"][3], self.vz_penalty)
        if self.variant in FLARE_VARIANTS:
            terms["flare"] = W_FLARE * flare_term(d, dz, vz)
        self.d_prev, self.adz_prev = d, abs(dz)
        outcome = info.get("outcome")
        if outcome is None and info.get("phase") == "hold":
            if info.get("hold_t", 0) == 0:                     # the touchdown step
                r = centre_bonus(env.sim._td["rel_pos_pad"]) if self.variant in HOLD_GRADED else 0.0
            else:
                r = hold_step_reward(self.variant, env.sim.hold_vrel())
            info["reward_terms"] = dict(hold=r)
            return r
        info["reward_terms"] = terms
        if outcome == "success":
            return R_SUCCESS
        if outcome == "hard_landing":
            return hard_landing_reward(self.variant, info["rel_vel"][2])
        if outcome in HOLD_FAIL_OUTCOMES and self.variant in HOLD_GRADED:
            return R_HOLD_FAIL_V4
        if outcome in CRASH_OUTCOMES or outcome in HOLD_FAIL_OUTCOMES:
            return R_CRASH
        return sum(terms.values())         # flight, including the timeout step
