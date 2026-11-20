"""v7 S2a: the V7 TEACHER (privileged expert pilot), 20 Nov 2026.

The V6 teacher (v6_drop/experts/teacher.py) adapted to the car-like V7 pad and the S500.
Tuned in Claude's sandbox on tuning seeds 31000+ (headless, top speed 9 m/s, 3 m/s^2):

  version                                   straight  wave  corners  random   (safe %)
  V6 logic with the V7 limits (100 each)        86       6     17       66
  this teacher (100 each)                      100      99     98       96   (393/400)

Changes (all decided by Yashwanth on 20 Nov: D1 to D4):
  1. Pad motion is LEARNED from zero each flight: velocity over 0.5 s (V6) and, new, the
     speed change (acceleration, incl. the sideways part v x w) over PAD_A_TAU = 0.2 s (D1;
     0.5 s gave 88 %, waves 54 %: the swing changed before it was learned). The filter's
     own lag is added back, plus a LEAD of T_LEAD = 0.3 s (the drone reaches a new speed
     ~0.3 s late; without it the drone trailed a wave by ~0.5 m):
         v_aim = v_est + (0.5 s + T_LEAD) x a_est
  2. Height = the TRUE lowest point of the S500 (skids) above the pad top. A leaning drone
     lowers one skid end ~2 cm at 10 deg; with the level formula the skid grazed the pad
     while "hovering 3 cm up" (all early tip-overs were such motors-on touches).
  3. Limits from the env: speed scale env.act_scale (+-10 m/s), capped by LENGTH (keeps a
     diagonal chase straight); S500 standing height (V6 used the old legs, 0.152 m).
  4. Drop rule "safe30m" (V6, 30 Oct: within 0.30 m, feet 1.5 to 6 cm, mismatch < 0.30 m/s,
     lean < 8 deg, predicted impact <= 0.8 m/s) (D2). The V6 teacher rule allows 4 deg only;
     at 8.5 m/s the drone leans 4.7 deg against air drag and never dropped.
  5. WAIT and ABORT (D4). "Settled" = the learned along-road acceleration is below
     A_CALM = 1 m/s^2 (turning is allowed: waves never stop turning). While not settled the
     teacher holds the skids H_WAIT = 0.25 m up and never drops (the pad brakes before every
     corner, then turns suddenly). Below 0.12 m with a speed mismatch above 0.35 m/s: climb.

Unchanged from V6: approach line K_BACK x height behind the pad along the pad heading (it
turns with the pad), pull to the line <= V_REL_MAX = 3 m/s relative to the pad, integral
near the pad, descent rules, go-around, yaw to the pad heading.
"""
import numpy as np

from v6_drop.envs import lmf2_params as P
from v6_drop.envs.lee_controller import yaw_rotation
from v6_drop.experts import drop_gates as G
from v6_drop.experts import teacher as T6
from v6_drop.experts.teacher import (E_DESC0, E_DESC_K, GO_AROUND_H, H_DROP, H_MIN, I_MAX, I_ZONE,
                                     K_BACK, K_I, K_XY, K_YAW, K_Z, PAD_V_TAU, T_BRAKE, V_REL_MAX,
                                     VZ_DOWN, VZ_LIFT, VZ_SLOW0, VZ_UP, wrap)

from ..envs.platform_v7 import PAD_TOP_Z
from ..envs.s500 import REST_HEIGHT, lowest_z

PAD_A_TAU = 0.2            # s, how fast the teacher "learns" the pad's speed change (D1)
T_LEAD = 0.3               # s, aim at the pad velocity this far AHEAD (the drone's own speed lag)
GATE = "safe30m"           # drop rule (D2), v6_drop/experts/drop_gates.py
A_CALM = 1.0               # m/s^2: "settled" when the learned along-road |a| is below this
H_WAIT = 0.25              # m, lowest-point height to wait at while the pad is not settled
H_ABORT, V_ABORT = 0.12, 0.35   # below H_ABORT m with a speed mismatch above V_ABORT m/s: climb
VZ_ABORT = 0.3             # m/s


def feet_height(pos_z):
    """Skid height above the pad top for a level S500 (0.22 m standing height)."""
    return float(pos_z - PAD_TOP_Z - REST_HEIGHT)


class TeacherV7(T6.Teacher):
    """act(env) with a LandingEnvV7. Same outputs as the V6 teacher."""

    def __init__(self, gate=GATE, t_lead=T_LEAD, a_tau=PAD_A_TAU, low_point=True,
                 a_calm=A_CALM):
        self.a_calm = a_calm
        self.low_point = bool(low_point)
        self.t_lead = float(t_lead)
        self.a_tau = float(a_tau)
        super().__init__(gate=gate)

    def reset(self):
        super().reset()
        self.pad_a_est = np.zeros(3)

    def pad_velocity_pred(self, s):
        """Update the learned pad velocity and acceleration; return the lag-corrected
        prediction of the pad's velocity now."""
        k_v = P.POLICY_DT / (PAD_V_TAU + P.POLICY_DT)
        k_a = P.POLICY_DT / (self.a_tau + P.POLICY_DT)
        self.pad_v_est = self.pad_v_est + k_v * (s["pad_v"] - self.pad_v_est)
        self.pad_a_est = self.pad_a_est + k_a * (s["pad_a"] - self.pad_a_est)
        return self.pad_v_est + (PAD_V_TAU + self.t_lead) * self.pad_a_est

    def act(self, env):
        """Returns (action dict, info dict). Reads the TRUE state of env.world."""
        world = env.world
        scale = env.act_scale
        s = world.true_state()
        pos, v, R = s["pos"], s["v"], s["R"]
        v_pad = self.pad_velocity_pred(s)
        rel = s["pad_center"] - pos
        d = float(np.hypot(rel[0], rel[1]))
        hf = float(lowest_z(R, pos) - PAD_TOP_Z) if self.low_point else feet_height(pos[2])
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        tilt = float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))

        # horizontal: learned pad velocity + pull to the approach line (+ integral)
        u = np.array([np.cos(s["pad_psi"]), np.sin(s["pad_psi"])])
        target = s["pad_center"][:2] - u * K_BACK * max(hf, 0.0)
        err = target - pos[:2]
        e = float(np.linalg.norm(err))
        pull = K_XY * err
        n = np.linalg.norm(pull)
        if n > V_REL_MAX:
            pull *= V_REL_MAX / n
        if hf < I_ZONE and e < I_ZONE:
            self.i_xy = self.i_xy + err * P.POLICY_DT
            n_i = np.linalg.norm(K_I * self.i_xy)
            if n_i > I_MAX:
                self.i_xy *= I_MAX / n_i
        v_xy = v_pad[:2] + pull + K_I * self.i_xy
        n = np.linalg.norm(v_xy)
        if n > scale[0]:
            v_xy *= scale[0] / n

        # vertical (V6): descend only near the line; climb if low and off the line
        if e < E_DESC0 + E_DESC_K * hf:
            vz = K_Z * (H_DROP - hf)
            phase = "hover" if hf < 0.15 else ("glide" if d < 2.0 else "follow")
        elif hf < GO_AROUND_H:
            vz, phase = K_Z * (GO_AROUND_H - hf) + 0.2, "go-around"
        else:
            vz, phase = 0.0, "follow"
        vz_down = min(VZ_DOWN, max(VZ_SLOW0, (hf - H_DROP) / T_BRAKE))
        vz = float(np.clip(vz, -vz_down, VZ_UP))
        if hf < H_MIN:
            vz = max(vz, VZ_LIFT)
        # "settled": the pad is not braking or speeding up hard (along-road part of the
        # learned acceleration). Turning (sideways part) is allowed: waves never stop turning.
        a_along = float(self.pad_a_est[:2] @ u)
        settled = self.a_calm is None or abs(a_along) < self.a_calm
        vrel = v - s["pad_v"]
        vrel_xy = float(np.hypot(vrel[0], vrel[1]))
        if not settled and hf < H_WAIT + 0.05:
            vz = float(np.clip(K_Z * (H_WAIT - hf), -vz_down, VZ_UP))
            phase = "wait"
        if hf < H_ABORT and vrel_xy > V_ABORT:
            vz, phase = VZ_ABORT, "abort"

        yaw_rate = float(np.clip(K_YAW * wrap(s["pad_psi"] - yaw), -scale[3], scale[3]))
        cmd = yaw_rotation(yaw).T @ np.array([v_xy[0], v_xy[1], vz])
        move = np.clip(np.array([*cmd, yaw_rate]) / scale, -1.0, 1.0)

        # drop gate (V6 rules and numbers); never while the pad is not settled
        ok = settled and bool(G.ok(self.gate_cfg, d=d, hf=hf, vxy=vrel_xy, vz=vrel[2], tilt=tilt))
        self.gate = self.gate + 1 if ok else 0
        drop = int(self.gate >= self.gate_cfg["hold"])
        if drop:
            phase = "drop"
        self.phase = phase
        return ({"move": move.astype(np.float32), "drop": drop},
                dict(phase=phase, d=d, hf=hf, e=e, gate=self.gate))
