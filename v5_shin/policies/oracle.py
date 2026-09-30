"""Scripted oracle landing controller (V5 Block 5). PRIVILEGED: it reads the
true platform state from the simulator. Its purpose is to show the task, as
built, is solvable within the action limits before any learning. It is not a
baseline for the paper's comparisons.

Law (world frame, then rotated into the heading frame):
  lead      rel_p* = rel_p + tau_lead * rel_v            (controller lag ~0.3-0.4 s)
  lateral   v_xy   = v_pad_xy + k_xy * rel_p*_xy          (feedforward + P)
  vertical  descend at v_down(h) only while |rel_p*_xy| < gate(h), else hold altitude
  yaw       rate 0 (commands are rotated into the heading frame, so yaw is free)
"""
import numpy as np

# Action limits (SPEC §5, [unspecified] -> accepted)
VXY_MAX = 10.0
VZ_MAX = 3.0
YAW_RATE_MAX = np.pi / 3


class OracleController:
    def __init__(self, k_xy=1.2, tau_lead=0.35, v_down_max=1.5, v_down_min=0.4,
                 gate_min=0.25, gate_slope=0.25):
        self.k_xy = k_xy
        self.tau_lead = tau_lead
        self.v_down_max = v_down_max
        self.v_down_min = v_down_min
        self.gate_min = gate_min
        self.gate_slope = gate_slope

    def __call__(self, rel_p, rel_v, v_pad, yaw):
        """rel_p, rel_v: platform minus drone (world); v_pad: platform velocity (world);
        yaw: drone yaw (rad). Returns the heading-frame command [vx, vy, vz, yaw_rate]."""
        lead = rel_p + self.tau_lead * rel_v
        v_xy = v_pad[:2] + self.k_xy * lead[:2]
        n = np.linalg.norm(v_xy)
        if n > VXY_MAX:
            v_xy *= VXY_MAX / n
        h = -rel_p[2]                                   # height of the CoM above the pad top
        gate = self.gate_min + self.gate_slope * h
        if np.linalg.norm(lead[:2]) < gate:
            vz = -np.clip(0.5 * h, self.v_down_min, self.v_down_max)
        else:
            vz = float(np.clip(0.8 * (max(h, 1.0) - h), -VZ_MAX, VZ_MAX))   # hold, min 1 m
        c, s = np.cos(yaw), np.sin(yaw)
        vx_h = c * v_xy[0] + s * v_xy[1]                # world -> heading frame
        vy_h = -s * v_xy[0] + c * v_xy[1]
        return np.array([vx_h, vy_h, vz, 0.0])


def random_policy(rng, hold_steps=1):
    """Uniform random heading-frame commands within the action limits, each held for
    hold_steps policy steps (reachability diagnostic)."""
    state = {"k": 0, "a": None}

    def act(*_):
        if state["k"] % hold_steps == 0:
            state["a"] = rng.uniform([-VXY_MAX, -VXY_MAX, -VZ_MAX, -YAW_RATE_MAX],
                                     [VXY_MAX, VXY_MAX, VZ_MAX, YAW_RATE_MAX])
        state["k"] += 1
        return state["a"]
    return act
