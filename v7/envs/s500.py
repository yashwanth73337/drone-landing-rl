"""v7 S1b: the S500-sized drone (shape only), 19 Nov 2026.

Decided 19 Nov (Yashwanth): the lab drone is an S500 frame with a gimbal camera, so V7
uses the S500's size. Only the SHAPE changes (what touches the pad). The flying physics
stay LMF2 (V6): mass 1.24 kg, inertia, motors, Lee controller. We do not know the lab
S500's motors and battery yet, so we do not invent them.

Sizes (v7/assets/s500.urdf):
  from S500 shop listings: wheelbase 0.50 m (motor centres at +-0.177, +-0.177 m),
                           landing gear 0.20 m below the frame
  ASSUMED until measured:  body plate 0.14 x 0.14 x 0.04 m; two skids along x (forward),
                           0.24 m long, at y = +-0.11 m, radius 0.008 m
  standing: body centre REST_HEIGHT = 0.22 m above the surface (V6 legs: 0.152 m)

Tipping rule of thumb (a box on a moving floor): it tips before it slides when the
friction coefficient is above (half the support width) / (centre height). Friction is
0.5 (pad 1.0 x drone 0.5, D24). Sideways: 0.11 / 0.22 = 0.50 (right at the edge);
forward: 0.12 / 0.22 = 0.55 (a little margin). The S1b tests measure what PyBullet does.
"""
import os

import numpy as np
import pybullet as p

from v6_drop.envs import lmf2_params as P
from v6_drop.envs.quad import LMF2Quad, MotorModel
from v6_drop.envs.lee_controller import LeeVelocityController

URDF = os.path.join(os.path.dirname(__file__), "..", "assets", "s500.urdf")

WHEELBASE = 0.50          # m, motor to motor across the diagonal (listings)
MOTOR_XY = WHEELBASE / 2 / np.sqrt(2)   # 0.177 m
GEAR_H = 0.20             # m, frame bottom to ground (listings)
PLATE_HALF_Z = 0.02       # m, ASSUMED
SKID_Y = 0.11             # m, ASSUMED (skids at y = +-0.11)
SKID_HALF_L = 0.12        # m, ASSUMED (0.24 m long, including the rounded ends)
SKID_R = 0.008            # m, ASSUMED
SKID_Z = -(PLATE_HALF_Z + GEAR_H) + SKID_R    # -0.212 m, skid axis below the body centre
REST_HEIGHT = PLATE_HALF_Z + GEAR_H            # 0.22 m, body centre above a surface, standing

# Lowest points for the referee and the feet height: the skid axis end points (body frame);
# the lowest skin is SKID_R below the lowest of them.
SKID_ENDS_B = np.array([[sx * (SKID_HALF_L - SKID_R), sy * SKID_Y, SKID_Z]
                        for sx in (1, -1) for sy in (1, -1)])
PLATE_CORNERS_B = np.array([[sx * 0.07, sy * 0.07, sz * PLATE_HALF_Z]
                            for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)])
ARM_TIPS_B = np.array([[sx * 0.25 / np.sqrt(2), sy * 0.25 / np.sqrt(2), 0.0]
                       for sx in (1, -1) for sy in (1, -1)])


def lowest_z(R, pos):
    """Height of the drone's lowest point (world z) for attitude R and body position pos.
    Skids (axis ends minus the radius), plate corners and arm tips; the skid is a capsule,
    so its lowest point is always an end or on the axis, never lower than an end - SKID_R."""
    skid = (SKID_ENDS_B @ R.T + pos)[:, 2].min() - SKID_R
    rest = np.vstack([PLATE_CORNERS_B, ARM_TIPS_B]) @ R.T + pos
    return float(min(skid, rest[:, 2].min()))


class S500Quad(LMF2Quad):
    """LMF2Quad with the S500 shape (same physics, motors and controller)."""

    def __init__(self, client, rng, gains):
        self.cid = client
        self.body = p.loadURDF(URDF, [0, 0, 1], flags=p.URDF_USE_INERTIA_FROM_FILE,
                               physicsClientId=client)
        p.changeDynamics(self.body, -1, linearDamping=P.LINEAR_DAMPING,
                         angularDamping=P.ANGULAR_DAMPING, physicsClientId=client)
        self.motors = MotorModel(rng)
        self.ctrl = LeeVelocityController(gains)
        self.last_wrench = np.zeros(4)
        self.ext_force = np.zeros(3)
        self.ext_torque = np.zeros(3)
