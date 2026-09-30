"""LMF2 quad in raw PyBullet: body loading, AerialGym-style motor model,
control allocation, and wrench application (V5 Block 1)."""
import os

import numpy as np
import pybullet as p

from . import lmf2_params as P
from .lee_controller import LeeVelocityController

URDF = os.path.join(os.path.dirname(__file__), "..", "assets", "lmf2.urdf")


class MotorModel:
    """AerialGym rpm-space first-order model, Euler discretisation with the
    'discrete' mixing factor 1/(dt + tau). The thrust constant cancels."""

    def __init__(self, rng, dt=P.PHYSICS_DT):
        self.dt = dt
        self.tau_inc = rng.uniform(*P.TAU_INC_RANGE, size=4)
        self.tau_dec = np.full(4, P.TAU_DEC)
        self.thrust = np.full(4, P.MASS * P.G / 4.0)

    def reset(self, thrust=None):
        self.thrust[:] = P.MASS * P.G / 4.0 if thrust is None else thrust

    def update(self, ref):
        ref = np.clip(ref, P.MOTOR_MIN_THRUST, P.MOTOR_MAX_THRUST)
        err = ref - self.thrust
        tau = np.where(err < 0, self.tau_dec, self.tau_inc)
        a = self.dt / (self.dt + tau)
        s = np.sqrt(self.thrust)
        s = s + a * (np.sqrt(ref) - s)
        self.thrust = s * s
        return self.thrust


class LMF2Quad:
    """Owns one drone body inside an existing PyBullet client."""

    def __init__(self, client, rng, gains):
        self.cid = client
        # URDF_USE_INERTIA_FROM_FILE is essential: without it PyBullet computes
        # inertia from the 0.5 m collision box (0.052 kg m^2, ~4x too large).
        self.body = p.loadURDF(
            URDF, [0, 0, 1],
            flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=client)
        p.changeDynamics(self.body, -1, linearDamping=P.LINEAR_DAMPING,
                         angularDamping=P.ANGULAR_DAMPING, physicsClientId=client)
        self.motors = MotorModel(rng)
        self.ctrl = LeeVelocityController(gains)
        self.last_wrench = np.zeros(4)
        self.ext_force = np.zeros(3)    # world frame, N  (Table II; Block 6)
        self.ext_torque = np.zeros(3)   # body frame, N m (Table II; Block 6)

    # ---- state --------------------------------------------------------------
    def state(self):
        pos, quat = p.getBasePositionAndOrientation(self.body, physicsClientId=self.cid)
        v, w = p.getBaseVelocity(self.body, physicsClientId=self.cid)
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        return dict(pos=np.array(pos), quat=np.array(quat), R=R,
                    v=np.array(v), w_body=R.T @ np.array(w))

    def reset_pose(self, pos, yaw=0.0, v=(0, 0, 0), w_body=(0, 0, 0)):
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.body, pos, quat, physicsClientId=self.cid)
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        p.resetBaseVelocity(self.body, v, R @ np.asarray(w_body, float),
                            physicsClientId=self.cid)
        self.motors.reset()

    # ---- one physics step of control + actuation -----------------------------
    def apply_control(self, cmd):
        s = self.state()
        wrench_ref = self.ctrl.compute(s["R"], s["v"], s["w_body"], cmd)
        thrusts = self.motors.update(P.ALLOCATION_INV @ wrench_ref)
        wrench = P.ALLOCATION @ thrusts
        self.last_wrench = wrench
        R = s["R"]
        f_world = R @ np.array([0.0, 0.0, wrench[0]]) + self.ext_force
        t_world = R @ (wrench[1:] + self.ext_torque)
        # WORLD_FRAME for both: base-link LINK_FRAME torque handling in
        # PyBullet is unreliable, so rotate ourselves.
        p.applyExternalForce(self.body, -1, f_world.tolist(), s["pos"].tolist(),
                             p.WORLD_FRAME, physicsClientId=self.cid)
        p.applyExternalTorque(self.body, -1, t_world.tolist(), p.WORLD_FRAME,
                              physicsClientId=self.cid)


def make_client(gui=False):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setGravity(0, 0, -P.G, physicsClientId=cid)
    p.setTimeStep(P.PHYSICS_DT, physicsClientId=cid)
    return cid
