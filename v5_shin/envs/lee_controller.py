"""Lee geometric velocity controller: numpy port of AerialGym's
LeeVelocityController + BaseLeeController.compute_body_torque (logic only,
no code copied). Command: [vx, vy, vz] in the vehicle (heading, yaw-only)
frame, plus yaw_rate. Output: body wrench [Fz, tau_x, tau_y, tau_z].

Note (matches AerialGym): the velocity loop is mass-scaled (F = m*a), while
the attitude loop outputs raw torques (N m), NOT inertia-scaled. The Table II
attitude gains are therefore tied to the LMF2 inertia.
"""
import numpy as np

from . import lmf2_params as P


def _vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def euler_xyz_from_R(R):
    """roll, pitch, yaw for R = Rz(yaw) Ry(pitch) Rx(roll) (PyBullet/AerialGym)."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def yaw_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class LeeVelocityController:
    def __init__(self, gains, mass=P.MASS, inertia=P.INERTIA):
        self.K_vel = np.asarray(gains["K_vel"], float)
        self.K_rot = np.asarray(gains["K_rot"], float)
        self.K_angvel = np.asarray(gains["K_angvel"], float)
        self.mass = float(mass)
        self.J = np.diag(inertia)
        self.gravity = np.array([0.0, 0.0, -P.G])

    def compute(self, R, v_world, omega_body, cmd):
        """R: body->world rotation; v_world: m/s; omega_body: rad/s;
        cmd: [vx, vy, vz, yaw_rate] (vehicle frame). Returns [Fz, tx, ty, tz]."""
        roll, pitch, yaw = euler_xyz_from_R(R)

        # Velocity loop (vehicle frame -> world)
        v_sp = yaw_rotation(yaw) @ np.asarray(cmd[:3], float)
        accel = self.K_vel * (v_sp - v_world)
        F = (accel - self.gravity) * self.mass
        thrust = float(F @ R[:, 2])

        # Desired orientation: b3 along F, heading = current yaw
        b3 = F / np.linalg.norm(F)
        tmp = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        b2 = np.cross(b3, tmp)
        b2 /= np.linalg.norm(b2)
        b1 = np.cross(b2, b3)
        Rd = np.column_stack([b1, b2, b3])

        # Desired body rates from the euler yaw rate
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        yr = float(cmd[3])
        w_d = np.array([-sp * yr, sr * cp * yr, cr * cp * yr])
        w_d[2] = np.clip(w_d[2], -P.MAX_YAW_RATE, P.MAX_YAW_RATE)

        # Attitude loop
        RtRd = R.T @ Rd
        e_R = 0.5 * _vee(RtRd.T - RtRd)
        e_w = omega_body - RtRd @ w_d
        ff = np.cross(omega_body, self.J @ omega_body)
        torque = -self.K_rot * e_R - self.K_angvel * e_w + ff
        return np.array([thrust, *torque])
