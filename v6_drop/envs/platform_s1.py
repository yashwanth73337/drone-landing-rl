"""v6_drop B2: the Stage 1 landing platform.

Stage 1 (supervisor, 28 Oct): the platform drives in a STRAIGHT line at a
CONSTANT speed. The speed is drawn once per episode, v ~ U(0, 4) m/s.
No turning, no speed changes (those are Stages 2-4).

Physics (pitfalls from V5, kept on purpose):
  * The platform is a KINEMATIC box (mass 0): our code moves it, forces do not.
  * D19: every physics step the box gets its pose AND its velocity. PyBullet
    then moves it during the step, and integrate() re-pins it to the exact
    straight line. Without the velocity, PyBullet thinks the box is still,
    and friction does not carry a drone resting on it (it slides off the back).
  * D24: PyBullet MULTIPLIES the two bodies' friction values. Pad 1.0 x drone
    0.5 = 0.5 effective (rubber on dry concrete). Defaults 0.5 x 0.5 = 0.25 (icy).

Size and height are the V5 values: 1.5 m square pad, top at z = 1.0 m.
The marker texture is added in B4 (pass visual_shape then); B2 uses a grey box.
"""
import numpy as np
import pybullet as p

from . import lmf2_params as P

PAD_SIZE = 1.5           # m, square pad side (V5)
PAD_TOP_Z = 1.0          # m, height of the pad top above the ground (V5)
V_MAX = 4.0              # m/s, Stage 1 speed range U(0, V_MAX)  [supervisor, 28 Oct]
PAD_FRICTION = 1.0       # D24: lateral friction of the pad body
DRONE_FRICTION = 0.5     # D24: lateral friction of the drone body; effective = 1.0 x 0.5


def set_contact_friction(client, pad_body, drone_body):
    """D24. Call once after both bodies exist."""
    p.changeDynamics(pad_body, -1, lateralFriction=PAD_FRICTION, physicsClientId=client)
    p.changeDynamics(drone_body, -1, lateralFriction=DRONE_FRICTION, physicsClientId=client)


class PlatformS1:
    """Straight line, constant speed. State: x, y (pad centre), psi (heading), v (speed)."""

    def __init__(self, client, visual_shape=None):
        self.cid = client
        half = [PAD_SIZE / 2, PAD_SIZE / 2, PAD_TOP_Z / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
        if visual_shape is None:
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                               rgbaColor=[0.6, 0.6, 0.6, 1],
                                               physicsClientId=client)
        self.body = p.createMultiBody(0, col, visual_shape, [0, 0, PAD_TOP_Z / 2],
                                      physicsClientId=client)
        p.changeDynamics(self.body, -1, lateralFriction=PAD_FRICTION, physicsClientId=client)
        self.x = self.y = self.psi = self.v = 0.0
        self.sync()

    # ---- episode start ------------------------------------------------------
    def reset(self, rng, xy=(0.0, 0.0), psi=0.0, v=None):
        """v=None: draw the speed from U(0, V_MAX) with rng (one draw per episode).
        Pass v to force a speed (tests, evaluation)."""
        self.x, self.y = float(xy[0]), float(xy[1])
        self.psi = float(psi)
        self.v = float(rng.uniform(0.0, V_MAX)) if v is None else float(v)
        self.sync()

    # ---- one physics step -----------------------------------------------------
    def integrate(self, dt=P.PHYSICS_DT):
        """Call AFTER p.stepSimulation: advance the exact straight-line pose, then re-pin."""
        self.x += self.v * np.cos(self.psi) * dt
        self.y += self.v * np.sin(self.psi) * dt
        self.sync()

    def sync(self):
        """Give PyBullet the pose AND the velocity (D19)."""
        q = p.getQuaternionFromEuler([0, 0, self.psi])
        p.resetBasePositionAndOrientation(self.body, [self.x, self.y, PAD_TOP_Z / 2], q,
                                          physicsClientId=self.cid)
        p.resetBaseVelocity(self.body, self.velocity().tolist(), [0, 0, 0],
                            physicsClientId=self.cid)

    # ---- true state (for the simulator, the reward checks and the critic) ------
    def pad_center(self):
        return np.array([self.x, self.y, PAD_TOP_Z])

    def velocity(self):
        return np.array([self.v * np.cos(self.psi), self.v * np.sin(self.psi), 0.0])

    def pad_to_world(self, pts_pad):
        """(N, 2) pad-frame xy (x forward, y left) -> (N, 3) world points on the pad top."""
        c, s = np.cos(self.psi), np.sin(self.psi)
        pts = np.atleast_2d(pts_pad)
        x = self.x + c * pts[:, 0] - s * pts[:, 1]
        y = self.y + s * pts[:, 0] + c * pts[:, 1]
        return np.stack([x, y, np.full(len(pts), PAD_TOP_Z)], 1)

    def world_to_pad(self, pts_world):
        """(N, 3) world points -> (N, 2) pad-frame xy. Inverse of pad_to_world."""
        c, s = np.cos(self.psi), np.sin(self.psi)
        d = np.atleast_2d(pts_world)[:, :2] - np.array([self.x, self.y])
        return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
