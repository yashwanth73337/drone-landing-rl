"""Moving landing platform (V5 Block 2).

Planar unicycle with forward speed v and yaw rate w, following a bounded random
walk (Shin et al. Sec. II-B, Table I):
    v_{t+1} = v_t + dv,   w_{t+1} = w_t + dw,  applied once per control step (0.1 s)
scaled by the curriculum scalar c (SPEC D5) and clipped (SPEC D6).
The body is a kinematic box (mass 0) whose pose is reset every physics step.
The episode ends at touchdown, so the solver never needs the box's velocity;
the platform's true velocity comes from this class.
"""
import numpy as np
import pybullet as p

from . import lmf2_params as P

# ---- SPEC §3 ---------------------------------------------------------------
PAD_SIZE = 1.5            # m, square pad side [paper p.5546]
PAD_TOP_Z = 1.0           # m, pad-top height [inferred, Fig. 9]
V0_MAX = 8.0              # m/s, v0 ~ U(0, 8)          [Table I]
DV_MAX = 0.5              # m/s per control step        [Table I]
DW_MAX = np.radians(3.0)  # rad/s per control step      [Table I]
W0 = 0.0                  # rad/s                       [Table I]
W_CLIP = np.radians(30.0)  # |w| bound                  [SPEC D6]




class Platform:
    def __init__(self, client, marker=True):
        self.cid = client
        half = [PAD_SIZE / 2, PAD_SIZE / 2, PAD_TOP_Z / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
        self.textured = marker
        if marker:
            # One box mesh, marker texture on the top face (pad_marker._write_box_obj).
            from .pad_marker import ensure_assets
            tex, obj = ensure_assets()
            vis = p.createVisualShape(p.GEOM_MESH, fileName=obj, physicsClientId=client)
        else:
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                      rgbaColor=[0.6, 0.6, 0.6, 1], physicsClientId=client)
        self.body = p.createMultiBody(0, col, vis, [0, 0, PAD_TOP_Z / 2],
                                      physicsClientId=client)
        if marker:
            tid = p.loadTexture(tex, physicsClientId=client)
            p.changeVisualShape(self.body, -1, textureUniqueId=tid, rgbaColor=[1, 1, 1, 1],
                                physicsClientId=client)
        self.c = 0.0
        self.x = self.y = self.psi = self.v = self.w = 0.0

    # ---- lifecycle ----------------------------------------------------------
    def reset(self, rng, c, xy=(0.0, 0.0), psi=0.0):
        self.c = float(np.clip(c, 0.0, 1.0))
        self.x, self.y = float(xy[0]), float(xy[1])
        self.psi = float(psi)
        self.v = rng.uniform(0.0, V0_MAX * self.c) if self.c > 0 else 0.0
        self.w = W0
        self.n_clip_v = self.n_clip_w = self.n_perturb = 0
        self._sync()

    def perturb(self, rng):
        """Once per control step (after integrating that step)."""
        c = self.c
        v_new = self.v + rng.uniform(-DV_MAX * c, DV_MAX * c) if c > 0 else self.v
        w_new = self.w + rng.uniform(-DW_MAX * c, DW_MAX * c) if c > 0 else self.w
        v_max = V0_MAX * c
        self.n_clip_v += int(v_new < 0.0 or v_new > v_max)
        self.n_clip_w += int(abs(w_new) > W_CLIP)
        self.n_perturb += 1
        self.v = float(np.clip(v_new, 0.0, v_max))
        self.w = float(np.clip(w_new, -W_CLIP, W_CLIP))

    def integrate(self, dt=P.PHYSICS_DT):
        """Exact unicycle arc over dt; once per physics step."""
        v, w, psi = self.v, self.w, self.psi
        if abs(w) < 1e-9:
            self.x += v * np.cos(psi) * dt
            self.y += v * np.sin(psi) * dt
        else:
            self.x += v / w * (np.sin(psi + w * dt) - np.sin(psi))
            self.y += v / w * (np.cos(psi) - np.cos(psi + w * dt))
        self.psi = psi + w * dt
        self._sync()

    # ---- state --------------------------------------------------------------
    def pad_center(self):
        return np.array([self.x, self.y, PAD_TOP_Z])

    def velocity(self):
        return np.array([self.v * np.cos(self.psi), self.v * np.sin(self.psi), 0.0])

    def pad_to_world(self, pts_pad):
        """(N, 2) pad-frame xy -> (N, 3) world points on the pad top."""
        c, s = np.cos(self.psi), np.sin(self.psi)
        pts = np.atleast_2d(pts_pad)
        x = self.x + c * pts[:, 0] - s * pts[:, 1]
        y = self.y + s * pts[:, 0] + c * pts[:, 1]
        return np.stack([x, y, np.full(len(pts), PAD_TOP_Z)], 1)

    def _sync(self):
        q = p.getQuaternionFromEuler([0, 0, self.psi])
        p.resetBasePositionAndOrientation(
            self.body, [self.x, self.y, PAD_TOP_Z / 2], q, physicsClientId=self.cid)

