"""v6_drop B5: the world. Puts B1-B4 together: ground, Stage 1 platform with the
L70_12 pad, LMF2 drone, 60 deg camera, and the Stage 1 start rules.

No RL yet: no observation vector, no reward, no termination, no drop action
(those are Part C and Steps 4-5). This file only does reset() and step(cmd).

Start rules (decided 28 Oct; Shin-like, scaled down for Stage 1):
  pad       centre at the world origin, heading +x, speed v ~ U(0, 4) m/s (B2)
  height    drone centre U(2, 5) m above the pad top        (Shin Table I: U(2, 8))
  offset    dx, dy ~ U(-3, 3) m from the pad centre         (Shin Table I)
  heading   drone yaw = pad heading + U(-10, 10) deg        (Shin: +-60, later stage)
  velocity  U(-1, 1) m/s per axis, world frame              (Shin Table II)
  turn rate U(-10, 10) deg/s per axis, body frame           (Shin Table II)
  view      pad centre >= 10 px inside the image            (Shin Sec. II-A)
            AND at least one pad marker detected by ArUco   (our rule, 28 Oct)
  Starts that break a view rule are re-drawn (rejection sampling).

Ground: plain grey (Option 1, 28 Oct). Visual randomization is a later, separate change.

Random streams: separate generators for the start geometry + pad speed, the
initial rates, and the controller gains, so changing one never shifts another.

ONE EGL World per process (found in B5, 28 Oct): with two EGL clients open in the
same process, BOTH render wrong images (the pad is measured metres off, or not at
all), and the damage stays after the second one is closed. So World refuses to
open a second EGL World while one is open. Training uses one World per worker
process. TinyRenderer Worlds are not affected.
"""
import numpy as np
import pybullet as p

from . import lmf2_params as P
from . import pad_board as PB
from .camera import Camera
from .platform_s1 import PAD_TOP_Z, set_contact_friction
from .quad import LMF2Quad, MotorModel, make_client
from .lee_controller import LeeVelocityController

# ---- Stage 1 start rules ---------------------------------------------------------
H_RANGE = (2.0, 5.0)               # m, drone centre above the pad top
DXY_MAX = 3.0                      # m, |dx|, |dy| from the pad centre
YAW_NOISE = np.radians(10.0)       # rad, drone yaw - pad heading
V0_MAX = 1.0                       # m/s per axis, world frame
W0_MAX = np.radians(10.0)          # rad/s per axis, body frame
FOV_MARGIN_PX = 10
MAX_TRIES = 200
GROUND_GREY = 0.45


def euler_R(roll, pitch, yaw):
    q = p.getQuaternionFromEuler([roll, pitch, yaw])
    return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)


class World:
    _egl_open = 0                        # EGL Worlds currently open in this process

    def __init__(self, renderer="egl", layout=PB.DEFAULT_LAYOUT, seed=0, gains_mode="random"):
        if renderer == "egl" and World._egl_open > 0:
            raise RuntimeError("an EGL World is already open in this process; "
                               "use one EGL World per process (see world.py header)")
        self.layout = layout
        self.renderer = renderer
        self._seed_streams(seed)
        self.cid = make_client(egl=(renderer == "egl"))
        self._closed = False
        if renderer == "egl":
            World._egl_open += 1
        self._make_ground()
        self.plat = PB.textured_platform(self.cid, layout)
        self.gains = P.sample_gains(self.rng_gains, gains_mode)
        self.quad = LMF2Quad(self.cid, self.rng_gains, self.gains)
        set_contact_friction(self.cid, self.plat.body, self.quad.body)
        self.cam = Camera(self.cid, renderer=renderer)
        self.det = PB.detector()
        self.t = 0                       # decisions since reset (0.1 s each)

    # ---- setup ----------------------------------------------------------------------
    def _seed_streams(self, seed):
        ss = np.random.SeedSequence(seed)
        self.rng_start, self.rng_init, self.rng_gains = [np.random.default_rng(c)
                                                         for c in ss.spawn(3)]

    def _make_ground(self):
        """Collision plane at z = 0 with a plain grey visual (a body without a visual
        is drawn from its collision shape: a plane would show PyBullet's checker)."""
        col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[100, 100, 0.001],
                                  rgbaColor=[GROUND_GREY] * 3 + [1], physicsClientId=self.cid)
        self.ground = p.createMultiBody(0, col, vis, [0, 0, 0], physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(self.ground, [0, 0, 0], [0, 0, 0, 1],
                                          physicsClientId=self.cid)

    def close(self):
        if self._closed:
            return
        p.disconnect(self.cid)
        self._closed = True
        if self.renderer == "egl":
            World._egl_open -= 1

    def reseed(self, seed, resample_gains=True):
        """Evaluation helper: per-episode seeding (pinned seed sets). Training samples
        the gains once per World (Shin Table II: 'Env. init')."""
        self._seed_streams(seed)
        if resample_gains:
            self.gains = P.sample_gains(self.rng_gains)
            self.quad.ctrl = LeeVelocityController(self.gains)
            self.quad.motors = MotorModel(self.rng_gains)

    # ---- episode start ----------------------------------------------------------------
    def pad_centre_in_view(self, R, pos):
        uv, d = self.cam.project(self.plat.pad_center(), R, pos)
        u, v = uv[0]
        return bool(d[0] > 0 and FOV_MARGIN_PX <= u <= self.cam.w - FOV_MARGIN_PX
                    and FOV_MARGIN_PX <= v <= self.cam.h - FOV_MARGIN_PX)

    def reset(self, pad_speed=None, spawn=None):
        """New episode. pad_speed: force the pad speed (tests, evaluation).
        spawn: dict(pos, yaw[, v, w_body]) to skip the start rules (tests only).
        Returns dict(image, spawn, measurement, tries)."""
        self.plat.reset(self.rng_start, xy=(0.0, 0.0), psi=0.0, v=pad_speed)
        tries, img, meas = 0, None, None
        if spawn is None:
            while True:
                tries += 1
                if tries > MAX_TRIES:
                    raise RuntimeError("start rejection sampling failed")
                H = self.rng_start.uniform(*H_RANGE)
                dx, dy = self.rng_start.uniform(-DXY_MAX, DXY_MAX, 2)
                yaw = self.plat.psi + self.rng_start.uniform(-YAW_NOISE, YAW_NOISE)
                pos = self.plat.pad_center() + np.array([dx, dy, H])
                R = euler_R(0.0, 0.0, yaw)
                if not self.pad_centre_in_view(R, pos):
                    continue
                img = self.cam.render(R, pos)                  # what the drone would see
                meas = PB.measure(img, self.cam, self.layout, self.det)
                if meas.ids:                                   # >= 1 pad marker detected
                    break
            v0 = self.rng_init.uniform(-V0_MAX, V0_MAX, 3)
            w0 = self.rng_init.uniform(-W0_MAX, W0_MAX, 3)
            spawn = dict(pos=pos, yaw=yaw, v=v0, w_body=w0, H=H, dx=dx, dy=dy)
        else:
            spawn = dict(spawn)
            spawn.setdefault("v", np.zeros(3))
            spawn.setdefault("w_body", np.zeros(3))
        self.quad.reset_pose(np.asarray(spawn["pos"], float).tolist(), yaw=spawn["yaw"],
                             v=np.asarray(spawn["v"], float).tolist(),
                             w_body=np.asarray(spawn["w_body"], float).tolist())
        self.quad.ext_force[:] = 0.0
        self.quad.ext_torque[:] = 0.0
        p.performCollisionDetection(physicsClientId=self.cid)
        self.t = 0
        if img is None:
            img = self.render()
            meas = PB.measure(img, self.cam, self.layout, self.det)
        return dict(image=img, spawn=spawn, measurement=meas, tries=tries)

    # ---- one decision ----------------------------------------------------------------
    def step(self, cmd):
        """Hold cmd = [vx, vy, vz, yaw_rate] (heading frame) for one decision:
        10 physics steps of 0.01 s. The pad moves with D19 sync inside."""
        for _ in range(P.SUBSTEPS):
            self.quad.apply_control(cmd)
            p.stepSimulation(physicsClientId=self.cid)
            self.plat.integrate()
        self.t += 1

    def render(self):
        s = self.quad.state()
        return self.cam.render(s["R"], s["pos"])

    # ---- true state (simulator only: checks, reward terms, critic) -------------------
    def true_state(self):
        s = self.quad.state()
        R, pos, v = s["R"], s["pos"], s["v"]
        pc, pv = self.plat.pad_center(), self.plat.velocity()
        return dict(pos=pos, v=v, R=R, quat=s["quat"], w_body=s["w_body"],
                    pad_center=pc, pad_v=pv,
                    rel_pos_body=R.T @ (pc - pos),          # pad centre seen from the body
                    rel_vel_body=R.T @ (pv - v),            # pad velocity relative to the drone
                    t=self.t)
