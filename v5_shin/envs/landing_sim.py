"""Landing simulation core (V5 Block 4): world assembly, Table I spawn, and
termination. No observation, reward or gym API yet (Blocks 6-10).

Episode outcomes (SPEC §6):
  success          any drone contact with the pad-top face (contact normal +z)
  crash_platform   contact with the platform on a side face
  crash_ground     contact with the ground
  tilt             tilt > 80 deg
  drift            horizontal distance to the pad > 15 m, or > 12 m above the pad top
  timeout          300 policy steps (truncation, not termination)
Termination is checked after EVERY physics substep; the policy step ends
early at the first terminal substep.
"""
import numpy as np
import pybullet as p

from . import lmf2_params as P
from .camera import Camera
from .ground import Ground
from .platform import PAD_SIZE, PAD_TOP_Z, Platform
from .quad import LMF2Quad, make_client

# ---- Table I (p.5544) -------------------------------------------------------
DZ0_RANGE = (2.0, 8.0)          # m, CoM above the pad top          [paper]
DXY0_MAX = 3.0                  # m, |dx0|, |dy0| (world axes)      [paper]
PSI0_MAX = np.radians(60.0)     # platform yaw misalignment          [paper]
# ---- SPEC D12 / §6 ------------------------------------------------------------
YAW_NOISE = np.radians(15.0)    # drone yaw = bearing to pad + U(+-15 deg)  [D12]
FOV_MARGIN_PX = 10              # pad centre must project >= 10 px inside the image
MAX_SPAWN_TRIES = 100
HORIZON = 300                   # policy steps                   [paper]
TILT_MAX = np.radians(80.0)     # [unspecified, SPEC §6]
DRIFT_XY = 15.0                 # m                               [unspecified]
DRIFT_Z = 12.0                  # m above the pad top             [unspecified]
TOP_NORMAL_MIN = 0.7            # contactNormalOnB.z for "top face"


class LandingSim:
    def __init__(self, renderer="tiny", egl=False, marker=True, gains_mode="random", seed=0):
        self.rng = np.random.default_rng(seed)
        self.cid = make_client(egl=egl)
        self.ground = Ground(self.cid)
        self.plat = Platform(self.cid, marker=marker)
        # Table II control gains: sampled once per env ("Env. init")
        self.gains = P.sample_gains(self.rng, gains_mode)
        self.quad = LMF2Quad(self.cid, self.rng, self.gains)
        self.cam = Camera(self.cid, renderer=renderer)
        self.t = 0
        self.done = True

    def close(self):
        p.disconnect(self.cid)

    # ---- spawn ----------------------------------------------------------------
    def sample_spawn(self, c):
        """Table I + D12, rejection-sampled until the pad centre is in the image.
        The pad starts at the world origin."""
        rng = self.rng
        pad = np.array([0.0, 0.0, PAD_TOP_Z])
        for tries in range(1, MAX_SPAWN_TRIES + 1):
            dx, dy = rng.uniform(-DXY0_MAX, DXY0_MAX, 2)
            dz = rng.uniform(*DZ0_RANGE)
            bearing = np.arctan2(-dy, -dx)            # drone -> pad
            yaw = bearing + rng.uniform(-YAW_NOISE, YAW_NOISE)
            psi0 = rng.uniform(-PSI0_MAX, PSI0_MAX)
            pos = pad + np.array([dx, dy, dz])
            c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
            R = np.array([[c_yaw, -s_yaw, 0], [s_yaw, c_yaw, 0], [0, 0, 1.0]])
            uv, depth = self.cam.project(pad, R, pos)
            u, v = uv[0]
            if depth[0] > 0 and FOV_MARGIN_PX <= u <= self.cam.w - FOV_MARGIN_PX \
                    and FOV_MARGIN_PX <= v <= self.cam.h - FOV_MARGIN_PX:
                return dict(pos=pos, yaw=yaw, psi_plat=yaw + psi0, psi0=psi0,
                            dxyz0=np.array([dx, dy, dz]), tries=tries, uv0=uv[0])
        raise RuntimeError("spawn rejection sampling failed")

    def reset(self, c=1.0, spawn=None):
        """spawn: optional dict(pos, yaw, psi_plat[, roll, pitch, v]) for hand-built tests."""
        sp = spawn if spawn is not None else self.sample_spawn(c)
        self.plat.reset(self.rng, c, xy=(0.0, 0.0), psi=sp["psi_plat"])
        self.quad.reset_pose(sp["pos"], yaw=sp["yaw"], v=sp.get("v", (0, 0, 0)),
                             roll=sp.get("roll", 0.0), pitch=sp.get("pitch", 0.0))
        p.performCollisionDetection(physicsClientId=self.cid)
        self.t = 0
        self.done = False
        self.spawn = sp
        return dict(spawn=sp)

    # ---- step -----------------------------------------------------------------
    def step(self, cmd):
        """cmd = [vx, vy, vz, yaw_rate] in the heading frame, held for one policy
        step. Returns (terminated, truncated, info)."""
        assert not self.done, "call reset()"
        outcome, sub = None, P.SUBSTEPS
        for k in range(P.SUBSTEPS):
            self.quad.apply_control(cmd)
            self.plat.integrate()
            p.stepSimulation(physicsClientId=self.cid)
            outcome = self.check_termination()
            if outcome is not None:
                sub = k + 1
                break
        self.t += 1
        if outcome is None:
            self.plat.perturb(self.rng)
        terminated = outcome is not None
        truncated = (not terminated) and self.t >= HORIZON
        if truncated:
            outcome = "timeout"
        self.done = terminated or truncated
        return terminated, truncated, dict(outcome=outcome, substeps=sub, t=self.t,
                                           **self.touchdown_info(outcome))

    def check_termination(self):
        cid, drone = self.cid, self.quad.body
        top = side = False
        for c in p.getContactPoints(bodyA=drone, bodyB=self.plat.body, physicsClientId=cid):
            if c[7][2] >= TOP_NORMAL_MIN:
                top = True
            else:
                side = True
        if top:
            return "success"
        if side:
            return "crash_platform"
        if p.getContactPoints(bodyA=drone, bodyB=self.ground.body, physicsClientId=cid):
            return "crash_ground"
        s = self.quad.state()
        if np.arccos(np.clip(s["R"][2, 2], -1, 1)) > TILT_MAX:
            return "tilt"
        rel = s["pos"] - self.plat.pad_center()
        if np.hypot(rel[0], rel[1]) > DRIFT_XY or rel[2] > DRIFT_Z:
            return "drift"
        return None

    def touchdown_info(self, outcome):
        """Logged for every terminal outcome (not used by success, SPEC §6)."""
        if outcome is None:
            return {}
        s = self.quad.state()
        rel = s["pos"] - self.plat.pad_center()
        c, sn = np.cos(self.plat.psi), np.sin(self.plat.psi)
        rel_pad = np.array([c * rel[0] + sn * rel[1], -sn * rel[0] + c * rel[1]])
        v_rel = s["v"] - self.plat.velocity()
        return dict(
            rel_pos=rel, rel_pos_pad=rel_pad, rel_vel=v_rel,
            tilt_deg=float(np.degrees(np.arccos(np.clip(s["R"][2, 2], -1, 1)))),
            com_over_pad=bool(np.all(np.abs(rel_pad) <= PAD_SIZE / 2)),
        )

    # ---- helpers for oracles and tests ------------------------------------------
    def true_relative_state(self):
        """World-frame platform-minus-drone position and velocity (Block 8 rotates
        these into the body frame for s_rel)."""
        s = self.quad.state()
        return self.plat.pad_center() - s["pos"], self.plat.velocity() - s["v"], s
