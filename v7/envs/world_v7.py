"""v7 S1d: the V7 world (20 Nov 2026).

Same rules as the V6 world (v6_drop/envs/world.py: flight checks, drop, touch detector,
referee with the five landing checks, start rules). Changed parts only:

  part              V6 (Stage 1)                      V7
  pad               PlatformS1, straight, 0 to 4 m/s  PlatformV7 (S1a): random driver or a named
                                                      path, 0 to v_max (9 default, 8 fallback)
  pad random stream shared with the start geometry    its OWN stream (rng_pad): changing the
                                                      pad never shifts the drone's start
  drone             LMF2Quad (legs, stands 0.152 m)   S500Quad (S1b, skids, stands 0.22 m)
  lowest point      V6 feet + body corners            s500.lowest_z (skids, plate, arm tips)
  camera            fixed 60 deg + PB.measure         GimbalCamera (S1c, 60 default, 45 option)
                                                      + measure_gimbal
  flight limit      HORIZON 300 decisions (30 s)      600 (60 s)
  lost              > 15 m from the pad               > 40 m
  ground picture    grey box 200 m x 200 m            grey box 4 km x 4 km (a 9 m/s pad drives up
                                                      to 540 m in 60 s; past the old box the
                                                      camera would see the empty background)
  true_state        pad centre and velocity           + pad acceleration, turn rate, heading,
                                                      speed, driver phase (teacher, S2)
  info["pad"]       (none)                            mode, start / mean / max / end speed

Start rules: the same values as V6 (height U(2, 5) m, offset +-3 m, yaw +-10 deg from the
pad heading, speed +-1 m/s, turn rate +-10 deg/s, pad centre in view, >= 1 marker seen).

ONE EGL world per process, as in V6: V6 and V7 worlds share the same counter.
"""
import numpy as np
import pybullet as p

from v6_drop.envs import lmf2_params as P
from v6_drop.envs import pad_board as PB
from v6_drop.envs import world as W6
from v6_drop.envs.quad import make_client
from v6_drop.envs.touch_detector import TouchDetector

from .gimbal_camera import PITCH_DEG, GimbalCamera, measure_gimbal
from .platform_v7 import MODES, PAD_SIZE, PAD_TOP_Z, V_MAX, set_contact_friction, textured_platform_v7
from . import s500

# ---- V7 limits (decided 19 Nov) ------------------------------------------------------
HORIZON = 600                      # decisions (60 s) for the S2 teacher study; final from S2
LOST_XY = 40.0                     # m from the pad centre, horizontally
LOST_Z = W6.LOST_Z                 # 12 m above the pad top (unchanged)
GROUND_HALF = 2000.0               # m, half side of the grey ground picture
REST_HEIGHT = s500.REST_HEIGHT     # 0.22 m (V6: 0.152)


class WorldV7(W6.World):
    """V6 World with the V7 pad, the S500 drone and the gimbal camera."""

    def __init__(self, renderer="egl", layout=PB.DEFAULT_LAYOUT, seed=0, gains_mode="random",
                 touch="perfect", touch_cfg=None, noise=None,
                 pitch_deg=PITCH_DEG, mode="random", v_max=V_MAX):
        assert touch in W6.TOUCH_MODES, touch
        assert mode in MODES, mode
        if renderer == "egl" and W6.World._egl_open > 0:
            raise RuntimeError("an EGL World is already open in this process; "
                               "use one EGL World per process (see v6_drop world.py header)")
        self.layout = layout
        self.renderer = renderer
        self.mode, self.v_max = mode, float(v_max)
        self._seed_streams(seed)
        self.cid = make_client(egl=(renderer == "egl"))
        self._closed = False
        if renderer == "egl":
            W6.World._egl_open += 1
        self._make_ground()
        self.plat = textured_platform_v7(self.cid, layout)
        self.gains = P.sample_gains(self.rng_gains, gains_mode)
        self.quad = s500.S500Quad(self.cid, self.rng_gains, self.gains)
        set_contact_friction(self.cid, self.plat.body, self.quad.body)
        self.cam = GimbalCamera(self.cid, pitch_deg=pitch_deg, renderer=renderer)
        self.plat_half = PAD_SIZE / 2
        self.det = PB.detector()
        self.t = 0
        self.frame_hook = None
        self._off_k = 0
        self.step_hook = None
        self.done = True
        self.touch = touch
        self.detector = TouchDetector(**(touch_cfg or {})) if touch == "sensor" else None
        self._k = 0
        self._tr = None
        self._on_pad = False
        self.noise = noise
        self._pad_stats = None

    # ---- setup ----------------------------------------------------------------------
    def _seed_streams(self, seed):
        """Four streams. The first three are the V6 ones (same children of the seed);
        the fourth drives the pad (start speed and driving plan)."""
        ss = np.random.SeedSequence(seed)
        self.rng_start, self.rng_init, self.rng_gains, self.rng_pad = [
            np.random.default_rng(c) for c in ss.spawn(4)]

    def _make_ground(self):
        col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[GROUND_HALF, GROUND_HALF, 0.001],
                                  rgbaColor=[W6.GROUND_GREY] * 3 + [1], physicsClientId=self.cid)
        self.ground = p.createMultiBody(0, col, vis, [0, 0, 0], physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(self.ground, [0, 0, 0], [0, 0, 0, 1],
                                          physicsClientId=self.cid)

    def measure(self, img):
        """Marker reading for the frame just rendered (gimbal: call right after render)."""
        return measure_gimbal(img, self.cam, self.layout, self.det)

    # ---- episode start ----------------------------------------------------------------
    def reset(self, pad_speed=None, spawn=None, mode=None, plan=None):
        """New episode. pad_speed: force the pad's START speed. mode: pad path for this
        flight ("random", "straight", "wave", "corners"; None = the World's mode).
        plan: fixed list of driver pieces (tests). spawn: as V6 (tests only)."""
        mode = self.mode if mode is None else mode
        self.plat.reset(self.rng_pad, xy=(0.0, 0.0), psi=0.0, v=pad_speed, mode=mode,
                        plan=plan, v_max=self.v_max)
        tries, img, meas = 0, None, None
        if spawn is None:
            while True:
                tries += 1
                if tries > W6.MAX_TRIES:
                    raise RuntimeError("start rejection sampling failed")
                H = self.rng_start.uniform(*W6.H_RANGE)
                dx, dy = self.rng_start.uniform(-W6.DXY_MAX, W6.DXY_MAX, 2)
                yaw = self.plat.psi + self.rng_start.uniform(-W6.YAW_NOISE, W6.YAW_NOISE)
                pos = self.plat.pad_center() + np.array([dx, dy, H])
                R = W6.euler_R(0.0, 0.0, yaw)
                if not self.pad_centre_in_view(R, pos):
                    continue
                self.quad.reset_pose(pos.tolist(), yaw=yaw)    # V6 6a: render AT the candidate
                img = self.cam.render(R, pos)
                meas = self.measure(img)
                if meas.ids:
                    break
            v0 = self.rng_init.uniform(-W6.V0_MAX, W6.V0_MAX, 3)
            w0 = self.rng_init.uniform(-W6.W0_MAX, W6.W0_MAX, 3)
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
        self.done = False
        self._k = 0
        self._on_pad = False
        self._tr = dict(touches=0, first_k=None, impact_max=None)
        self._pad_stats = dict(mode=mode, v_start=float(self.plat.v), v_sum=0.0, n=0,
                               v_max_seen=float(self.plat.v))
        if self.detector is not None:
            self.detector.reset()
        if self.noise is not None:
            self.noise.reset()
        if img is None:
            img = self.render()
            meas = self.measure(img)
        return dict(image=img, spawn=spawn, measurement=meas, tries=tries)

    # ---- one decision (V6 World.step; only HORIZON, LOST_XY and the pad report differ) --
    def step(self, cmd, drop=False):
        assert not self.done, "episode over: call reset()"
        if drop:
            info = self._touchdown_phase(reason="drop")
            return self._finish(info, truncated=False)
        for _ in range(P.SUBSTEPS):
            v_pre = self.quad.state()["v"].copy()
            self.quad.apply_control(cmd)
            p.stepSimulation(physicsClientId=self.cid)
            self.plat.integrate()
            self._k += 1
            ps = self._pad_stats
            ps["v_sum"] += self.plat.v
            ps["n"] += 1
            ps["v_max_seen"] = max(ps["v_max_seen"], float(self.plat.v))
            if self.noise is not None:
                self.noise.advance()
            kind = self._contact_kind()
            if kind == "top" and self.touch == "perfect":
                info = self._touchdown_phase(reason="auto_cut", v_pre=v_pre, already_touching=True)
                return self._finish(info, truncated=False)
            if kind in ("ground", "side"):
                return self._finish(self._flight_end("crash_ground" if kind == "ground"
                                                     else "crash_platform"), truncated=False)
            if self.touch == "sensor":
                if kind == "top":
                    if not self._on_pad:
                        imp = float(max(0.0, -(v_pre[2] - self.plat.velocity()[2])))
                        tr = self._tr
                        tr["touches"] += 1
                        if tr["first_k"] is None:
                            tr["first_k"] = self._k
                        tr["impact_max"] = imp if tr["impact_max"] is None else max(tr["impact_max"], imp)
                        if imp > W6.V_IMPACT_MAX:
                            return self._finish(self._hard_touch(imp), truncated=False)
                    self._on_pad = True
                else:
                    self._on_pad = False
            if self._tilt() > W6.TILT_FLIGHT_MAX:
                return self._finish(self._flight_end("flip"), truncated=False)
            fired = self.touch == "sensor" and self.detector.update(self.own_vz())
            if self.step_hook is not None:
                self.step_hook(self._k)
            if fired:
                info = self._touchdown_phase(reason="touch_detect", v_pre=v_pre,
                                             already_touching=(kind == "top"))
                return self._finish(info, truncated=False)
        self.t += 1
        s = self.true_state()
        rel = s["pos"] - s["pad_center"]
        if np.hypot(rel[0], rel[1]) > LOST_XY or rel[2] > LOST_Z:
            return self._finish(self._flight_end("lost", count_step=False), truncated=False)
        if self.t >= HORIZON:
            return self._finish(self._flight_end("timeout", count_step=False), truncated=True)
        return False, False, dict(outcome=None, t=self.t)

    def _finish(self, info, truncated):
        term, trunc, info = super()._finish(info, truncated)
        info["pad"] = self.pad_report()
        return term, trunc, info

    def pad_report(self):
        ps = self._pad_stats
        return dict(mode=ps["mode"], v_start=ps["v_start"],
                    v_mean=ps["v_sum"] / ps["n"] if ps["n"] else ps["v_start"],
                    v_max=ps["v_max_seen"], v_end=float(self.plat.v), phase=self.plat.phase)

    # ---- geometry --------------------------------------------------------------------
    def _lowest_z(self):
        """Height of the S500's lowest point (skids, plate, arm tips), from its geometry."""
        s = self.quad.state()
        return s500.lowest_z(s["R"], s["pos"])

    # ---- true state (simulator only: checks, reward terms, critic, teacher) ------------
    def true_state(self):
        s = super().true_state()
        s.update(pad_a=self.plat.acceleration(), pad_w=float(self.plat.w),
                 pad_psi=float(self.plat.psi), pad_speed=float(self.plat.v),
                 pad_phase=self.plat.phase)
        return s
