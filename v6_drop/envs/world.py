"""v6_drop B5: the world. Puts B1-B4 together: ground, Stage 1 platform with the
L70_12 pad, LMF2 drone, 60 deg camera, and the Stage 1 start rules.

No RL yet: no observation vector and no reward (Steps 4-5). reset() starts an
episode; step(cmd, drop) runs one decision and, since C1, ends the episode with an
outcome (flight checks, the drop, the auto-cut on touch, and the five landing checks).

C1 rules (decided 28 Oct):
  Flight (checked every physics step, motors on):
    crash_ground    any drone contact with the ground
    crash_platform  contact with the platform that is not on its top face
    flip            tilt > 80 deg
    lost            > 15 m from the pad horizontally, or > 12 m above the pad top
    timeout         300 decisions (30 s) without an outcome  (TRUNCATION, not termination)
  Touchdown: the motors go OFF for the rest of the episode when
    (a) the policy chooses drop=True (motors off at the start of that decision), or
    (b) any foot first touches the pad top with the motors still on (AUTO-CUT, Option A).
  After the motors go off, the simulator runs a 2 s window inside the same step() and
  judges the landing, in this order:
    1 first contact must be the pad TOP     else crash_platform / crash_ground
    2 impact speed rel. to the pad <= 1.0   else hard_landing   (episode ends at impact)
    3 tilt <= 30 deg during the window       else tip_over
    4 centre stays over the pad              else slide_off
    5 |v_rel| <= 0.1 m/s at the end          else not_still
    all five pass -> success (= a SAFE landing). impact <= 0.5 m/s is reported too.
  Impact speed: after a drop only gravity acts, so it is computed EXACTLY by energy
  conservation from the state at the drop: v_touch = sqrt(vz0^2 + 2 g gap0) (D4). For an
  auto-cut touch (thrust on until the touch) the per-step reading is used (+-0.1 m/s).

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
from .platform_s1 import PAD_SIZE, PAD_TOP_Z, set_contact_friction
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
# ---- C1 flight checks (V5 values; horizon as Shin et al.) --------------------------
HORIZON = 300                      # decisions (30 s)
TILT_FLIGHT_MAX = np.radians(80.0)
LOST_XY = 15.0                     # m from the pad centre, horizontally
LOST_Z = 12.0                      # m above the pad top
TOP_NORMAL_MIN = 0.7               # contact normal z above this = the pad's TOP face
# ---- C1 landing checks (agreed 24-28 Oct) ----------------------------------------------
SETTLE_T = 2.0                     # s with motors off after the touchdown/drop (D1: 2 s)
V_IMPACT_MAX = 1.0                 # m/s, impact speed relative to the pad
V_IMPACT_STRICT = 0.5              # m/s, stricter extra metric (reported only)
TILT_LAND_MAX = np.radians(30.0)
V_STILL = 0.1                      # m/s, |v_drone - v_pad| at the end of the window
OUTCOMES = ("success", "hard_landing", "tip_over", "slide_off", "not_still",
            "crash_platform", "crash_ground", "flip", "lost", "timeout")


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
        self.plat_half = PAD_SIZE / 2
        self.det = PB.detector()
        self.t = 0                       # decisions since reset (0.1 s each)
        self.done = True                 # call reset() first

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
                # Put the drone AT the candidate pose before rendering. Otherwise it still
                # stands where the last episode ended (often on the pad), its body/shadow
                # changes the image, and the same seed gives a different start depending
                # on the previous episode (found in Step 6a, 29 Oct).
                self.quad.reset_pose(pos.tolist(), yaw=yaw)
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
        self.done = False
        if img is None:
            img = self.render()
            meas = PB.measure(img, self.cam, self.layout, self.det)
        return dict(image=img, spawn=spawn, measurement=meas, tries=tries)

    # ---- one decision ----------------------------------------------------------------
    def step(self, cmd, drop=False):
        """One decision (0.1 s). cmd = [vx, vy, vz, yaw_rate] (heading frame), held for 10
        physics steps. drop=True cuts the motors now and judges the landing (2 s window).
        Returns (terminated, truncated, info); info['outcome'] is None while flying."""
        assert not self.done, "episode over: call reset()"
        if drop:
            info = self._touchdown_phase(reason="drop")
            return self._finish(info, truncated=False)
        for _ in range(P.SUBSTEPS):
            v_pre = self.quad.state()["v"].copy()
            self.quad.apply_control(cmd)
            p.stepSimulation(physicsClientId=self.cid)
            self.plat.integrate()
            kind = self._contact_kind()
            if kind == "top":                                      # AUTO-CUT (Option A)
                info = self._touchdown_phase(reason="auto_cut", v_pre=v_pre, already_touching=True)
                return self._finish(info, truncated=False)
            if kind in ("ground", "side"):
                return self._finish(self._flight_end("crash_ground" if kind == "ground"
                                                     else "crash_platform"), truncated=False)
            if self._tilt() > TILT_FLIGHT_MAX:
                return self._finish(self._flight_end("flip"), truncated=False)
        self.t += 1
        s = self.true_state()
        rel = s["pos"] - s["pad_center"]
        if np.hypot(rel[0], rel[1]) > LOST_XY or rel[2] > LOST_Z:
            return self._finish(self._flight_end("lost", count_step=False), truncated=False)
        if self.t >= HORIZON:
            return self._finish(self._flight_end("timeout", count_step=False), truncated=True)
        return False, False, dict(outcome=None, t=self.t)

    # ---- helpers ------------------------------------------------------------------------
    def _finish(self, info, truncated):
        self.done = True
        info["t"] = self.t
        return (not truncated), truncated, info

    def _flight_end(self, outcome, count_step=True):
        if count_step:
            self.t += 1
        return dict(outcome=outcome, touchdown=None)

    def _tilt(self):
        R = self.quad.state()["R"]
        return float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))

    def _contact_kind(self):
        """'top' if the drone touches the pad's top face, 'side' if it touches the
        platform only elsewhere, 'ground' if it touches the ground, else None."""
        cid, d = self.cid, self.quad.body
        if p.getContactPoints(bodyA=d, bodyB=self.ground, physicsClientId=cid):
            return "ground"
        cps = p.getContactPoints(bodyA=d, bodyB=self.plat.body, physicsClientId=cid)
        if not cps:
            return None
        # contactNormalOnB (index 7) points from the platform (B) towards the drone (A)
        if any(c[7][2] > TOP_NORMAL_MIN for c in cps):
            return "top"
        return "side"

    _FEET_B = np.array([[sx * P.FOOT_XY, sy * P.FOOT_XY, P.FOOT_Z]
                        for sx in (1, -1) for sy in (1, -1)])
    _BODY_CORNERS_B = np.array([[sx * 0.08, sy * 0.08, -0.03] for sx in (1, -1) for sy in (1, -1)])

    def _lowest_z(self):
        """Height of the drone's lowest point, from its geometry (feet spheres and the
        body's bottom corners). Not getAABB: PyBullet pads the AABB by ~1 mm."""
        s = self.quad.state()
        feet = s["pos"] + self._FEET_B @ s["R"].T
        body = s["pos"] + self._BODY_CORNERS_B @ s["R"].T
        return float(min(feet[:, 2].min() - P.FOOT_R, body[:, 2].min()))

    def _motors_off_step(self):
        """One physics step with the motors OFF: no thrust, no torque (cut instantly;
        the LMF2 motors spin down in 5 ms anyway)."""
        self.quad.motors.thrust[:] = 0.0
        self.quad.last_wrench = np.zeros(4)
        p.stepSimulation(physicsClientId=self.cid)
        self.plat.integrate()

    def _touchdown_phase(self, reason, v_pre=None, already_touching=False):
        """Motors off; judge the landing over a SETTLE_T window (the five checks)."""
        n_win = int(round(SETTLE_T / P.PHYSICS_DT))
        td = dict(reason=reason, impact=None, impact_method=None, land_xy=None,
                  tilt_max_deg=0.0, slide=None, v_rel_end=None, strict_soft=None)
        k0 = 0
        if already_touching:                       # auto-cut: contact happened this substep
            first = "top"
            vz_rel = v_pre[2] - self.plat.velocity()[2]
            td["impact"], td["impact_method"] = float(max(0.0, -vz_rel)), "step_reading"
        else:                                      # drop: free fall until the first contact
            # D4: after the cut only gravity acts, so energy conservation gives the exact
            # (continuous-physics) impact speed from the state AT THE DROP:
            #   v_touch^2 = vz0^2 + 2 g (gap at the drop)
            # This is free of the per-step resolution. (Using the last look before contact
            # instead is NOT exact: PyBullet's semi-implicit Euler falls slightly faster
            # than continuous physics, so that version was off by up to 0.04 m/s.)
            vz0 = self.quad.state()["v"][2] - self.plat.velocity()[2]
            low0 = self._lowest_z()
            first = None
            for k0 in range(1, n_win + 1):
                self._motors_off_step()
                first = self._contact_kind()
                if first is not None:
                    gap0 = low0 - (PAD_TOP_Z if first == "top" else 0.0)
                    td["impact"] = float(np.sqrt(max(vz0 * vz0 + 2.0 * P.G * gap0, 0.0)))
                    td["impact_method"] = "free_fall_exact"
                    break
            if first is None:                      # (cannot happen within 2 s near the pad)
                return dict(outcome="crash_ground", touchdown=td)
        s = self.true_state()
        land_xy = self.plat.world_to_pad([s["pos"]])[0]
        td["land_xy"] = land_xy.tolist()
        if first == "ground":
            return dict(outcome="crash_ground", touchdown=td)
        if first == "side":
            return dict(outcome="crash_platform", touchdown=td)
        td["strict_soft"] = bool(td["impact"] <= V_IMPACT_STRICT)
        if td["impact"] > V_IMPACT_MAX:                                   # check 2
            return dict(outcome="hard_landing", touchdown=td)
        half = self.plat_half
        for _ in range(k0, n_win):
            self._motors_off_step()
            tilt = self._tilt()
            td["tilt_max_deg"] = max(td["tilt_max_deg"], float(np.degrees(tilt)))
            s = self.true_state()
            xy = self.plat.world_to_pad([s["pos"]])[0]
            if tilt > TILT_LAND_MAX:                                       # check 3
                return dict(outcome="tip_over", touchdown=td)
            if abs(xy[0]) > half or abs(xy[1]) > half:                     # check 4
                td["slide"] = float(np.linalg.norm(xy - land_xy))
                return dict(outcome="slide_off", touchdown=td)
            if self._contact_kind() == "ground":
                return dict(outcome="crash_ground", touchdown=td)
        s = self.true_state()
        xy = self.plat.world_to_pad([s["pos"]])[0]
        td["slide"] = float(np.linalg.norm(xy - land_xy))
        td["v_rel_end"] = float(np.linalg.norm(s["v"] - s["pad_v"]))
        if td["v_rel_end"] > V_STILL:                                      # check 5
            return dict(outcome="not_still", touchdown=td)
        return dict(outcome="success", touchdown=td)

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
