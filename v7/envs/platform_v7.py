"""v7 S1a: the V7 landing platform with a random "driver" (19 Nov 2026).

Stage 1 (v6_drop/envs/platform_s1.py) drove in a straight line at one constant speed.
In V7 the platform moves like a car whose plan the drone does not know:
  keep    hold the speed for a while
  speed   speed up or slow down to a new speed (at most A_MAX = 2 m/s^2)
  wave    a smooth left-right wave (heading swings +-amp, then back)
  corner  like a car at a street corner: brake to a corner speed, turn 90 deg left or
          right on an arc, then speed back up to the old speed
Speeds stay in [0, v_max], default V_MAX = 9 m/s (19 Nov: supervisor "4 m/s is too low";
the lab drone flies at most 10 m/s and must be faster than the pad to catch it from rest;
8 m/s is the fallback; the S2 teacher study decides). Pass v_max to reset() to change it.
Sideways acceleration on curves (v^2 / R = v x turn rate) stays at most A_LAT = 2 m/s^2.
Why 2 m/s^2 (both limits): the drone can change its speed by at most 3 m/s^2, and a drone
sitting on the pad with motors off is held only by friction 0.5 x g = 4.9 m/s^2 (D24).

Modes (reset(mode=...)):
  "random"   start speed U(0, v_max); then random pieces, one after another (training)
  "straight" one speed U(0, v_max) for the whole flight                  (named test path 1)
  "wave"     one speed U(2, v_max) and an endless smooth wave            (named test path 2)
  "corners"  cruise U(4, v_max): straight roads of 3 to 6 s joined by 90 deg corners, left or
             right at random                                              (named test path 3)
  plan=[...] a fixed list of pieces (tests). See Driver for the piece formats.

Physics (same pitfalls as Stage 1, kept on purpose):
  * KINEMATIC box (mass 0): our code moves it, forces do not.
  * D19: every physics step the box gets its pose AND its velocity, now also its ANGULAR
    velocity (turn rate). Without it, a drone resting on a turning pad is not turned with it.
  * D24: friction pad 1.0 x drone 0.5 = 0.5 effective.
Randomness: reset() takes EXACTLY two numbers from the caller's rng (start speed, plan
seed), so the caller's other draws (drone start) do not depend on how long the plan is.
"""
import numpy as np
import pybullet as p

from v6_drop.envs import lmf2_params as P
from v6_drop.envs.platform_s1 import (DRONE_FRICTION, PAD_FRICTION, PAD_SIZE, PAD_TOP_Z,
                                      set_contact_friction)

V_MAX = 9.0              # m/s, default top pad speed (19 Nov; fallback 8; S2 decides)
A_MAX = 2.0              # m/s^2, speeding up / braking
A_LAT = 2.0              # m/s^2, sideways on curves (v^2 / R)
MODES = ("random", "straight", "wave", "corners")
# random pieces: probabilities and ranges
P_KEEP, P_SPEED, P_WAVE, P_CORNER = 0.30, 0.30, 0.20, 0.20
KEEP_T = (2.0, 5.0)      # s
ACC_RANGE = (0.5, A_MAX)  # m/s^2, how hard one speed change is
WAVE_AMP = (np.radians(20.0), np.radians(40.0))   # heading swing
WAVE_P_MIN = 6.0         # s, shortest wave period
V_WAVE_MIN = 2.0         # m/s, below this a wave is replaced by "keep"
V_TURN = (3.0, 5.0)      # m/s, corner speed (a car at a street corner, roughly 10 to 18 km/h)
V_TURN_MIN = 2.0         # m/s, a pad slower than this first speeds up to it (a car must roll to turn)
R_MIN = 6.0              # m, tightest corner radius
CORNER_LEG_T = (3.0, 6.0)  # s, straight road between corners ("corners" mode)

__all__ = ["PlatformV7", "Driver", "V_MAX", "A_MAX", "A_LAT", "MODES", "PAD_SIZE", "PAD_TOP_Z",
           "PAD_FRICTION", "DRONE_FRICTION", "set_contact_friction", "textured_platform_v7"]


def wave_period(v, amp):
    """Shortest period that keeps the wave's peak sideways acceleration v*amp*2pi/T <= A_LAT."""
    return max(WAVE_P_MIN, v * amp * 2 * np.pi / A_LAT)


def corner_radius(v):
    return max(R_MIN, v * v / A_LAT)


class Driver:
    """Produces (a_long, turn rate) for every physics step.

    Piece formats (also usable in plan=[...]):
      ("keep", T)                      a = 0, turn = 0 for T seconds
      ("speed", v_target, acc)         change speed at |a| = acc until v_target
      ("turn", angle, R)               turn by angle (rad, + = left) on radius R, speed kept
      ("wave", amp, period, n)         heading offset amp*sin(2 pi t / period) for n periods
      ("corner", sign, v_c, acc)       macro: ("speed", v_c, acc), ("turn", sign*90deg, R(v_c)),
                                       ("speed", <speed before the corner>, acc)
    When the list runs out: "random" draws new pieces, the named modes repeat their pattern.
    """

    def __init__(self, mode, rng, plan=None, cruise=None, wave=None, v_max=V_MAX):
        self.mode, self.rng, self.v_max = mode, rng, float(v_max)
        self.queue = list(plan) if plan is not None else []
        self.fixed = plan is not None
        self.cruise = cruise          # corners: speed to return to
        self.wave = wave              # wave: (amp, period)
        self.cur, self.t_in = None, 0.0
        self.name = "keep"

    # ---- next piece -----------------------------------------------------------------
    def _draw(self, v):
        r = self.rng
        if self.fixed:
            return ("keep", 1e9)      # a fixed plan ends by keeping the last speed
        if self.mode == "straight":
            return ("keep", 1e9)
        if self.mode == "wave":
            amp, per = self.wave
            return ("wave", amp, per, 1)
        if self.mode == "corners":
            if self.name == "corner-out" or self.name == "start":
                return ("keep", r.uniform(*CORNER_LEG_T))
            return ("corner", r.choice([-1.0, 1.0]), r.uniform(*V_TURN), r.uniform(1.0, A_MAX))
        # random
        k = r.choice(4, p=[P_KEEP, P_SPEED, P_WAVE, P_CORNER])
        if k == 0:
            return ("keep", r.uniform(*KEEP_T))
        if k == 1:
            return ("speed", r.uniform(0.0, self.v_max), r.uniform(*ACC_RANGE))
        if k == 2:
            if v < V_WAVE_MIN:
                return ("keep", r.uniform(*KEEP_T))
            amp = r.uniform(*WAVE_AMP)
            return ("wave", amp, wave_period(v, amp), 1)
        return ("corner", r.choice([-1.0, 1.0]), r.uniform(*V_TURN), r.uniform(1.0, A_MAX))

    def _start(self, piece, v):
        kind = piece[0]
        if kind == "corner":
            _, sign, v_c, acc = piece
            v_c = max(min(v_c, max(v, V_TURN_MIN)), V_TURN_MIN)   # brake to v_c, never below 2
            back = v
            self.queue[0:0] = [("speed", v_c, acc, "corner-in"),
                               ("turn", sign * np.pi / 2, corner_radius(v_c), "corner-turn"),
                               ("speed", back, acc, "corner-out")]
            return self._start(self.queue.pop(0), v)
        self.cur, self.t_in = piece, 0.0
        names = {"keep": "keep", "speed": "speed", "turn": "turn", "wave": "wave"}
        self.name = piece[-1] if (kind in ("speed", "turn") and isinstance(piece[-1], str)) else names[kind]
        if kind == "speed" and self.name == "speed":
            self.name = "speed up" if piece[1] > v else "slow down"
        if kind == "turn":
            self.turned = 0.0

    def controls(self, v, dt):
        """(a_long, turn rate) for the next dt. May finish the current piece early inside dt
        (then the rest of dt uses the next piece's first value: exact enough at dt = 0.01 s)."""
        for _ in range(100):
            if self.cur is None:
                self._start(self.queue.pop(0) if self.queue else self._draw(v), v)
            kind = self.cur[0]
            if kind == "keep":
                if self.t_in < self.cur[1] - 1e-9:
                    self.t_in += dt
                    return 0.0, 0.0
            elif kind == "speed":
                gap = self.cur[1] - v
                if abs(gap) > 1e-9:
                    acc = self.cur[2]
                    a = np.sign(gap) * min(acc, abs(gap) / dt)    # land exactly on the target
                    return float(a), 0.0
            elif kind == "turn":
                ang, R = self.cur[1], self.cur[2]
                left = abs(ang) - self.turned
                if left > 1e-9 and v > 1e-6:
                    w = v / R
                    step = min(w * dt, left)
                    self.turned += step
                    return 0.0, float(np.sign(ang) * step / dt)
            elif kind == "wave":
                amp, per, n = self.cur[1], self.cur[2], self.cur[3]
                if self.t_in < n * per - 1e-9:
                    tm = self.t_in + 0.5 * dt                       # midpoint of this step
                    self.t_in += dt
                    return 0.0, float(amp * 2 * np.pi / per * np.cos(2 * np.pi * tm / per))
            self.cur = None                                         # piece done -> next one
        raise RuntimeError("driver: no piece produced controls")


class PlatformV7:
    """Pad centre x, y, heading psi, speed v; turn rate w and acceleration a of the last step."""

    def __init__(self, client, visual_shape=None):
        self.cid = client
        half = [PAD_SIZE / 2, PAD_SIZE / 2, PAD_TOP_Z / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
        if visual_shape is None:
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                               rgbaColor=[0.6, 0.6, 0.6, 1], physicsClientId=client)
        self.body = p.createMultiBody(0, col, visual_shape, [0, 0, PAD_TOP_Z / 2],
                                      physicsClientId=client)
        p.changeDynamics(self.body, -1, lateralFriction=PAD_FRICTION, physicsClientId=client)
        self.x = self.y = self.psi = self.v = self.w = self.a = 0.0
        self.mode, self.driver, self.v_max = "straight", None, V_MAX
        self.sync()

    # ---- episode start ----------------------------------------------------------------
    def reset(self, rng, xy=(0.0, 0.0), psi=0.0, v=None, mode="random", plan=None, v_max=V_MAX):
        """Takes exactly two draws from rng: the start speed and the plan seed.
        v: force the start speed (tests, evaluation). plan: fixed list of pieces (tests).
        v_max: top pad speed for this flight (default V_MAX = 9 m/s)."""
        assert mode in MODES, mode
        assert v_max > 4.0, "corners mode cruises at U(4, v_max)"
        self.v_max = float(v_max)
        u_speed = rng.uniform()                      # draw 1
        plan_seed = int(rng.integers(2 ** 62))       # draw 2
        prng = np.random.default_rng(plan_seed)
        lo = {"random": 0.0, "straight": 0.0, "wave": V_WAVE_MIN, "corners": 4.0}[mode]
        v0 = lo + u_speed * (self.v_max - lo) if v is None else float(v)
        self.x, self.y, self.psi, self.v = float(xy[0]), float(xy[1]), float(psi), float(v0)
        self.w = self.a = 0.0
        self.mode, self.plan_seed = mode, plan_seed
        wave = None
        if mode == "wave" and plan is None:
            amp = prng.uniform(*WAVE_AMP)
            wave = (amp, wave_period(max(self.v, V_WAVE_MIN), amp))
        self.driver = Driver(mode, prng, plan=plan, cruise=self.v, wave=wave, v_max=self.v_max)
        self.driver.name = "start"
        self.sync()

    # ---- one physics step --------------------------------------------------------------
    def integrate(self, dt=P.PHYSICS_DT):
        """Call AFTER p.stepSimulation: advance the pad by its driver, then re-pin (D19)."""
        a, w = self.driver.controls(self.v, dt)
        v_new = min(max(self.v + a * dt, 0.0), self.v_max + 1e-9)
        v_mid, psi_mid = 0.5 * (self.v + v_new), self.psi + 0.5 * w * dt
        self.x += v_mid * np.cos(psi_mid) * dt
        self.y += v_mid * np.sin(psi_mid) * dt
        self.psi += w * dt
        self.v, self.w, self.a = v_new, w, a
        self.sync()

    def sync(self):
        """Give PyBullet the pose, the velocity and the turn rate (D19 + turning)."""
        q = p.getQuaternionFromEuler([0, 0, self.psi])
        p.resetBasePositionAndOrientation(self.body, [self.x, self.y, PAD_TOP_Z / 2], q,
                                          physicsClientId=self.cid)
        p.resetBaseVelocity(self.body, self.velocity().tolist(), [0, 0, self.w],
                            physicsClientId=self.cid)

    # ---- true state (simulator, referee, teacher, critic) ---------------------------------
    @property
    def phase(self):
        return self.driver.name if self.driver is not None else "keep"

    def pad_center(self):
        return np.array([self.x, self.y, PAD_TOP_Z])

    def velocity(self):
        return np.array([self.v * np.cos(self.psi), self.v * np.sin(self.psi), 0.0])

    def acceleration(self):
        """World acceleration of the pad centre: along the road a, sideways v * w."""
        c, s = np.cos(self.psi), np.sin(self.psi)
        return np.array([self.a * c - self.v * self.w * s, self.a * s + self.v * self.w * c, 0.0])

    def pad_to_world(self, pts_pad):
        c, s = np.cos(self.psi), np.sin(self.psi)
        pts = np.atleast_2d(pts_pad)
        x = self.x + c * pts[:, 0] - s * pts[:, 1]
        y = self.y + s * pts[:, 0] + c * pts[:, 1]
        return np.stack([x, y, np.full(len(pts), PAD_TOP_Z)], 1)

    def world_to_pad(self, pts_world):
        c, s = np.cos(self.psi), np.sin(self.psi)
        d = np.atleast_2d(pts_world)[:, :2] - np.array([self.x, self.y])
        return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def textured_platform_v7(client, layout=None):
    """A PlatformV7 whose top face shows the marker board (same board as V6)."""
    from v6_drop.envs import pad_board as PB
    layout = PB.DEFAULT_LAYOUT if layout is None else layout
    tex, obj = PB.ensure_assets(layout)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=obj, physicsClientId=client)
    plat = PlatformV7(client, visual_shape=vis)
    tid = p.loadTexture(tex, physicsClientId=client)
    p.changeVisualShape(plat.body, -1, textureUniqueId=tid, rgbaColor=[1, 1, 1, 1],
                        physicsClientId=client)
    return plat
