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

Post-touchdown hold (SPEC D20; OFF by default, hold_steps=0 = the paper's task above).
With hold_steps > 0 the first pad-top contact is a TOUCHDOWN, not the end:
  hard_landing     |impact v_z relative to the pad| > V_IMP at touchdown (episode ends)
  then the policy keeps flying for hold_steps policy steps:
  bounce           no pad-top contact during a whole policy step
  slide_off        CoM leaves the pad square (pad frame)
  tip_over         tilt > HOLD_TILT_MAX
  crash_platform / crash_ground   side or ground contact
  success          hold completed and |v_rel| <= V_STILL in each of the last HOLD_STILL_STEPS
  not_still        hold completed but that speed condition failed
  The 300-step horizon applies to flight only; a hold may run up to hold_steps past it.

Hold practice (SPEC D22, training only): reset(on_pad=True) starts the episode already in the
hold phase, the drone touching the pad top (random pad-frame offset within +-PRACTICE_XY,
random heading), moving with the pad plus a soft PRACTICE_VZ descent, as at the moment of a soft
touchdown (so it is in contact during the first step). info["practice"] marks these episodes.

Smooth pad speed changes (SPEC D23, pad_smooth=True; default False = the paper's model): the
random walk is unchanged (same draws, same clipping, same v and w at every 0.1 s mark), but each
change is spread evenly over the next policy step's physics substeps instead of applied at once.
Max acceleration at c = 1: 0.5 m/s per 0.1 s = 5 m/s^2.

Drone-pad friction (SPEC D24): PyBullet multiplies the two bodies' lateral friction values; both
were the default 0.5, so the effective value was 0.25. Now pad 1.0 x drone 0.5 = 0.5 (rubber on dry
concrete, sliding value). Only contact is affected, so the paper task (ends at first contact) and
flight are unchanged; it matters for the D20 hold.
"""
import numpy as np
import pybullet as p

from . import dr as DR
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
# ---- SPEC D20: post-touchdown hold (agreed 24-25 Oct 2026) --------------------------
HOLD_STEPS = 10                 # policy steps = 1 s (value used when the hold is on)
HOLD_STILL_STEPS = 5            # last 0.5 s of the hold
V_IMP = 1.0                     # m/s, max |impact v_z| relative to the pad
V_STILL = 0.1                   # m/s, max |v_drone - v_pad_surface| when settled
HOLD_TILT_MAX = np.radians(30.0)
PAD_FRICTION = 1.0              # SPEC D24: lateral friction of the pad body
DRONE_FRICTION = 0.5            # SPEC D24: of the drone; effective drone-pad = product = 0.5
HOLD_OUTCOMES = ("hard_landing", "bounce", "slide_off", "tip_over", "not_still")
PRACTICE_XY = 0.5               # m, |pad-frame offset| of the CoM for an on-pad start (D22)
PRACTICE_GAP = 0.0              # m, gap between the drone's lowest point and the pad top
PRACTICE_VZ = 0.3               # m/s, descent relative to the pad at the start (< V_IMP)


def hold_verdict(vrel_hist, n=HOLD_STILL_STEPS, v_still=V_STILL):
    """End-of-hold decision from the per-step relative speeds (m/s)."""
    last = list(vrel_hist)[-n:]
    return "success" if len(last) == n and max(last) <= v_still else "not_still"


class LandingSim:
    """dr: DR.DRConfig; default = everything on (the paper). Tests of the bare dynamics
    pass DR.DRConfig.off(). Independent RNG streams per source: turning a DR group on or
    off never changes the spawn / platform sequence of a pinned seed."""

    def __init__(self, renderer="tiny", egl=False, marker=True, gains_mode="random", seed=0,
                 dr=None, motor_mode="asym", gui=False, hold_steps=0, pad_smooth=False):
        self.dr = dr if dr is not None else DR.DRConfig()
        self.hold_steps = int(hold_steps)      # 0 = paper task (success at first top contact)
        self.pad_smooth = bool(pad_smooth)     # D23: spread each pad speed change over a step
        self.motor_mode = motor_mode           # 'asym' always in training; others: diagnostic
        self._seed_streams(seed)
        self.cid = make_client(gui=gui, egl=egl)       # gui=True only for scripts/watch.py
        self.ground = Ground(self.cid, randomize=self.dr.visual)
        self.plat = Platform(self.cid, marker=marker)
        # Table II control gains: sampled once per env ("Env. init")
        self.gains = P.sample_gains(self.rng_gains, gains_mode)
        self.quad = LMF2Quad(self.cid, self.rng_gains, self.gains, motor_mode=motor_mode)
        p.changeDynamics(self.plat.body, -1, lateralFriction=PAD_FRICTION, physicsClientId=self.cid)
        p.changeDynamics(self.quad.body, -1, lateralFriction=DRONE_FRICTION, physicsClientId=self.cid)
        self.cam = Camera(self.cid, renderer=renderer)
        self.visual = DR.default_visual()
        self.t = 0
        self.done = True

    def _seed_streams(self, seed):
        ss = np.random.SeedSequence(seed)
        g = [np.random.default_rng(c) for c in ss.spawn(6)]   # 6th added (D22): the first
        self.rng, self.rng_gains, self.rng_dr, self.rng_vis, self.rng_noise = g[:5]   # 5 unchanged
        self.rng_practice = g[5]

    def close(self):
        p.disconnect(self.cid)

    def reseed(self, seed, resample_gains=True):
        """Evaluation helper: per-episode seeding. Training samples gains once per env
        ("Env. init", Table II); evaluation re-samples per episode so a pinned seed
        set covers the whole gain range."""
        from .lee_controller import LeeVelocityController
        from .quad import MotorModel
        self._seed_streams(seed)
        if resample_gains:
            self.gains = P.sample_gains(self.rng_gains)
            self.quad.ctrl = LeeVelocityController(self.gains)
            self.quad.motors = MotorModel(self.rng_gains, mode=self.motor_mode)

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

    def reset(self, c=1.0, spawn=None, on_pad=False):
        """spawn: optional dict(pos, yaw, psi_plat[, roll, pitch, v]) for hand-built tests.
        on_pad: hold-practice start (D22; needs hold_steps > 0)."""
        assert not on_pad or self.hold_steps > 0, "on_pad needs the hold (hold_steps > 0)"
        sp = spawn if spawn is not None else self.sample_spawn(c)
        self.plat.reset(self.rng, c, xy=(0.0, 0.0), psi=sp["psi_plat"])
        # Table II initial state (episode start); a hand-built spawn's "v" overrides
        v0, w0 = DR.sample_initial_rates(self.rng_dr) if self.dr.initial_state \
            else (np.zeros(3), np.zeros(3))
        if "v" in sp:
            v0, w0 = np.asarray(sp["v"], float), np.asarray(sp.get("w_body", (0, 0, 0)), float)
        self.quad.reset_pose(sp["pos"], yaw=sp["yaw"], v=v0, w_body=w0,
                             roll=sp.get("roll", 0.0), pitch=sp.get("pitch", 0.0))
        # Table II visual appearance (episode start)
        if self.dr.visual:
            self.visual = DR.sample_episode_visual(self.rng_vis)
            self.ground.set_appearance(self.visual["tex"], self.visual["scale"],
                                       self.visual["brightness"])
            self.visual["scale"] = self.ground.appearance["scale"]     # discretised value
        else:
            self.visual = DR.default_visual()
        self.quad.ext_force[:] = 0.0
        self.quad.ext_torque[:] = 0.0
        p.performCollisionDetection(physicsClientId=self.cid)
        self.t = 0
        self.done = False
        self.phase = "flight"                  # "hold" after a soft touchdown (D20)
        self.hold_t = 0
        self._vrel_hist = []
        self._td = None                        # touchdown_info at the touchdown substep
        self.practice = bool(on_pad)
        self._pad_from = self._pad_to = (self.plat.v, self.plat.w)    # D23 ramp end points
        if on_pad:
            self._place_on_pad()
        self.spawn = sp
        return dict(spawn=sp, v0=v0, w0=w0, visual=dict(self.visual))

    def _rest_offset(self):
        """CoM height above the drone's lowest point (level attitude), from the collision
        geometry; computed once."""
        if getattr(self, "_rest_off", None) is None:
            body, cid = self.quad.body, self.cid
            pos, q = p.getBasePositionAndOrientation(body, physicsClientId=cid)
            p.resetBasePositionAndOrientation(body, pos, [0, 0, 0, 1], physicsClientId=cid)
            zmin = min(p.getAABB(body, l, physicsClientId=cid)[0][2]
                       for l in range(-1, p.getNumJoints(body, physicsClientId=cid)))
            self._rest_off = pos[2] - zmin
            p.resetBasePositionAndOrientation(body, pos, q, physicsClientId=cid)
        return self._rest_off

    def _place_on_pad(self):
        """D22: drone touching the pad top, moving with the pad surface and descending at
        PRACTICE_VZ relative to it (a soft touchdown), hold phase."""
        rng, plat = self.rng_practice, self.plat
        ox, oy = rng.uniform(-PRACTICE_XY, PRACTICE_XY, 2)
        c, sn = np.cos(plat.psi), np.sin(plat.psi)
        pc = plat.pad_center()
        pos = np.array([pc[0] + c * ox - sn * oy, pc[1] + sn * ox + c * oy,
                        PAD_TOP_Z + self._rest_offset() + PRACTICE_GAP])
        r = pos - pc
        v = plat.velocity() + np.array([-plat.w * r[1], plat.w * r[0], -PRACTICE_VZ])
        self.quad.reset_pose(pos, yaw=float(rng.uniform(-np.pi, np.pi)), v=v,
                             w_body=np.zeros(3))
        p.performCollisionDetection(physicsClientId=self.cid)
        self._v_pre, self._vplat_pre = v.copy(), plat.velocity()
        self._td = self.touchdown_info("success")      # rel_vel = (0, 0, -PRACTICE_VZ)
        self.phase = "hold"

    # ---- step -----------------------------------------------------------------
    def step(self, cmd):
        """cmd = [vx, vy, vz, yaw_rate] in the heading frame, held for one policy
        step. Returns (terminated, truncated, info)."""
        assert not self.done, "call reset()"
        if self.dr.forces:          # Table II: resampled each (policy) step, held 0.1 s
            self.quad.ext_force[:], self.quad.ext_torque[:] = DR.sample_disturbance(self.rng_dr)
        if self.phase == "hold":
            return self._step_hold(cmd)
        outcome, sub = None, P.SUBSTEPS
        for k in range(P.SUBSTEPS):
            # velocities just before this substep: the contact solver zeroes the
            # post-step velocity, so impact speed must be read pre-step
            self._pad_ramp(k)                            # D23 (no-op unless pad_smooth)
            self._v_pre = self.quad.state()["v"]
            self._vplat_pre = self.plat.velocity()
            self.quad.apply_control(cmd)
            p.stepSimulation(physicsClientId=self.cid)   # pad moves with its velocity (SPEC D19)
            self.plat.integrate()                        # re-pin pad to the exact arc
            outcome = self.check_termination()
            if outcome is not None:
                sub = k + 1
                break
        self.t += 1
        if outcome == "success" and self.hold_steps > 0:     # D20: touchdown, not the end
            self._td = self.touchdown_info(outcome)
            if abs(self._td["rel_vel"][2]) > V_IMP:
                outcome = "hard_landing"
            else:
                outcome = None
                self.phase = "hold"
        if outcome is None:
            self._pad_perturb()
        terminated = outcome is not None
        truncated = (not terminated) and self.phase == "flight" and self.t >= HORIZON
        if truncated:
            outcome = "timeout"
        self.done = terminated or truncated
        if self.hold_steps == 0:                         # paper task: unchanged info
            return terminated, truncated, dict(outcome=outcome, substeps=sub, t=self.t,
                                               **self.touchdown_info(outcome))
        info = dict(outcome=outcome, substeps=sub, t=self.t, phase=self.phase,
                    touchdown=self._td is not None, hold_t=self.hold_t, practice=self.practice)
        if self.done:
            info.update(self._terminal_info(outcome))
        return terminated, truncated, info

    # ---- pad speed changes (D19 / D23) -----------------------------------------------
    def _pad_perturb(self):
        """Random-walk step of the pad (once per policy step). Without pad_smooth the new v, w
        apply at once (paper); with it they are reached at the end of the next policy step."""
        plat = self.plat
        v0, w0 = plat.v, plat.w
        plat.perturb(self.rng)
        if self.pad_smooth:
            self._pad_from, self._pad_to = (v0, w0), (plat.v, plat.w)
            plat.v, plat.w = v0, w0
        plat._sync()                                     # speed to the solver (SPEC D19)

    def _pad_ramp(self, k):
        """D23: pad v, w for physics substep k of this policy step (linear ramp)."""
        if not self.pad_smooth:
            return
        f = (k + 1) / P.SUBSTEPS
        (v0, w0), (v1, w1) = self._pad_from, self._pad_to
        self.plat.v, self.plat.w = v0 + f * (v1 - v0), w0 + f * (w1 - w0)
        self.plat._sync()

    def pad_target(self):
        """(v, w) the pad has at the end of the next policy step (the random-walk value)."""
        return self._pad_to if self.pad_smooth else (self.plat.v, self.plat.w)

    # ---- post-touchdown hold (D20) --------------------------------------------------
    def _step_hold(self, cmd):
        outcome, sub, top_seen = None, P.SUBSTEPS, False
        for k in range(P.SUBSTEPS):
            self._pad_ramp(k)
            self._v_pre = self.quad.state()["v"]
            self._vplat_pre = self.plat.velocity()
            self.quad.apply_control(cmd)
            p.stepSimulation(physicsClientId=self.cid)
            self.plat.integrate()
            outcome, top = self.check_hold()
            top_seen = top_seen or top
            if outcome is not None:
                sub = k + 1
                break
        self.t += 1
        if outcome is None and not top_seen:
            outcome = "bounce"
        if outcome is None:
            self.hold_t += 1
            self._vrel_hist.append(self.hold_vrel())
            if self.hold_t >= self.hold_steps:
                outcome = hold_verdict(self._vrel_hist)
        if outcome is None:
            self._pad_perturb()
        self.done = outcome is not None
        info = dict(outcome=outcome, substeps=sub, t=self.t, phase="hold",
                    touchdown=True, hold_t=self.hold_t, practice=self.practice)
        if self.done:
            info.update(self._terminal_info(outcome))
        return self.done, False, info

    def check_hold(self):
        """(outcome or None, pad-top contact this substep). Precedence: side-only
        contact, ground, tip-over, slide-off."""
        cid, drone = self.cid, self.quad.body
        top = side = False
        for c in p.getContactPoints(bodyA=drone, bodyB=self.plat.body, physicsClientId=cid):
            if c[7][2] >= TOP_NORMAL_MIN:
                top = True
            else:
                side = True
        if side and not top:
            return "crash_platform", top
        if p.getContactPoints(bodyA=drone, bodyB=self.ground.body, physicsClientId=cid):
            return "crash_ground", top
        s = self.quad.state()
        if np.arccos(np.clip(s["R"][2, 2], -1, 1)) > HOLD_TILT_MAX:
            return "tip_over", top
        if not self._com_over_pad(s["pos"]):
            return "slide_off", top
        return None, top

    def _com_over_pad(self, pos):
        rel = pos - self.plat.pad_center()
        c, sn = np.cos(self.plat.psi), np.sin(self.plat.psi)
        rel_pad = np.array([c * rel[0] + sn * rel[1], -sn * rel[0] + c * rel[1]])
        return bool(np.all(np.abs(rel_pad) <= PAD_SIZE / 2))

    def hold_vrel(self):
        """|v_drone - v_pad_surface| (m/s); the surface velocity includes the pad's yaw rate."""
        s = self.quad.state()
        r = s["pos"] - self.plat.pad_center()
        w = self.plat.w
        v_surf = self.plat.velocity() + np.array([-w * r[1], w * r[0], 0.0])
        return float(np.linalg.norm(s["v"] - v_surf))

    def _terminal_info(self, outcome):
        """After a touchdown, report the TOUCHDOWN (impact) info plus the hold summary;
        before one, the usual terminal info."""
        if self._td is None:
            return self.touchdown_info(outcome)
        last = self._vrel_hist[-HOLD_STILL_STEPS:]
        return dict(self._td, hold_steps_done=self.hold_t,
                    hold_vrel_max_last=(float(max(last)) if last else float("nan")))

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
        v_rel = self._v_pre - self._vplat_pre          # pre-contact (impact) relative velocity
        return dict(
            rel_pos=rel, rel_pos_pad=rel_pad, rel_vel=v_rel,
            tilt_deg=float(np.degrees(np.arccos(np.clip(s["R"][2, 2], -1, 1)))),
            com_over_pad=bool(np.all(np.abs(rel_pad) <= PAD_SIZE / 2)),
        )

    # ---- sensing ----------------------------------------------------------------
    def render(self):
        """One camera frame with this episode's light direction and RGB scaling."""
        s = self.quad.state()
        return self.cam.render(s["R"], s["pos"], light_direction=self.visual["light_dir"],
                               rgb_scale=self.visual["rgb_scale"])

    def measure(self):
        """u_t = [v_body, q]: noisy if dr.sensor_noise (p.5546)."""
        return DR.measure(self.quad.state(), self.rng_noise, noise=self.dr.sensor_noise)

    # ---- helpers for oracles and tests ------------------------------------------
    def true_relative_state(self):
        """World-frame platform-minus-drone position and velocity (Block 8 rotates
        these into the body frame for s_rel)."""
        s = self.quad.state()
        return self.plat.pad_center() - s["pos"], self.plat.velocity() - s["v"], s
