"""v6_drop B2: the Stage 1 platform.

Checks:
  1. speed is drawn from U(0, 4) m/s, one value per episode
  2. the pad moves exactly v * t along its heading, and never turns
  3. PyBullet's solver sees the pad velocity (D19)
  4. a drone resting on the moving pad (motors off) is carried along (D19);
     and, as a teaching check, WITHOUT D19 it slides off the back
  5. PyBullet multiplies friction values (D24): 0.5 x 0.5 -> 0.25, 1.0 x 0.5 -> 0.5

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_b2_platform.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import lmf2_params as P
from v6_drop.envs.quad import LMF2Quad, make_client
from v6_drop.envs.platform_s1 import (DRONE_FRICTION, PAD_FRICTION, PAD_SIZE, PAD_TOP_Z,
                                      V_MAX, PlatformS1, set_contact_friction)

REST_Z = PAD_TOP_Z + P.REST_HEIGHT + 0.001   # drone centre when standing on the pad top (+1 mm)
# C0 (28 Oct): the drone stands on four feet (REST_HEIGHT 0.152 m), not on V5's 0.5 m cube.


def make_world():
    cid = make_client()
    plat = PlatformS1(cid)
    rng = np.random.default_rng(0)
    quad = LMF2Quad(cid, rng, P.sample_gains(rng, "nominal"))
    set_contact_friction(cid, plat.body, quad.body)
    return cid, plat, quad


def place_on_pad(quad, plat, rel_xy=(0.0, 0.0), rel_v=(0.0, 0.0)):
    """Drone resting on the pad top, moving with the pad plus rel_v (world xy)."""
    pos = plat.pad_center() + np.array([rel_xy[0], rel_xy[1], 0.0])
    pos[2] = REST_Z
    v = plat.velocity() + np.array([rel_v[0], rel_v[1], 0.0])
    quad.reset_pose(pos.tolist(), yaw=plat.psi, v=v.tolist())


def run_motors_off(cid, plat, quad, T):
    """Motors OFF: no control force, only gravity, contact and friction."""
    log = []
    for _ in range(int(round(T / P.PHYSICS_DT))):
        p.stepSimulation(physicsClientId=cid)
        plat.integrate()
        s = quad.state()
        log.append((s["pos"].copy(), s["v"].copy(), plat.pad_center(), plat.velocity()))
    return log


# ---------------------------------------------------------------- 1. speed sampling
def test_speed_sampling():
    cid = make_client()
    try:
        plat, rng = PlatformS1(cid), np.random.default_rng(123)
        vs = []
        for _ in range(4000):
            plat.reset(rng)
            vs.append(plat.v)
        vs = np.array(vs)
        quarters = np.histogram(vs, bins=4, range=(0, V_MAX))[0]
        print(f"\n[speed] min={vs.min():.3f} max={vs.max():.3f} mean={vs.mean():.3f} "
              f"per 1 m/s bin={quarters.tolist()}")
        assert vs.min() >= 0.0 and vs.max() <= V_MAX
        assert vs.mean() == pytest.approx(V_MAX / 2, abs=0.08)
        assert np.all(np.abs(quarters - 1000) < 120)      # roughly uniform
    finally:
        p.disconnect(cid)


def test_forced_speed():
    cid = make_client()
    try:
        plat = PlatformS1(cid)
        plat.reset(np.random.default_rng(0), v=2.5)
        assert plat.v == 2.5
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- 2. straight motion
@pytest.mark.parametrize("v,psi_deg", [(0.0, 0.0), (3.0, 0.0), (4.0, 30.0), (1.5, -120.0)])
def test_straight_line_exact(v, psi_deg):
    cid = make_client()
    try:
        plat = PlatformS1(cid)
        psi = np.radians(psi_deg)
        plat.reset(np.random.default_rng(0), xy=(1.0, -2.0), psi=psi, v=v)
        T = 2.0
        for _ in range(int(round(T / P.PHYSICS_DT))):
            p.stepSimulation(physicsClientId=cid)
            plat.integrate()
        expect = np.array([1.0, -2.0]) + v * T * np.array([np.cos(psi), np.sin(psi)])
        pos, quat = p.getBasePositionAndOrientation(plat.body, physicsClientId=cid)
        yaw = p.getEulerFromQuaternion(quat)[2]
        err = np.linalg.norm(np.array(pos[:2]) - expect)
        print(f"\n[line v={v} psi={psi_deg}] end=({pos[0]:.4f}, {pos[1]:.4f}) "
              f"expected=({expect[0]:.4f}, {expect[1]:.4f}) err={err:.2e} m")
        assert err < 1e-6
        assert pos[2] == pytest.approx(PAD_TOP_Z / 2, abs=1e-9)
        assert yaw == pytest.approx(psi, abs=1e-9)
        assert plat.psi == pytest.approx(psi, abs=1e-12)      # never turns
    finally:
        p.disconnect(cid)


def test_frames_roundtrip():
    cid = make_client()
    try:
        plat = PlatformS1(cid)
        plat.reset(np.random.default_rng(0), xy=(3.0, 1.0), psi=np.radians(40), v=1.0)
        pts = np.array([[0.3, 0.0], [0.0, -0.5], [-0.6, 0.7]])
        back = plat.world_to_pad(plat.pad_to_world(pts))
        assert np.allclose(back, pts, atol=1e-12)
        fwd = plat.pad_to_world([[1.0, 0.0]])[0] - plat.pad_center()
        assert np.allclose(fwd[:2], [np.cos(np.radians(40)), np.sin(np.radians(40))])
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- 3. solver velocity (D19)
def test_solver_sees_pad_velocity():
    cid = make_client()
    try:
        plat = PlatformS1(cid)
        plat.reset(np.random.default_rng(0), psi=np.radians(25), v=3.0)
        p.stepSimulation(physicsClientId=cid)
        plat.integrate()
        lin, ang = p.getBaseVelocity(plat.body, physicsClientId=cid)
        print(f"\n[D19] solver velocity={np.round(lin, 4)}  class velocity={np.round(plat.velocity(), 4)}")
        assert np.allclose(lin, plat.velocity(), atol=1e-9)
        assert np.allclose(ang, 0.0)
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- 4. carry (D19)
@pytest.mark.parametrize("v", [0.0, 2.0, 4.0])
def test_resting_drone_is_carried(v):
    """Motors off, drone placed on the pad with the pad's velocity. After 2 s it
    must still be on the pad, near where it started (pad frame), moving with it."""
    cid, plat, quad = make_world()
    try:
        plat.reset(np.random.default_rng(0), v=v)
        place_on_pad(quad, plat)
        log = run_motors_off(cid, plat, quad, T=2.0)
        pos, vel, pc, pv = log[-1]
        slip = np.linalg.norm(pos[:2] - pc[:2])
        vrel = np.linalg.norm(vel - pv)
        height = pos[2] - PAD_TOP_Z
        print(f"\n[carry v={v}] slip after 2 s={slip*100:.2f} cm  |v_rel|={vrel:.4f} m/s  "
              f"centre height above pad={height:.4f} m")
        assert slip < 0.02 and vrel < 0.02
        assert height == pytest.approx(P.REST_HEIGHT, abs=0.01)   # standing, not sinking or bouncing
    finally:
        p.disconnect(cid)


class _TeleportOnly(PlatformS1):
    """The V5 bug before D19: the pose is set, the velocity is NOT (solver sees 0)."""
    def sync(self):
        q = p.getQuaternionFromEuler([0, 0, self.psi])
        p.resetBasePositionAndOrientation(self.body, [self.x, self.y, PAD_TOP_Z / 2], q,
                                          physicsClientId=self.cid)
        p.resetBaseVelocity(self.body, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)


def test_without_d19_drone_is_left_behind():
    """Teaching check: the same 4 m/s carry with the pre-D19 platform. The drone
    keeps its own velocity, friction acts against a 'still' pad, and the pad
    slides away under it. It must end up far behind (or off the pad)."""
    cid = make_client()
    try:
        plat = _TeleportOnly(cid)
        rng = np.random.default_rng(0)
        quad = LMF2Quad(cid, rng, P.sample_gains(rng, "nominal"))
        set_contact_friction(cid, plat.body, quad.body)
        plat.reset(rng, v=4.0)
        place_on_pad(quad, plat)
        log = run_motors_off(cid, plat, quad, T=2.0)
        pos, _, pc, _ = log[-1]
        lag = plat.world_to_pad([pos])[0, 0]                 # pad-frame x; negative = behind
        print(f"\n[no D19, v=4] drone pad-frame x after 2 s = {lag:+.2f} m "
              f"(pad half-size {PAD_SIZE/2} m)")
        assert lag < -PAD_SIZE / 2                            # slid off the back
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- 5. friction (D24)
@pytest.mark.parametrize("pad_mu,drone_mu", [(0.5, 0.5), (PAD_FRICTION, DRONE_FRICTION)])
def test_friction_is_multiplied(pad_mu, drone_mu):
    """Still pad, drone sliding at 1 m/s, motors off. Sliding friction decelerates it
    at mu_eff * g. PyBullet uses mu_eff = pad_mu * drone_mu (a product)."""
    cid, plat, quad = make_world()
    try:
        p.changeDynamics(plat.body, -1, lateralFriction=pad_mu, physicsClientId=cid)
        p.changeDynamics(quad.body, -1, lateralFriction=drone_mu, physicsClientId=cid)
        plat.reset(np.random.default_rng(0), v=0.0)
        place_on_pad(quad, plat, rel_v=(1.0, 0.0))
        log = run_motors_off(cid, plat, quad, T=0.1)
        v_end = log[-1][1][0]
        decel = (1.0 - v_end) / 0.1
        expect = pad_mu * drone_mu * P.G
        print(f"\n[friction {pad_mu} x {drone_mu}] decel={decel:.2f} m/s^2  "
              f"product rule={expect:.2f}  min rule={min(pad_mu, drone_mu) * P.G:.2f}")
        assert decel == pytest.approx(expect, rel=0.15)
    finally:
        p.disconnect(cid)


def test_friction_values_set():
    cid, plat, quad = make_world()
    try:
        assert p.getDynamicsInfo(plat.body, -1, physicsClientId=cid)[1] == PAD_FRICTION
        assert p.getDynamicsInfo(quad.body, -1, physicsClientId=cid)[1] == DRONE_FRICTION
    finally:
        p.disconnect(cid)
