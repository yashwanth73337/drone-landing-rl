"""v7 S1b: the S500-sized drone (shape only; LMF2 mass, motors, controller).

Checks:
  1. the URDF loads with the LMF2 mass and inertia, friction 0.5 (D24 drone side)
  2. it stands level on the ground with its centre REST_HEIGHT = 0.22 m up
  3. lowest_z() (used later for the feet height and the referee) matches PyBullet's own
     closest distance to the ground for tilted poses (within 1 mm)
  4. a drop from 3 cm onto a moving pad (speed matched) hits at about 0.77 m/s, rocks a
     little (a few degrees) and stays on the pad
  5. TIPPING (important, new with the S500 skids): a drop with a forward or backward speed
     mismatch of up to 0.6 m/s is fine, 1.0 m/s tips the drone over (skids 0.24 m long,
     ASSUMED). The V6 legs did not tip at 1.0 m/s. Sideways, 1.5 m/s only slides.
  6. a drone standing on the pad (motors off) stays on it and level while the pad brakes
     9 -> 3 m/s at 2 m/s^2 and turns a 90 deg corner

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v7/tests/test_s1b_s500.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import lmf2_params as P
from v6_drop.envs.quad import LMF2Quad, make_client
from v7.envs import s500 as S
from v7.envs.platform_v7 import PAD_SIZE, PAD_TOP_Z, PlatformV7, set_contact_friction

DT = P.PHYSICS_DT


def gains():
    return P.sample_gains(np.random.default_rng(0), "nominal")


def tilt_deg(R):
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def ground(cid):
    return p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid),
                             physicsClientId=cid)


def euler_R(r, pch, y):
    return np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([r, pch, y]))).reshape(3, 3)


def on_pad(cls, rest, dv=(0.0, 0.0), gap=0.03, vpad=4.0, plan=None, T=2.0):
    """Drone (motors off) starts gap above a straight-moving pad with a speed mismatch dv
    (forward, sideways); returns max tilt (deg), end position in the pad frame, contact log."""
    cid = make_client()
    q = cls(cid, np.random.default_rng(0), gains())
    pl = PlatformV7(cid)
    set_contact_friction(cid, pl.body, q.body)
    if plan is None:
        pl.reset(np.random.default_rng(0), v=vpad, mode="straight")
    else:
        pl.reset(np.random.default_rng(0), v=vpad, plan=plan)
    q.reset_pose([0.0, 0.0, PAD_TOP_Z + rest + gap], v=(vpad + dv[0], dv[1], 0.0))
    tmax, v_hit, prev_v = 0.0, None, None
    for _ in range(int(round(T / DT))):
        prev_v = q.state()["v"]
        p.stepSimulation(physicsClientId=cid)
        pl.integrate(DT)
        s = q.state()
        if v_hit is None and p.getContactPoints(q.body, pl.body, physicsClientId=cid):
            v_hit = float(-prev_v[2])
        tmax = max(tmax, tilt_deg(s["R"]))
    s = q.state()
    rel = pl.world_to_pad(np.array([s["pos"]]))[0]
    p.disconnect(cid)
    return tmax, rel, v_hit


def test_1_urdf():
    cid = make_client()
    q = S.S500Quad(cid, np.random.default_rng(0), gains())
    info = p.getDynamicsInfo(q.body, -1, physicsClientId=cid)
    p.disconnect(cid)
    assert info[0] == pytest.approx(P.MASS)
    assert np.allclose(info[2], [0.0134, 0.0144, 0.0138])
    assert info[1] == pytest.approx(0.5)
    assert S.REST_HEIGHT == pytest.approx(0.22)
    assert S.MOTOR_XY == pytest.approx(0.1768, abs=1e-4)


def test_2_stands_level():
    cid = make_client()
    ground(cid)
    q = S.S500Quad(cid, np.random.default_rng(0), gains())
    q.reset_pose([0, 0, S.REST_HEIGHT + 0.03])
    for _ in range(300):
        p.stepSimulation(physicsClientId=cid)
    s = q.state()
    low = S.lowest_z(s["R"], s["pos"])
    p.disconnect(cid)
    print(f"\n  standing: centre {s['pos'][2]:.4f} m, tilt {tilt_deg(s['R']):.3f} deg, lowest {low * 1000:.2f} mm")
    assert s["pos"][2] == pytest.approx(S.REST_HEIGHT, abs=0.002)
    assert tilt_deg(s["R"]) < 0.5 and abs(low) < 0.001


def test_3_lowest_point_matches_pybullet():
    """PyBullet's closest distance from the drone to the ground plane = lowest_z (a
    bounding box is too loose for rotated capsules, so we ask PyBullet for the distance)."""
    cid = make_client()
    g = ground(cid)
    q = S.S500Quad(cid, np.random.default_rng(0), gains())
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(50):
        r, pch = rng.uniform(-0.5, 0.5, 2)
        pos = np.array([0.0, 0.0, 0.5])
        q.reset_pose(pos.tolist(), yaw=rng.uniform(-np.pi, np.pi), roll=r, pitch=pch)
        R = q.state()["R"]
        d = min(c[8] for c in p.getClosestPoints(q.body, g, 1.0, physicsClientId=cid))
        worst = max(worst, abs(S.lowest_z(R, pos) - d))
    p.disconnect(cid)
    print(f"\n  lowest_z vs PyBullet distance, 50 tilted poses up to 29 deg: worst {worst * 1000:.2f} mm")
    assert worst < 0.001


def test_4_drop_3cm():
    tmax, rel, v_hit = on_pad(S.S500Quad, S.REST_HEIGHT, gap=0.03)
    print(f"\n  3 cm drop on a 4 m/s pad: hit {v_hit:.2f} m/s (formula 0.77), tilt max {tmax:.2f} deg")
    assert v_hit == pytest.approx(np.sqrt(2 * P.G * 0.03), abs=0.1)
    assert tmax < 5.0 and np.all(np.abs(rel) < PAD_SIZE / 2)


def test_5_tipping_with_speed_mismatch():
    s500, legs = {}, {}
    for dv in (0.3, 0.6, 1.0):
        s500[dv] = max(on_pad(S.S500Quad, S.REST_HEIGHT, (dv, 0))[0],
                       on_pad(S.S500Quad, S.REST_HEIGHT, (-dv, 0))[0])
    legs[1.0] = max(on_pad(LMF2Quad, P.REST_HEIGHT, (1.0, 0))[0],
                    on_pad(LMF2Quad, P.REST_HEIGHT, (-1.0, 0))[0])
    side = on_pad(S.S500Quad, S.REST_HEIGHT, (0, 1.5))[0]
    print("\n  3 cm drop, forward/backward mismatch, max tilt (deg): S500 "
          + ", ".join(f"{k} m/s: {v:.0f}" for k, v in s500.items())
          + f"; V6 legs 1.0 m/s: {legs[1.0]:.0f}; S500 sideways 1.5 m/s: {side:.0f}")
    assert s500[0.3] < 30 and s500[0.6] < 30          # within the drop rule (0.3 m/s) and more
    assert s500[1.0] > 30                             # tips: the S500 skids are short
    assert legs[1.0] < 30                             # the V6 legs did not tip
    assert side < 30                                  # sideways it only slides


def test_6_standing_through_braking_and_corner():
    plan = [("keep", 0.5), ("speed", 3.0, 2.0), ("turn", np.pi / 2, 6.0, "corner-turn"), ("keep", 1.0)]
    T = 0.5 + 3.0 + (np.pi / 2) * 6.0 / 3.0 + 1.0
    tmax, rel, _ = on_pad(S.S500Quad, S.REST_HEIGHT, gap=0.001, vpad=9.0, plan=plan, T=T)
    print(f"\n  standing, pad brakes 9 -> 3 m/s then turns 90 deg: tilt max {tmax:.2f} deg, "
          f"end {np.round(rel, 3)} m from the pad centre")
    assert tmax < 5.0 and np.all(np.abs(rel) < PAD_SIZE / 2)
