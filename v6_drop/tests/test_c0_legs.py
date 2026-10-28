"""v6_drop C0: the drone gets legs (decided 28 Oct: "keep it like a real drone").

V5 / AerialGym touched the world with an invisible 0.5 m cube. v6 uses
assets/lmf2_legs.urdf: body, crossed arms, 4 legs under the motors and 4 small
feet; what you see is what touches. Mass and inertia are the LMF2 values.
Sizes are PLACEHOLDERS until the lab drone is measured (lmf2_params.py, C0 block).

Checks:
  1. model: 11 collision shapes, mass and inertia unchanged, feet are the lowest point
  2. standing: on the pad the body centre rests at REST_HEIGHT (0.152 m), level
  3. drop (motors off, matched speed, pad at 4 m/s): impact speed ~ sqrt(2 g h), no bounce
  4. speed mismatch at the drop: the drone slides a little and stops, does not tip
  5. a tilted drop (20 deg) lands upright on its feet
  6. slide vs tip: tipping needs friction > foot half-spread / centre height
     = 0.10 / 0.152 = 0.66; ours is 0.5, so a fast sideways landing slides instead
  7. the drone never sees its own body, arms, legs or propellers: the camera is fixed
     to the body, so tilting does not change that. (Under EGL, its SHADOW on the pad
     can appear in the image; that is realistic and was checked not to hurt detection.)

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_c0_legs.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import lmf2_params as P
from v6_drop.envs.camera import Camera
from v6_drop.envs.platform_s1 import PAD_TOP_Z, PlatformS1, set_contact_friction
from v6_drop.envs.quad import URDF, LMF2Quad, make_client


def make(v_pad=4.0):
    cid = make_client()
    plat = PlatformS1(cid)
    plat.reset(np.random.default_rng(0), v=v_pad)
    rng = np.random.default_rng(0)
    quad = LMF2Quad(cid, rng, P.sample_gains(rng, "nominal"))
    set_contact_friction(cid, plat.body, quad.body)
    return cid, plat, quad


def tilt_deg(quat):
    R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    return np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))


def drop(h, dv=(0.0, 0.0, 0.0), roll_deg=0.0, v_pad=4.0, T=1.5):
    """Feet h metres above the pad, moving with the pad + dv, motors OFF."""
    cid, plat, quad = make(v_pad)
    try:
        q = p.getQuaternionFromEuler([np.radians(roll_deg), 0.0, 0.0])
        p.resetBasePositionAndOrientation(quad.body, [0, 0, 5], q, physicsClientId=cid)
        low = 5 - min(p.getAABB(quad.body, -1, physicsClientId=cid)[0][2], 5)
        pos0 = [0.0, 0.0, PAD_TOP_Z + low + h]
        p.resetBasePositionAndOrientation(quad.body, pos0, q, physicsClientId=cid)
        p.resetBaseVelocity(quad.body, (plat.velocity() + np.array(dv)).tolist(), [0, 0, 0],
                            physicsClientId=cid)
        impact, first, rebound, tmax = None, None, 0.0, 0.0
        z_settle = None
        for k in range(int(round(T / P.PHYSICS_DT))):
            v_pre = quad.state()["v"]                      # read BEFORE the step (V5 rule)
            p.stepSimulation(physicsClientId=cid)
            plat.integrate()
            s = quad.state()
            tmax = max(tmax, tilt_deg(s["quat"]))
            touching = p.getContactPoints(bodyA=quad.body, bodyB=plat.body, physicsClientId=cid)
            if touching and first is None:
                first, impact = k, -v_pre[2]
                z_settle = s["pos"][2]
            if first is not None and k > first + 2:
                rebound = max(rebound, s["pos"][2] - (PAD_TOP_Z + P.REST_HEIGHT))
        s = quad.state()
        return dict(impact=impact, rebound=rebound, tilt_max=tmax, tilt_end=tilt_deg(s["quat"]),
                    vrel=np.linalg.norm(s["v"] - plat.velocity()),
                    slide=plat.world_to_pad([s["pos"]])[0], z=s["pos"][2] - PAD_TOP_Z)
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- 1. model
def test_model_shapes_mass_inertia():
    assert URDF.endswith("lmf2_legs.urdf")
    cid, plat, quad = make()
    try:
        shapes = p.getCollisionShapeData(quad.body, -1, physicsClientId=cid)
        kinds = sorted(s[2] for s in shapes)
        info = p.getDynamicsInfo(quad.body, -1, physicsClientId=cid)
        print(f"\n[model] {len(shapes)} collision shapes; mass {info[0]} kg; inertia {np.round(info[2], 4)}")
        assert len(shapes) == 11
        # 4 feet (spheres), body + 2 arms (boxes), 4 legs: PyBullet stores URDF cylinders
        # as convex meshes, so they report GEOM_MESH.
        assert kinds.count(p.GEOM_SPHERE) == 4 and kinds.count(p.GEOM_BOX) == 3
        assert kinds.count(p.GEOM_MESH) == 4
        assert info[0] == pytest.approx(P.MASS, abs=1e-6)
        assert np.allclose(info[2], P.INERTIA, atol=1e-9)
        p.resetBasePositionAndOrientation(quad.body, [0, 0, 5], [0, 0, 0, 1], physicsClientId=cid)
        low = 5 - p.getAABB(quad.body, -1, physicsClientId=cid)[0][2]
        assert low == pytest.approx(P.REST_HEIGHT, abs=1e-3)      # feet are the lowest point
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- 2. standing
def test_stands_on_feet():
    r = drop(0.0, v_pad=0.0)
    print(f"\n[standing] centre height {r['z']:.4f} m (REST_HEIGHT {P.REST_HEIGHT:.3f}), "
          f"tilt {r['tilt_end']:.2f} deg")
    assert r["z"] == pytest.approx(P.REST_HEIGHT, abs=0.003) and r["tilt_end"] < 0.5


# ---------------------------------------------------------------- 3. drop impact
@pytest.mark.parametrize("h", [0.02, 0.05, 0.10])
def test_drop_impact_matches_free_fall(h):
    r = drop(h)
    theory = np.sqrt(2 * P.G * h)
    print(f"\n[drop {h*100:.0f} cm, pad 4 m/s] impact {r['impact']:.2f} m/s (theory {theory:.2f})  "
          f"rebound {r['rebound']*1000:.1f} mm  final |v_rel| {r['vrel']:.3f}")
    # Resolution: velocity is read once per physics step, and gravity adds g * dt =
    # 0.098 m/s per step, so the reading can be one step early or late (+-0.1 m/s).
    # Sandbox: 2 cm -> 0.69 (theory 0.63), 5 cm -> 0.98 (0.99), 10 cm -> 1.37 (1.40).
    assert r["impact"] == pytest.approx(theory, abs=0.10)
    assert r["rebound"] < 0.002 and r["vrel"] < 0.02 and r["tilt_end"] < 1.0


# ---------------------------------------------------------------- 4. speed mismatch
@pytest.mark.parametrize("dv", [(1.0, 0, 0), (-1.0, 0, 0), (0, 1.0, 0)])
def test_mismatch_slides_and_stops(dv):
    r = drop(0.05, dv=dv)
    print(f"\n[mismatch {dv}] slide to pad-frame {np.round(r['slide'], 3)} m  max tilt "
          f"{r['tilt_max']:.1f} deg  final |v_rel| {r['vrel']:.3f}")
    assert np.linalg.norm(r["slide"]) < 0.2 and r["vrel"] < 0.02 and r["tilt_max"] < 5.0


# ---------------------------------------------------------------- 5. tilted drop
def test_tilted_drop_lands_upright():
    r = drop(0.05, roll_deg=20.0)
    print(f"\n[tilted 20 deg] final tilt {r['tilt_end']:.2f} deg, centre height {r['z']:.3f} m")
    assert r["tilt_end"] < 1.0 and r["z"] == pytest.approx(P.REST_HEIGHT, abs=0.005)


# ---------------------------------------------------------------- 6. slide vs tip
def test_slides_rather_than_tips():
    limit = P.FOOT_XY / P.REST_HEIGHT
    r = drop(0.05, dv=(0, 1.5, 0))
    print(f"\n[slide vs tip] tipping needs friction > {limit:.2f}; ours 0.50 -> max tilt "
          f"{r['tilt_max']:.1f} deg, slide {np.round(r['slide'], 3)} m")
    assert 0.5 < limit and r["tilt_max"] < 5.0


# ---------------------------------------------------------------- 7. self view
@pytest.mark.parametrize("pitch_deg", [0, 15, 30, -15])
def test_never_sees_itself(pitch_deg):
    cid, plat, quad = make(v_pad=0.0)
    try:
        cam = Camera(cid, renderer="tiny")           # TinyRenderer: no shadows
        q = p.getQuaternionFromEuler([0.0, np.radians(pitch_deg), 0.0])
        pos = [0.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + 0.05]
        p.resetBasePositionAndOrientation(quad.body, pos, q, physicsClientId=cid)
        R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
        with_drone = cam.render(R, np.array(pos)).astype(int)
        p.resetBasePositionAndOrientation(quad.body, [50, 50, 50], q, physicsClientId=cid)
        without = cam.render(R, np.array(pos)).astype(int)
        changed = int((np.abs(with_drone - without) > 10).sum())
        print(f"\n[self view pitch {pitch_deg:+d}] pixels showing the drone itself: {changed}")
        assert changed == 0
    finally:
        p.disconnect(cid)
