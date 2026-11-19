"""v7 S1c: the gimbal camera.

Checks:
  1. same image as V6: 512 x 320, fx = fy = 256 px, 64 deg tall view
  2. STEADY: for any lean (pitch, roll up to 30 deg) the camera looks exactly 60 deg below
     the horizon and points along the drone's heading
  3. teaching check: the top edge of the view. Drone leaning 17 deg nose-down (speeding
     up at 3 m/s^2): gimbal top edge still 28 deg below the horizon (sees 1.88 x height
     ahead); the V6 fixed camera's top edge drops to 45 deg (sees only 1.0 x height)
  4. the calculated pixel of a black ball matches the rendered image (<= 2 px), for a
     level and a leaning drone (TinyRenderer, and EGL if it loads)
  5. SELF-VIEW: with the mount 0.15 m forward and 0.10 m below the body centre (ASSUMED),
     the camera never sees the drone's own skids for leans up to 10 deg; up to 20 deg it
     sees a little of them at the image bottom (60 deg: <= 2 % of the image; 45 deg: 0)
  6. DROP MOMENT (feet 3 cm above the pad, pad markers L70_09): markers found and the
     feet height measured within 1 cm wherever the drone is within 0.2 m of the pad centre;
     at 45 deg also 0.3 to 0.4 m behind it

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v7/tests/test_s1c_gimbal.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import lmf2_params as P
from v6_drop.envs import pad_board as PB
from v6_drop.envs.camera import Camera
from v6_drop.envs.quad import make_client
from v6_drop.envs.reward import geometry_from_body
from v7.envs.gimbal_camera import GimbalCamera, heading, measure_gimbal
from v7.envs.platform_v7 import PAD_TOP_Z, textured_platform_v7
from v7.envs.s500 import REST_HEIGHT, S500Quad


def _egl_ok():
    try:
        cid = make_client(egl=True)
        p.disconnect(cid)
        return True
    except Exception:
        return False


RENDERERS = ["tiny", pytest.param("egl", marks=pytest.mark.skipif(not _egl_ok(), reason="no EGL"))]


def euler_R(roll_deg, pitch_deg, yaw_deg):
    q = p.getQuaternionFromEuler(np.radians([roll_deg, pitch_deg, yaw_deg]).tolist())
    return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)


def below_horizon_deg(v):
    return float(np.degrees(np.arcsin(-v[2] / np.linalg.norm(v))))


def test_1_same_image():
    cid = make_client()
    cam = GimbalCamera(cid, renderer="tiny")
    p.disconnect(cid)
    assert (cam.w, cam.h) == (512, 320)
    assert cam.fx == pytest.approx(256.0) and cam.fy == pytest.approx(256.0)
    assert cam.vfov == pytest.approx(64.01, abs=0.01)


def test_2_steady():
    cid = make_client()
    cam = GimbalCamera(cid, renderer="tiny")
    worst_angle, worst_head = 0.0, 0.0
    for r in (-30, -10, 0, 10, 30):
        for pt in (-30, -17, 0, 17, 30):
            for yw in (-120, 0, 45):
                R = euler_R(r, pt, yw)
                _, fwd, up = cam.pose_world(R, np.zeros(3))
                worst_angle = max(worst_angle, abs(below_horizon_deg(fwd) - 60.0))
                h = np.degrees(np.arctan2(fwd[1], fwd[0]))
                worst_head = max(worst_head, abs((h - np.degrees(heading(R)) + 180) % 360 - 180))
                assert abs(np.dot(fwd, up)) < 1e-12 and up[2] > 0
    p.disconnect(cid)
    print(f"\n  75 leans: angle below horizon off by at most {worst_angle:.2e} deg, "
          f"heading off by at most {worst_head:.2e} deg")
    assert worst_angle < 1e-9 and worst_head < 1e-6


def test_3_top_edge_when_leaning():
    cid = make_client()
    gim, fixed = GimbalCamera(cid, renderer="tiny"), Camera(cid, renderer="tiny")
    half = np.radians(fixed.vfov / 2)
    R = euler_R(0, 17, 0)                                    # 17 deg nose-down
    out = {}
    for name, cam in (("gimbal", gim), ("fixed V6", fixed)):
        _, fwd, up = cam.pose_world(R, np.zeros(3))
        top = np.cos(half) * fwd + np.sin(half) * up
        out[name] = below_horizon_deg(top)
    p.disconnect(cid)
    ratio = {k: 1 / np.tan(np.radians(v)) for k, v in out.items()}
    print(f"\n  17 deg nose-down: top edge {out['gimbal']:.1f} deg below the horizon (gimbal, "
          f"sees {ratio['gimbal']:.2f} x height ahead) vs {out['fixed V6']:.1f} deg (fixed, "
          f"{ratio['fixed V6']:.2f} x height)")
    assert out["gimbal"] == pytest.approx(60 - fixed.vfov / 2, abs=0.01)
    assert out["fixed V6"] == pytest.approx(60 + 17 - fixed.vfov / 2, abs=0.01)


@pytest.mark.parametrize("renderer", RENDERERS)
@pytest.mark.parametrize("lean", [(0, 0), (0, 17), (12, -10)])
def test_4_projection_matches_render(renderer, lean):
    pos = np.array([0.0, 0.0, PAD_TOP_Z + 2.0])
    R = euler_R(lean[0], lean[1], 30)
    worst = 0.0
    for ahead, side in ((1.3, 0.0), (2.4, 0.9), (1.6, -1.0)):
        cid = make_client(egl=(renderer == "egl"))
        try:
            y = heading(R)
            ball = np.array([pos[0] + ahead * np.cos(y) - side * np.sin(y),
                             pos[1] + ahead * np.sin(y) + side * np.cos(y), PAD_TOP_Z])
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 0, 0, 1], physicsClientId=cid)
            p.createMultiBody(0, -1, vis, ball.tolist(), physicsClientId=cid)
            cam = GimbalCamera(cid, renderer=renderer)
            pred, depth = cam.project(ball, R, pos)
            img = cam.render(R, pos)
            vs, us = np.nonzero(img < 80)
            assert depth[0] > 0 and len(us) > 0
            worst = max(worst, float(np.linalg.norm([us.mean() - pred[0, 0], vs.mean() - pred[0, 1]])))
        finally:
            p.disconnect(cid)
    print(f"\n  [{renderer} lean {lean}] worst ball error {worst:.2f} px")
    assert worst <= 2.0


def _own_pixels(cid, q, cam, roll, pitch):
    pos = np.array([0.0, 0.0, 3.0])
    q.reset_pose(pos.tolist(), yaw=0.3, roll=np.radians(roll), pitch=np.radians(pitch))
    R = q.state()["R"]
    eye, fwd, up = cam.pose_world(R, pos)
    view = p.computeViewMatrix(eye.tolist(), (eye + fwd).tolist(), up.tolist())
    seg = p.getCameraImage(cam.w, cam.h, view, cam.proj, renderer=p.ER_TINY_RENDERER,
                           physicsClientId=cid)[4]
    return int((np.asarray(seg).reshape(cam.h, cam.w) == q.body).sum())


def test_5_self_view():
    cid = make_client()
    q = S500Quad(cid, np.random.default_rng(0), P.sample_gains(np.random.default_rng(0), "nominal"))
    total = 512 * 320
    res = {}
    for pitch_deg in (60, 45):
        cam = GimbalCamera(cid, pitch_deg=pitch_deg, renderer="tiny")
        for lim in (10, 20):
            res[(pitch_deg, lim)] = max(_own_pixels(cid, q, cam, r, pt) for r in (-lim, 0, lim)
                                        for pt in (-lim, -lim / 2, 0, lim / 2, lim))
    p.disconnect(cid)
    print("\n  own-drone pixels (max over leans): "
          + ", ".join(f"{a} deg camera, lean <= {b}: {n} ({100 * n / total:.1f} %)" for (a, b), n in res.items()))
    assert res[(60, 10)] == 0 and res[(45, 10)] == 0
    assert res[(60, 20)] <= 0.02 * total and res[(45, 20)] == 0


def test_6_drop_moment_markers():
    renderer = "egl" if _egl_ok() else "tiny"      # decide BEFORE opening (one EGL client per process)
    cid = make_client(egl=(renderer == "egl"))
    ground = p.createVisualShape(p.GEOM_BOX, halfExtents=[50, 50, 0.001], rgbaColor=[0.45, 0.45, 0.45, 1],
                                 physicsClientId=cid)
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid), ground, physicsClientId=cid)
    pl = textured_platform_v7(cid)
    pl.reset(np.random.default_rng(0), v=0.0, mode="straight")
    q = S500Quad(cid, np.random.default_rng(0), P.sample_gains(np.random.default_rng(0), "nominal"))
    det = PB.detector()
    out = {}
    for pitch_deg in (60, 45):
        cam = GimbalCamera(cid, pitch_deg=pitch_deg, renderer=renderer)
        rows = []
        for dx in (-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
            for dy in (-0.2, 0.0, 0.2):
                for r, pt in ((0, 0), (0, 8), (0, -8), (8, 0), (-8, 0)):
                    pos = np.array([dx, dy, PAD_TOP_Z + REST_HEIGHT + 0.03])
                    q.reset_pose(pos.tolist(), yaw=0.0, roll=np.radians(r), pitch=np.radians(pt))
                    R = q.state()["R"]
                    m = measure_gimbal(cam.render(R, pos), cam, PB.DEFAULT_LAYOUT, det)
                    err = None
                    if m.ok and m.confident:
                        err = abs(geometry_from_body(m.p_body, R)[1] - REST_HEIGHT - 0.03)
                    rows.append((dx, err))
        out[pitch_deg] = rows
    p.disconnect(cid)
    for pitch_deg, rows in out.items():
        near = [e for dx, e in rows if dx >= -0.2]
        far = [e for dx, e in rows if dx < -0.2]
        errs = [e for e in near if e is not None]
        print(f"\n  [{renderer}] {pitch_deg} deg camera at the drop: within 0.2 m of the centre "
              f"{len(errs)}/{len(near)} found (height error max {1000 * max(errs):.1f} mm); "
              f"0.3 to 0.4 m behind: {sum(e is not None for e in far)}/{len(far)} found")
        assert len(errs) == len(near) and max(errs) < 0.01
    assert all(e is not None for dx, e in out[45])
