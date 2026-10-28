"""v6_drop B3: the 60 deg forward-pitched camera.

The marker pad arrives in B4, so here the camera is checked with small black
balls placed at known world points (white background):
  1. intrinsics: 512 x 320, fx = fy = 256 px, 64 deg vertical FOV, axis 60 deg down
  2. the analytic projection (cam.project) matches the rendered image (<= 2 px),
     for level, yawed, rolled and pitched drones
  3. the footprint at the drop moment. C0 (legs): body centre 0.202 m above the pad
     (standing height 0.152 m + 5 cm). A level drone sees the ground from +0.14 m to
     +0.53 m ahead of its centre; 10 deg nose-down +0.11 .. +0.37 m; nose-up +0.18 .. +0.85 m.
     (With V5's cube, 0.30 m: +0.14 .. +0.71, +0.09 .. +0.50, +0.19 .. +1.15 m.)
  4. the point straight below the drone (the pad centre at the drop) is NOT in view
  5. exactly one render per render() call

Each rendering test runs under TinyRenderer, and under EGL if the plugin loads.

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_b3_camera.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import camera as CAM
from v6_drop.envs.camera import Camera
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.envs.quad import make_client

from v6_drop.envs import lmf2_params as P

DROP_CAM_H = P.REST_HEIGHT + 0.05   # m: body centre at the drop = standing height + 5 cm (C0 legs)


def _egl_ok():
    try:
        cid = make_client(egl=True)
        p.disconnect(cid)
        return True
    except Exception:
        return False


RENDERERS = ["tiny", pytest.param("egl", marks=pytest.mark.skipif(
    not _egl_ok(), reason="EGL plugin unavailable"))]


def euler_R(roll_deg, pitch_deg, yaw_deg):
    q = p.getQuaternionFromEuler(np.radians([roll_deg, pitch_deg, yaw_deg]).tolist())
    return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)


def add_ball(cid, xyz, r=0.02):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[0, 0, 0, 1],
                              physicsClientId=cid)
    return p.createMultiBody(0, -1, vis, list(xyz), physicsClientId=cid)


def dark_centroid(img, thr=80):
    vs, us = np.nonzero(img < thr)
    if len(us) == 0:
        return None, 0
    return np.array([us.mean(), vs.mean()]), len(us)


def ground_point_ahead(R, pos, ahead, z=PAD_TOP_Z):
    """World point on the plane z, 'ahead' metres along the drone's heading
    from the point straight below the body centre."""
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([pos[0] + ahead * np.cos(yaw), pos[1] + ahead * np.sin(yaw), z])


# ---------------------------------------------------------------- 1. intrinsics
def test_intrinsics():
    assert CAM.WIDTH == 512 and CAM.HEIGHT == 320
    assert CAM.vfov_deg() == pytest.approx(64.01, abs=0.01)
    cid = make_client()
    try:
        cam = Camera(cid, renderer="tiny")
        assert cam.fx == pytest.approx(256.0) and cam.fy == pytest.approx(256.0)
        assert np.degrees(np.arcsin(-cam.axis_b[2])) == pytest.approx(60.0)
        assert np.allclose(CAM.OFFSET_B, [0.15, 0.0, 0.0])
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- 2. projection == render
# (drone position, roll, pitch, yaw [deg], ball offsets in the heading frame [m])
POSES = {
    "level_2m":     ((0.0, 0.0, PAD_TOP_Z + 2.0), 0, 0, 0,
                     [(1.2, 0.0), (2.5, 1.0), (1.5, -1.2)]),
    "yaw_40":       ((3.0, -1.0, PAD_TOP_Z + 1.5), 0, 0, 40,
                     [(1.0, 0.0), (1.8, 0.6), (0.9, -0.5)]),
    "roll10_pitch": ((0.0, 0.0, PAD_TOP_Z + 3.0), 10, -5, 0,
                     [(2.0, 0.0), (3.5, 1.5), (2.5, -1.5)]),
    "drop_moment":  ((0.0, 0.0, PAD_TOP_Z + DROP_CAM_H), 0, 0, 0,
                     [(0.25, 0.0), (0.40, 0.12), (0.20, -0.08)]),
}


@pytest.mark.parametrize("renderer", RENDERERS)
@pytest.mark.parametrize("name", list(POSES))
def test_projection_matches_render(renderer, name):
    pos, roll, pitch, yaw, offs = POSES[name]
    pos = np.array(pos)
    R = euler_R(roll, pitch, yaw)
    worst = 0.0
    for ox, oy in offs:
        cid = make_client(egl=(renderer == "egl"))
        try:
            c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
            ball = np.array([pos[0] + c * ox - s * oy, pos[1] + s * ox + c * oy, PAD_TOP_Z])
            add_ball(cid, ball)
            cam = Camera(cid, renderer=renderer)
            pred, depth = cam.project(ball, R, pos)
            img = cam.render(R, pos)
            assert img.shape == (cam.h, cam.w) and img.dtype == np.uint8
            got, n = dark_centroid(img)
            assert depth[0] > 0 and got is not None, f"ball not seen ({name}, {ox}, {oy})"
            err = np.linalg.norm(got - pred[0])
            worst = max(worst, err)
        finally:
            p.disconnect(cid)
    print(f"\n[{renderer} {name}] worst ball-centre error {worst:.2f} px")
    assert worst <= 2.0


# ---------------------------------------------------------------- 3. Part A footprint
@pytest.mark.parametrize("pitch_deg,near,far", [(0, 0.14, 0.53), (10, 0.11, 0.37),
                                                 (-10, 0.18, 0.85)])
def test_drop_moment_footprint(pitch_deg, near, far):
    """Analytic footprint check (C0 legs, body centre 0.202 m above the pad).
    Body pitch +10 = nose-down in PyBullet. Tilting also moves the camera (0.15 m
    ahead of the centre) down or up, which the projection includes.
    The point 'near' m ahead must sit on the BOTTOM image edge, 'far' on the TOP edge
    (centre column), within 0.01 m of ground distance (a few px)."""
    cid = make_client()
    try:
        cam = Camera(cid, renderer="tiny")
        R = euler_R(0, pitch_deg, 0)
        pos = np.array([0.0, 0.0, PAD_TOP_Z + DROP_CAM_H])

        def v_of(ahead):
            uv, d = cam.project(ground_point_ahead(R, pos, ahead), R, pos)
            return uv[0, 1], d[0]

        v_near_in, d1 = v_of(near + 0.01)
        v_near_out, _ = v_of(near - 0.01)
        v_far_in, d2 = v_of(far - 0.01)
        v_far_out, _ = v_of(far + 0.01)
        print(f"\n[footprint pitch {pitch_deg:+d}] v at {near-0.01:.2f}/{near+0.01:.2f} m = "
              f"{v_near_out:.0f}/{v_near_in:.0f} px; v at {far-0.01:.2f}/{far+0.01:.2f} m = "
              f"{v_far_in:.0f}/{v_far_out:.0f} px  (image rows 0..{cam.h})")
        assert d1 > 0 and d2 > 0
        assert v_near_in < cam.h < v_near_out          # bottom edge between the two points
        assert v_far_out < 0 < v_far_in                # top edge between the two points
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- 4. blind spot below
@pytest.mark.parametrize("renderer", RENDERERS)
def test_point_below_drone_not_visible(renderer):
    """At the drop moment the pad centre is straight below the body centre.
    It must NOT appear in the image; a point 0.3 m ahead must appear."""
    pos = np.array([0.0, 0.0, PAD_TOP_Z + DROP_CAM_H])
    R = np.eye(3)
    seen = {}
    for ahead in (0.0, 0.3):
        cid = make_client(egl=(renderer == "egl"))
        try:
            add_ball(cid, ground_point_ahead(R, pos, ahead))
            cam = Camera(cid, renderer=renderer)
            seen[ahead] = dark_centroid(cam.render(R, pos))[1]
        finally:
            p.disconnect(cid)
    print(f"\n[{renderer} blind spot] dark pixels: below={seen[0.0]}  0.3 m ahead={seen[0.3]}")
    assert seen[0.0] == 0 and seen[0.3] > 20


# ---------------------------------------------------------------- 5. render count
@pytest.mark.parametrize("renderer", RENDERERS)
def test_one_render_per_call(renderer):
    cid = make_client(egl=(renderer == "egl"))
    try:
        cam = Camera(cid, renderer=renderer)
        for _ in range(7):
            cam.render(np.eye(3), np.array([0.0, 0.0, 3.0]))
        assert cam.n_renders == 7
    finally:
        p.disconnect(cid)
