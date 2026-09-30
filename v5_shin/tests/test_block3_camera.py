"""Block 3: pitched camera, ArUco pad, ground, both renderers.

Main check: OpenCV detects ArUco id 0 in the rendered frame, and the detected
corners match the analytic pinhole projection of the pad-frame marker corners
(max error <= 2 px, correct corner order) at several drone and platform poses.
This validates the camera pose, mount pitch, intrinsics, texture orientation and
the platform/marker sync in one go. Each test runs under TinyRenderer and EGL
(EGL is skipped if the plugin cannot load).

Run:  python -m pytest v5_shin/tests/test_block3_camera.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v5_shin.envs import camera as CAM
from v5_shin.envs import pad_marker as PM
from v5_shin.envs.camera import Camera
from v5_shin.envs.ground import Ground
from v5_shin.envs.platform import PAD_TOP_Z, Platform
from v5_shin.envs.quad import make_client


def _egl_ok():
    try:
        cid = make_client(egl=True)
        p.disconnect(cid)
        return True
    except Exception:
        return False


RENDERERS = ["tiny", pytest.param("egl", marks=pytest.mark.skipif(
    not _egl_ok(), reason="EGL plugin unavailable"))]


def euler_R(roll, pitch, yaw):
    q = p.getQuaternionFromEuler([roll, pitch, yaw])
    return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)


@pytest.fixture
def scene(request):
    r = request.param
    cid = make_client(egl=(r == "egl"))
    ground = Ground(cid)
    plat = Platform(cid)
    cam = Camera(cid, renderer=r)
    yield cid, ground, plat, cam
    p.disconnect(cid)


def test_intrinsics():
    assert CAM.vfov_deg() == pytest.approx(64.01, abs=0.01)
    cid = make_client()
    try:
        cam = Camera(cid, renderer="tiny")
        assert cam.fx == pytest.approx(256.0) and cam.fy == pytest.approx(256.0)
        # optical axis 60 deg below body +x
        assert np.degrees(np.arcsin(-cam.axis_b[2])) == pytest.approx(60.0)
    finally:
        p.disconnect(cid)


# (drone xyz relative to the pad centre, roll, pitch, yaw [deg], pad world xy, pad psi [deg])
POSES = {
    "on_axis_3m":   ((-1.73, 0.0, 3.0), 0, 0, 0, (0, 0), 0),
    "on_axis_6m":   ((-3.46, 0.0, 6.0), 0, 0, 0, (0, 0), 0),
    "yaw30_pad40":  ((-1.0, -0.58, 2.0), 0, 0, 30, (0, 0), 40),
    "roll10_pitch": ((-2.0, 0.3, 3.0), 10, -5, 0, (0, 0), 0),
    "moved_pad":    ((-1.5, 0.87, 3.0), 0, 0, -30, (5, -3), 120),
}


@pytest.mark.parametrize("scene", RENDERERS, indirect=True)
@pytest.mark.parametrize("name", list(POSES))
def test_detection_matches_projection(scene, name):
    cid, _, plat, cam = scene
    rel, roll, pitch, yaw, pad_xy, psi = POSES[name]
    plat.reset(np.random.default_rng(0), 0.0, xy=pad_xy, psi=np.radians(psi))
    R = euler_R(*np.radians([roll, pitch, yaw]))
    pos = np.array([pad_xy[0] + rel[0], pad_xy[1] + rel[1], PAD_TOP_Z + rel[2]])
    pred, depth = cam.project(plat.pad_to_world(PM.MARKER_CORNERS_PAD), R, pos)
    assert (depth > 0).all()
    assert ((pred >= 5) & (pred <= [cam.w - 5, cam.h - 5])).all(), "pose must keep marker in view"
    img = cam.render(R, pos)
    assert img.shape == (cam.h, cam.w) and img.dtype == np.uint8
    corners, ids, _ = PM.detector().detectMarkers(img)
    assert ids is not None and list(ids.ravel()) == [PM.MARKER_ID], f"no detection ({name})"
    err = np.abs(corners[0][0] - pred).max()
    size = np.linalg.norm(pred[0] - pred[2])
    print(f"\n[{cam.renderer} {name}] marker diag {size:.0f} px, max corner err {err:.2f} px")
    assert err <= 2.0      # order is implied: corner i is compared with corner i


@pytest.mark.parametrize("scene", RENDERERS, indirect=True)
def test_one_render_per_call(scene):
    _, _, plat, cam = scene
    plat.reset(np.random.default_rng(0), 0.0)
    for _ in range(7):
        cam.render(np.eye(3), np.array([-1.73, 0, 4.0]))
    assert cam.n_renders == 7


@pytest.mark.parametrize("scene", RENDERERS, indirect=True)
def test_ground_texture_swap_keeps_marker(scene, tmp_path):
    """EGL regression: swapping the ground texture must not corrupt the pad texture."""
    import cv2
    _, ground, plat, cam = scene
    plat.reset(np.random.default_rng(0), 0.0)
    alt = np.zeros((128, 128), np.uint8)
    alt[::16] = 255
    path = str(tmp_path / "alt.png")
    cv2.imwrite(path, cv2.cvtColor(alt, cv2.COLOR_GRAY2BGR))
    before = cam.render(np.eye(3), np.array([-1.73, 0, 4.0]))
    ground.set_texture(path)
    after = cam.render(np.eye(3), np.array([-1.73, 0, 4.0]))
    assert np.abs(before[:60].astype(int) - after[:60]).mean() > 5   # ground changed
    _, ids, _ = PM.detector().detectMarkers(after)
    assert ids is not None and list(ids.ravel()) == [0]


def test_platform_single_body_box_collision():
    """The marker is texture only: the platform is one body whose collision shape
    is the 1.5 x 1.5 x 1.0 box (no extra geometry that could affect contacts)."""
    cid = make_client()
    try:
        n0 = p.getNumBodies(physicsClientId=cid)
        plat = Platform(cid)
        assert p.getNumBodies(physicsClientId=cid) == n0 + 1
        data = p.getCollisionShapeData(plat.body, -1, physicsClientId=cid)
        assert len(data) == 1 and data[0][2] == p.GEOM_BOX
        assert np.allclose(data[0][3], [1.5, 1.5, PAD_TOP_Z])
    finally:
        p.disconnect(cid)


def test_detection_all_spawn_altitudes():
    """Table I spawns at 2-8 m above the pad. The marker must be detected on the optical
    axis at every altitude under every available renderer (EGL z-fighting regression)."""
    rs = ["tiny"] + (["egl"] if _egl_ok() else [])
    for r in rs:
        cid = make_client(egl=(r == "egl"))
        try:
            Ground(cid)
            plat = Platform(cid)
            plat.reset(np.random.default_rng(0), 0.0)
            cam = Camera(cid, renderer=r)
            miss = []
            for a in np.arange(2.0, 8.01, 0.5):
                pos = np.array([-a / np.tan(np.radians(60)), 0, PAD_TOP_Z + a])
                _, ids, _ = PM.detector().detectMarkers(cam.render(np.eye(3), pos))
                if ids is None:
                    miss.append(a)
            print(f"\n[{r}] altitudes 2-8 m, missed: {miss}")
            assert not miss
        finally:
            p.disconnect(cid)


def test_texture_format():
    """EGL needs RGB textures whose width is a multiple of 4."""
    import cv2
    PM.ensure_assets()
    img = cv2.imread(PM.TEX_PATH, cv2.IMREAD_UNCHANGED)
    assert img.ndim == 3 and img.shape[2] == 3 and img.shape[1] % 4 == 0
