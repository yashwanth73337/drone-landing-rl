"""v6_drop B4: the multi-marker pad and pose estimation.
Layout L70_09 since C0 (28 Oct; legged drone, lower camera). It was L70_12 in B4.

Checks:
  1. layout: 31 unique ids, everything on the pad with a margin, big marker in the
     rear half, small grid in the front half, quiet zones between markers
  2. texture: RGB, width % 4 == 0; OpenCV finds all 31 markers on the flat texture
     at the right pad-frame positions (<= 2 mm)
  3. rendering: detected corners match the analytic projection (<= 2.5 px), so the
     texture orientation, platform pose and camera agree
  4. pose: measure() returns the pad centre and orientation in the body frame
     (near: < 1 cm and < 2 deg; 3 m away: < 10 cm and < 3 deg)
  5. drop moment (feet 5 cm above the pad): 27 cases (tilt -10/0/+10 deg, offsets
     -0.15/0/+0.15 m along and across), a small marker is always detected and the pad
     centre error is < 1 cm
  7. (C0) markers below MIN_SIDE_PX are used only when nothing bigger is seen
  6. the platform is still ONE body with the 1.5 x 1.5 x 1.0 box collision shape
     (the board is texture only)
The full detection-range table is produced by scripts/detect_range.py.

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_b4_pad.py -v -s
"""
import itertools

import cv2
import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import lmf2_params as P
from v6_drop.envs import pad_board as PB
from v6_drop.envs.camera import Camera
from v6_drop.envs.platform_s1 import PAD_SIZE, PAD_TOP_Z
from v6_drop.envs.quad import make_client

L = PB.DEFAULT_LAYOUT


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


def add_ground(cid, grey=0.45):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[60, 60, 0.01],
                              rgbaColor=[grey, grey, grey, 1], physicsClientId=cid)
    p.createMultiBody(0, -1, vis, [0, 0, -0.01], physicsClientId=cid)


def scene(renderer, psi=0.0, xy=(0.0, 0.0)):
    cid = make_client(egl=(renderer == "egl"))
    add_ground(cid)
    plat = PB.textured_platform(cid, L)
    plat.reset(np.random.default_rng(0), xy=xy, psi=psi, v=0.0)
    cam = Camera(cid, renderer=renderer)
    return cid, plat, cam


# ---------------------------------------------------------------- 1. layout
def test_layout_geometry():
    assert L.name == "L70_09" and L.big_size == 0.70 and L.small_size == 0.09
    ms = L.markers()
    ids = [m[0] for m in ms]
    assert len(ms) == 31 and len(set(ids)) == 31 and max(ids) < 50
    h = PAD_SIZE / 2
    for mid, cx, cy, s in ms:
        assert abs(cx) + s / 2 <= h - 0.05 + 1e-9 and abs(cy) + s / 2 <= h - 0.05 + 1e-9
    big = ms[0]
    assert big[0] == 0 and big[1] + big[3] / 2 <= 0.0 + 1e-9        # big marker: rear half
    for mid, cx, cy, s in ms[1:]:
        assert cx - s / 2 >= 0.0                                    # small grid: front half
    # quiet zone: gap between any two markers >= one cell of the smaller marker (6 cells/side)
    for a, b in itertools.combinations(ms, 2):
        gx = abs(a[1] - b[1]) - (a[3] + b[3]) / 2
        gy = abs(a[2] - b[2]) - (a[3] + b[3]) / 2
        assert max(gx, gy) >= min(a[3], b[3]) / 6 - 1e-9, (a, b)


# ---------------------------------------------------------------- 2. texture
def test_texture_decodes_at_right_places():
    img = PB.make_texture(L)
    assert img.shape == (PB.TEX_PX, PB.TEX_PX) and PB.TEX_PX % 4 == 0
    tex, _ = PB.ensure_assets(L)
    disk = cv2.imread(tex, cv2.IMREAD_UNCHANGED)
    assert disk.ndim == 3 and disk.shape[2] == 3 and disk.shape[1] % 4 == 0
    corners, ids, _ = PB.detector().detectMarkers(img)
    assert ids is not None and sorted(ids.ravel().tolist()) == list(range(31))
    cp = L.corners_pad()
    worst = 0.0
    for c, i in zip(corners, ids.ravel()):
        uv = c.reshape(4, 2)                              # OpenCV (col, row), pixel centres
        got = PB.texel_to_pad(uv[:, ::-1] + 0.5)          # -> edge-based (row, col) -> pad xy
        worst = max(worst, np.abs(got - cp[int(i)][:, :2]).max())
    print(f"\n[texture] all 31 ids found; worst corner position error {worst*1000:.2f} mm")
    assert worst <= 0.002


# ---------------------------------------------------------------- 3. rendering orientation
# (drone xyz relative to the pad centre, roll, pitch, yaw [deg], pad psi [deg])
POSES = {
    "behind_3m":   ((-2.5, 0.0, 3.0), 0, 0, 0, 0),
    "side_yaw":    ((-1.6, -1.0, 2.0), 0, 0, 30, 25),
    "rolled":      ((-2.0, 0.4, 2.5), 8, -4, 0, 0),
    "over_front":  ((0.0, 0.0, 0.6), 0, 0, 0, 0),
}


@pytest.mark.parametrize("renderer", RENDERERS)
@pytest.mark.parametrize("name", list(POSES))
def test_corners_match_projection(renderer, name):
    rel, roll, pitch, yaw, psi = POSES[name]
    cid, plat, cam = scene(renderer, psi=np.radians(psi), xy=(2.0, -1.0))
    try:
        R = euler_R(roll, pitch, yaw)
        pos = plat.pad_center() + np.array(rel)
        img = cam.render(R, pos)
        corners, ids, _ = PB.detector().detectMarkers(img)
        assert ids is not None, f"nothing detected ({name})"
        cp = L.corners_pad()
        worst = 0.0
        for c, i in zip(corners, ids.ravel()):
            pred, _ = cam.project(plat.pad_to_world(cp[int(i)][:, :2]), R, pos)
            worst = max(worst, np.abs(c.reshape(4, 2) - pred).max())
        print(f"\n[{renderer} {name}] ids {sorted(ids.ravel().tolist())}  worst corner error {worst:.2f} px")
        # 2.5 px: from 3 m, the 0.12 m markers are only ~9 px wide and their corners
        # are ~2 px off under TinyRenderer (measured 2.05 px, 'rolled'); big markers ~1 px.
        assert worst <= 2.5       # corner order is implied: corner k is compared with corner k
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- 4. pose
@pytest.mark.parametrize("name,rel,yaw,psi,tol_cm,tol_deg", [
    ("near_front", (0.0, 0.05, 0.45), 0, 0, 1.0, 2.0),
    ("far_3m", (-2.5, 0.5, 2.5), 10, 15, 10.0, 3.0),
])
def test_measure_pose(name, rel, yaw, psi, tol_cm, tol_deg):
    cid, plat, cam = scene("tiny", psi=np.radians(psi))
    try:
        R = euler_R(0, 0, yaw)
        pos = plat.pad_center() + np.array(rel)
        m = PB.measure(cam.render(R, pos), cam, L)
        assert m.ok
        truth_p = R.T @ (plat.pad_center() - pos)
        R_world_pad = euler_R(0, 0, psi)
        truth_R = R.T @ R_world_pad
        err_cm = np.linalg.norm(m.p_body - truth_p) * 100
        ang = np.degrees(np.arccos(np.clip((np.trace(m.R_body_pad.T @ truth_R) - 1) / 2, -1, 1)))
        print(f"\n[pose {name}] ids {m.ids}  position error {err_cm:.2f} cm  "
              f"rotation error {ang:.2f} deg  reprojection {m.reproj_px:.2f} px")
        assert err_cm < tol_cm and ang < tol_deg
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- 5. drop moment
@pytest.mark.parametrize("renderer", RENDERERS)
def test_drop_moment_always_measured(renderer):
    cid, plat, cam = scene(renderer)
    try:
        worst, n = 0.0, 0
        for pitch in (-10, 0, 10):
            for dx in (-0.15, 0.0, 0.15):
                for dy in (-0.15, 0.0, 0.15):
                    R = euler_R(0, pitch, 0)
                    pos = np.array([dx, dy, PAD_TOP_Z + P.REST_HEIGHT + 0.05])
                    m = PB.measure(cam.render(R, pos), cam, L)
                    assert m.ok and any(i in L.small_ids() for i in m.ids), (pitch, dx, dy)
                    err = np.linalg.norm(m.p_body - R.T @ (plat.pad_center() - pos))
                    worst, n = max(worst, err), n + 1
        print(f"\n[{renderer} drop] {n}/27 measured, worst pad-centre error {worst*100:.2f} cm")
        assert worst < 0.01
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- 6. collision unchanged
def test_platform_collision_is_plain_box():
    cid = make_client()
    try:
        n0 = p.getNumBodies(physicsClientId=cid)
        plat = PB.textured_platform(cid, L)
        assert p.getNumBodies(physicsClientId=cid) == n0 + 1
        data = p.getCollisionShapeData(plat.body, -1, physicsClientId=cid)
        assert len(data) == 1 and data[0][2] == p.GEOM_BOX
        assert np.allclose(data[0][3], [PAD_SIZE, PAD_SIZE, PAD_TOP_Z])
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------- 7. tiny-marker fallback (C0)
def test_tiny_markers_only_as_fallback():
    """From 3 m behind/above, the big marker is wide and the 0.09 m markers are a few px.
    Only markers >= MIN_SIDE_PX may be used; the big one must be among them."""
    cid, plat, cam = scene("tiny")
    try:
        R = euler_R(0, 0, 0)
        pos = plat.pad_center() + np.array([-2.5, 0.0, 3.0])
        img = cam.render(R, pos)
        corners, ids, _ = PB.detector().detectMarkers(img)
        sides = {int(i): np.linalg.norm(c.reshape(4, 2) - np.roll(c.reshape(4, 2), 1, 0), axis=1).mean()
                 for c, i in zip(corners, ids.ravel())}
        m = PB.measure(img, cam, L)
        print(f"\n[fallback] detected sides (px) {dict(sorted((k, round(v, 1)) for k, v in sides.items()))}"
              f"  used {m.ids}")
        assert L.big_id in m.ids
        assert all(sides[i] >= PB.MIN_SIDE_PX for i in m.ids)
    finally:
        p.disconnect(cid)
