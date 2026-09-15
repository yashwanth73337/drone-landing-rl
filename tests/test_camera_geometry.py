"""
Camera-geometry unit test, run WITHOUT a live PyBullet simulation.

VisionLanderAviary's visibility metrics are pure projective geometry: they
depend only on the drone pose, the marker corners and the intrinsics. This
test stubs out PyBullet's two quaternion helpers (which are just arithmetic)
and drives the geometry methods directly, so the projection code can be
checked against closed form without needing the simulator to be installed.

What it checks:

  1. Marker centre lands where the closed form says it should in the image.
  2. The measured full-visibility dropout height equals
         h_drop = (s/2 + d_xy) / tan(FOV/2)
     across marker sizes, FOVs and offsets.
  3. Body tilt is correctly attributed as the cause of a loss of sight.
  4. Frame overflow at close range is attributed to proximity, not lateral
     error -- the distinction Strand A's blind-fraction number never made.
  5. The marker's black-cell layout matches the ArUco bit grid.

This does NOT replace the in-simulator physics-equivalence gate. It verifies
the maths, not the integration.
"""

import sys
import types
import numpy as np

# ---------------------------------------------------------------------------
# Minimal PyBullet stub: only the pure-arithmetic helpers the geometry uses.
# ---------------------------------------------------------------------------
_pb = types.ModuleType("pybullet")
_pb.GEOM_BOX = 3


def _quat_to_mat(q):
    x, y, z, w = q
    return [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ]


def _euler_to_quat(rpy):
    r, p_, y = [v / 2.0 for v in rpy]
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p_),
                              np.sin(p_), np.cos(y), np.sin(y))
    return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]


_pb.getMatrixFromQuaternion = _quat_to_mat
_pb.getQuaternionFromEuler = _euler_to_quat
sys.modules.setdefault("pybullet", _pb)

# gym_pybullet_drones imports pybullet_data at module scope purely to locate
# asset files; nothing in this test touches an asset.
_pbd = types.ModuleType("pybullet_data")
_pbd.getDataPath = lambda: "/tmp"
sys.modules.setdefault("pybullet_data", _pbd)

sys.path.insert(0, "src")
from envs.VisionLanderAviary import VisionLanderAviary   # noqa: E402


class GeometryProbe(VisionLanderAviary):
    """VisionLanderAviary's geometry methods, driven by hand-set poses.

    __init__ is bypassed entirely -- we set exactly the attributes the
    projection code reads, and nothing else. No simulator, no bodies.
    """

    def __init__(self, marker_size=0.25, fov=90.0, res=(128, 128)):
        self.CAM_RES = np.asarray(res, dtype=int)
        self.CAM_FOV_DEG = float(fov)
        self.CAM_OFFSET_BODY = np.zeros(3)
        self.CAM_NEAR = 0.003
        self.MARKER_SIZE = float(marker_size)
        self.MARKER_LIFT = 0.0
        self.MARKER_PLANE_OFFSET = 0.0009
        w, h = int(res[0]), int(res[1])
        half = np.tan(np.radians(fov) / 2.0)
        self.CAM_FX = (w / 2.0) / half
        self.CAM_FY = (h / 2.0) / (half * (h / w))
        self.CAM_CX = w / 2.0
        self.CAM_CY = h / 2.0

        self._pad = np.array([0.0, 0.0, 0.5])
        self._pos = np.array([0.0, 0.0, 1.0])
        self._rpy = np.array([0.0, 0.0, 0.0])

    # --- the two hooks the geometry code calls into ---------------------
    def _getDroneStateVector(self, i):
        s = np.zeros(20)
        s[0:3] = self._pos
        s[3:7] = _euler_to_quat(list(self._rpy))
        s[7:10] = self._rpy
        return s

    def _getLandingTargetPos(self):
        return self._pad.copy()

    # --- test driving ---------------------------------------------------
    def place(self, height_above_pad, offset=0.0, roll=0.0, pitch=0.0,
              offset_axis=0):
        self._pos = self._pad.copy()
        self._pos[2] += height_above_pad
        self._pos[offset_axis] += offset
        self._rpy = np.array([roll, pitch, 0.0])


def measure_dropout(probe, offset=0.0, hi=2.0, lo=0.0, iters=60):
    def visible(h):
        probe.place(h, offset=offset)
        return probe._geometricCameraInfo()["cam_fully_in_frame"]
    if not visible(hi):
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (hi + lo)
        if visible(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (hi + lo)


def main():
    ok_all = True

    def check(label, got, want, tol):
        nonlocal ok_all
        good = abs(got - want) <= tol
        ok_all &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {label:<46} "
              f"got {got: .6f}  want {want: .6f}")

    print("=" * 78)
    print("  CAMERA GEOMETRY UNIT TEST  (stubbed PyBullet, no simulator)")
    print("=" * 78)

    # ---- 1. image position of the marker centre ------------------------
    print("\n1. Marker centre position in normalised image coords")
    print("   closed form: n = offset / (h * tan(FOV/2))")
    probe = GeometryProbe(marker_size=0.25, fov=90.0)
    for h, off in ((0.50, 0.0), (0.50, 0.10), (0.50, 0.30), (0.25, 0.10)):
        probe.place(h, offset=off)
        info = probe._geometricCameraInfo()
        # Datum: the marker's black surface sits MARKER_PLANE_OFFSET above
        # the pad top, and h is measured to the pad top.
        h_eff = h - probe.MARKER_PLANE_OFFSET
        want = off / (h_eff * np.tan(np.radians(90.0) / 2.0))
        got = abs(info["cam_v"]) if abs(info["cam_v"]) > abs(info["cam_u"]) \
            else abs(info["cam_u"])
        check(f"h={h:.2f} offset={off:.2f}", got, want, 1e-9)

    # ---- 2. dropout height vs closed form ------------------------------
    print("\n2. Full-visibility dropout height")
    print("   closed form: h_drop = (s/2 + offset) / tan(FOV/2)")
    for marker in (0.15, 0.20, 0.25, 0.30, 0.40):
        for fov in (60.0, 90.0, 120.0):
            pr = GeometryProbe(marker_size=marker, fov=fov)
            meas = measure_dropout(pr, offset=0.0)
            pred = pr.predictedDropoutHeight(0.0)
            check(f"marker={marker:.2f}m fov={fov:.0f}deg", meas, pred, 1e-4)

    print("\n   with lateral offset:")
    pr = GeometryProbe(marker_size=0.25, fov=90.0)
    for off in (0.05, 0.11, 0.30):
        meas = measure_dropout(pr, offset=off)
        pred = pr.predictedDropoutHeight(off)
        check(f"marker=0.25m fov=90deg offset={off:.2f}m", meas, pred, 1e-4)

    # ---- 3. Strand A configuration reproduced --------------------------
    print("\n3. Strand A's reported 0.15-0.20 m dropout, from geometry alone")
    pr = GeometryProbe(marker_size=0.30, fov=80.0)
    meas = measure_dropout(pr, offset=0.0)
    print(f"   marker 0.30 m, FOV 80 deg -> h_drop = {meas:.4f} m")
    inside = 0.15 <= meas <= 0.20
    ok_all &= inside
    print(f"   [{'PASS' if inside else 'FAIL'}] falls inside the measured "
          f"0.15-0.20 m band")

    # ---- 4. loss-cause attribution -------------------------------------
    print("\n4. Loss-cause attribution")
    pr = GeometryProbe(marker_size=0.25, fov=90.0)

    pr.place(0.50, offset=0.0)
    c = pr._geometricCameraInfo()["cam_loss_cause"]
    print(f"   [{'PASS' if c is None else 'FAIL'}] "
          f"high + centred            -> {c}")
    ok_all &= (c is None)

    pr.place(0.08, offset=0.0)
    c = pr._geometricCameraInfo()["cam_loss_cause"]
    print(f"   [{'PASS' if c == 'too_close' else 'FAIL'}] "
          f"below dropout, centred    -> {c}")
    ok_all &= (c == "too_close")

    pr.place(0.50, offset=0.80)
    c = pr._geometricCameraInfo()["cam_loss_cause"]
    print(f"   [{'PASS' if c == 'lateral' else 'FAIL'}] "
          f"far off to one side       -> {c}")
    ok_all &= (c == "lateral")

    # Tilt attribution. Rather than guess a pose, search for one that is
    # fully visible when level and not visible when pitched -- that is the
    # definition of the 'tilt' cause, so the search IS the specification.
    # tilt_limit is 1.0 rad in train_lander.py's ENV_KWARGS, so pitches well
    # below that are reachable attitudes, not contrived ones.
    found = None
    for off in np.arange(0.0, 0.45, 0.01):
        for pitch in np.arange(0.05, 0.9, 0.01):
            for sign in (+1.0, -1.0):
                pr.place(0.50, offset=off)
                if not pr._geometricCameraInfo()["cam_fully_in_frame"]:
                    continue
                pr.place(0.50, offset=off, pitch=sign * pitch)
                info = pr._geometricCameraInfo()
                if not info["cam_fully_in_frame"]:
                    found = (off, sign * pitch, info["cam_loss_cause"])
                    break
            if found:
                break
        if found:
            break

    if found is None:
        print("   [FAIL] no tilt-induced loss found in the search range")
        ok_all = False
    else:
        off, pitch, c = found
        good = (c == "tilt")
        ok_all &= good
        print(f"   [{'PASS' if good else 'FAIL'}] "
              f"offset={off:.2f} pitch={pitch:+.2f} rad "
              f"(level: visible)  -> {c}")

    # ---- 5. marker bit layout ------------------------------------------
    print("\n5. Marker construction")
    import cv2
    pr = GeometryProbe()
    pr.MARKER_ID = 0
    pr.MARKER_DICT_ID = cv2.aruco.DICT_4X4_50
    pr._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    grid = pr._markerCellGrid()
    n_black = int((grid == 0).sum())
    border_black = bool(
        (grid[0, :] == 0).all() and (grid[-1, :] == 0).all()
        and (grid[:, 0] == 0).all() and (grid[:, -1] == 0).all()
    )
    print(f"   grid shape {grid.shape}, {n_black} black cells "
          f"-> {n_black + 1} visual boxes (incl. white backing)")
    print(f"   [{'PASS' if grid.shape == (6, 6) else 'FAIL'}] "
          f"6x6 grid (4x4 payload + 1-cell border)")
    print(f"   [{'PASS' if border_black else 'FAIL'}] "
          f"outer border is fully black")
    ok_all &= (grid.shape == (6, 6)) and border_black

    corners = pr._markerCornersWorld()
    span_x = float(corners[:, 0].max() - corners[:, 0].min())
    span_y = float(corners[:, 1].max() - corners[:, 1].min())
    check("corner span x == marker_size", span_x, pr.MARKER_SIZE, 1e-12)
    check("corner span y == marker_size", span_y, pr.MARKER_SIZE, 1e-12)

    print()
    print("=" * 78)
    print(f"  GEOMETRY UNIT TEST: {'PASS' if ok_all else 'FAIL'}")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
