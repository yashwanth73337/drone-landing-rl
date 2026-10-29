"""v6_drop B4: the multi-marker landing pad (ArUco board) and pose estimation.

Layout (decided 28 Oct): one BIG marker in the REAR half of the pad for far range,
and a GRID of SMALL markers in the FRONT half for close range. Reason: the 60 deg
camera cannot see the pad centre or the rear half while the drone is above the pad
(B3 test), and the strip it does see slides with small timing errors, so a single
row of small markers is missed often; a grid always has one marker fully in view.

Pad frame: origin at the centre of the pad top, x forward (travel direction),
y left, z up. Texture image: top row = pad +x edge, left column = pad +y edge
(same convention as V5, checked by the rendering test).

Rules kept from V5 (Block 3):
  * the marker is TEXTURE ONLY on the top face of ONE closed box mesh (a separate
    quad above the box z-fights under EGL: stripes, lost detection);
  * textures are RGB with width % 4 == 0 (EGL row alignment);
  * the box's side and bottom faces sample a white texel.

Pose estimation: detect all markers, collect the corners of every detected marker
whose id belongs to the layout, and run ONE solvePnP (IPPE, planar) over all of them.
"""
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pybullet as p

from .camera import OFFSET_B
from .platform_s1 import PAD_SIZE, PAD_TOP_Z, PlatformS1

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
DICT_ID = cv2.aruco.DICT_4X4_50
TEX_PX = 1536                  # 1.5 m / 1536 px ~ 1 mm per texel; multiple of 4
MIN_SIDE_PX = 10.0             # px; smaller markers are used only if nothing bigger is seen (C0)


@dataclass(frozen=True)
class Layout:
    """All sizes in metres. Marker size = outer black square (ArUco border included)."""
    name: str
    big_size: float
    big_center: tuple                       # pad-frame (x, y)
    small_size: float
    small_xs: tuple                         # pad-frame x of the small-marker rows
    small_ys: tuple                         # pad-frame y of the small-marker columns
    big_id: int = 0

    def markers(self):
        """[(id, cx, cy, size)]: big first (id 0), then small ids 1.. row by row."""
        out = [(self.big_id, self.big_center[0], self.big_center[1], self.big_size)]
        k = 1
        for x in self.small_xs:
            for y in self.small_ys:
                out.append((k, x, y, self.small_size))
                k += 1
        return out

    def corners_pad(self):
        """{id: (4, 3) pad-frame corners on z = 0} in ArUco order TL, TR, BR, BL of the
        marker image (image top = pad +x, image left = pad +y)."""
        out = {}
        for mid, cx, cy, s in self.markers():
            h = s / 2
            out[mid] = np.array([[cx + h, cy + h, 0.0], [cx + h, cy - h, 0.0],
                                 [cx - h, cy - h, 0.0], [cx - h, cy + h, 0.0]])
        return out

    def small_ids(self):
        return [m[0] for m in self.markers()[1:]]


# Candidate layouts for the detection-range study (B4, 28 Oct). Results (Claude's
# sandbox, EGL, seed 7000): drop moment 100% detected for all three (error ~1-2 mm);
# chase up to 8.9 m: L70_12 99.0%, L60_12 97.1%, L70_10 98.5%; approach: 91 / 91 / 84%.
# Chosen by Yashwanth on 28 Oct: L70_12 (best long range; 0.10 m small markers worse).
LAYOUTS = {
    "L70_12": Layout("L70_12", 0.70, (-0.35, 0.0), 0.12, (0.16, 0.32, 0.48, 0.64), (-0.25, 0.0, 0.25)),
    "L60_12": Layout("L60_12", 0.60, (-0.35, 0.0), 0.12, (0.16, 0.32, 0.48, 0.64), (-0.25, 0.0, 0.25)),
    "L70_10": Layout("L70_10", 0.70, (-0.35, 0.0), 0.10, (0.15, 0.29, 0.43, 0.57), (-0.22, 0.0, 0.22)),
    # C0 (28 Oct, legged drone: camera 0.17-0.25 m above the pad at the drop instead of
    # 0.27-0.35 m): smaller, denser grids so one marker always fits in the shorter strip.
    "L70_09": Layout("L70_09", 0.70, (-0.35, 0.0), 0.09,
                     (0.065, 0.175, 0.285, 0.395, 0.505, 0.615), (-0.22, -0.11, 0.0, 0.11, 0.22)),
    "L70_08": Layout("L70_08", 0.70, (-0.35, 0.0), 0.08,
                     (0.06, 0.16, 0.26, 0.36, 0.46, 0.56, 0.66), (-0.25, -0.15, -0.05, 0.05, 0.15, 0.25)),
}
DEFAULT_LAYOUT = LAYOUTS["L70_09"]


# ---------------------------------------------------------------- texture + mesh
def texture_path(layout):
    return os.path.join(ASSETS, f"pad_{layout.name}.png")


def make_texture(layout, path=None):
    """Draw the whole pad top (white) with every marker. Returns the grey image."""
    d = cv2.aruco.getPredefinedDictionary(DICT_ID)
    img = np.full((TEX_PX, TEX_PX), 255, np.uint8)
    mpp = PAD_SIZE / TEX_PX                                   # metres per texel
    h = PAD_SIZE / 2
    for mid, cx, cy, s in layout.markers():
        n = int(round(s / mpp))
        mk = cv2.aruco.generateImageMarker(d, mid, n, borderBits=1)
        r0 = int(round((h - (cx + s / 2)) / mpp))             # image row of the +x edge
        c0 = int(round((h - (cy + s / 2)) / mpp))             # image col of the +y edge
        img[r0:r0 + n, c0:c0 + n] = mk
    if path is not None:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return img


def texel_to_pad(rc):
    """(N, 2) texture (row, col) pixel positions -> (N, 2) pad-frame (x, y)."""
    mpp = PAD_SIZE / TEX_PX
    rc = np.atleast_2d(rc).astype(float)
    return np.stack([PAD_SIZE / 2 - rc[:, 0] * mpp, PAD_SIZE / 2 - rc[:, 1] * mpp], 1)


def box_obj_path():
    return os.path.join(ASSETS, "pad_box_v6.obj")


def write_box_obj(path):
    """ONE closed box mesh (1.5 x 1.5 x PAD_TOP_Z, centred at the body origin) with
    the texture on the top face only; other faces sample a white texel (V5 rule)."""
    hh, t = PAD_SIZE / 2, PAD_TOP_Z / 2
    W = (0.001, 0.999)                                   # a texel in the white top-left margin
    faces = [
        ([(-hh, hh, t), (-hh, -hh, t), (hh, -hh, t), (hh, hh, t)],    # top: image top -> +x,
         [(0, 0), (1, 0), (1, 1), (0, 1)]),                         #      image left -> +y
        ([(-hh, -hh, -t), (-hh, hh, -t), (hh, hh, -t), (hh, -hh, -t)], [W] * 4),
        ([(hh, -hh, -t), (hh, hh, -t), (hh, hh, t), (hh, -hh, t)], [W] * 4),
        ([(-hh, hh, -t), (-hh, -hh, -t), (-hh, -hh, t), (-hh, hh, t)], [W] * 4),
        ([(hh, hh, -t), (-hh, hh, -t), (-hh, hh, t), (hh, hh, t)], [W] * 4),
        ([(-hh, -hh, -t), (hh, -hh, -t), (hh, -hh, t), (-hh, -hh, t)], [W] * 4),
    ]
    with open(path, "w") as f:
        f.write("# v6 platform box, pad texture on the top face\n")
        for i, (vs, uvs) in enumerate(faces):
            for v in vs:
                f.write("v %g %g %g\n" % v)
            for uv in uvs:
                f.write("vt %g %g\n" % uv)
            e1 = np.subtract(vs[1], vs[0]); e2 = np.subtract(vs[2], vs[0])
            nrm = np.cross(e1, e2); nrm = nrm / np.linalg.norm(nrm)
            f.write("vn %g %g %g\n" % tuple(nrm))
            b, ni = 4 * i, i + 1
            f.write(f"f {b+1}/{b+1}/{ni} {b+2}/{b+2}/{ni} {b+3}/{b+3}/{ni}\n")
            f.write(f"f {b+1}/{b+1}/{ni} {b+3}/{b+3}/{ni} {b+4}/{b+4}/{ni}\n")


def ensure_assets(layout):
    os.makedirs(ASSETS, exist_ok=True)
    tex = texture_path(layout)
    if not os.path.exists(tex):
        make_texture(layout, tex)
    obj = box_obj_path()
    if not os.path.exists(obj):
        write_box_obj(obj)
    return tex, obj


def textured_platform(client, layout=DEFAULT_LAYOUT):
    """A PlatformS1 whose top face shows the board."""
    tex, obj = ensure_assets(layout)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=obj, physicsClientId=client)
    plat = PlatformS1(client, visual_shape=vis)
    tid = p.loadTexture(tex, physicsClientId=client)
    p.changeVisualShape(plat.body, -1, textureUniqueId=tid, rgbaColor=[1, 1, 1, 1],
                        physicsClientId=client)
    return plat


# ---------------------------------------------------------------- detection + pose
def detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT_ID), params)


def cam_to_body(cam):
    """Rotation from OpenCV camera axes (x right, y down, z forward) to the body frame."""
    right = np.cross(cam.axis_b, cam.up_b)
    return np.column_stack([right, -cam.up_b, cam.axis_b])


@dataclass
class PadMeasurement:
    ids: list = field(default_factory=list)      # layout ids used for the pose
    ok: bool = False                             # pose computed
    confident: bool = False                      # >= 1 used marker is >= MIN_SIDE_PX wide (Step 5)
    p_body: np.ndarray = None                    # pad centre (top) in the BODY frame, m
    R_body_pad: np.ndarray = None                # pad frame -> body frame rotation
    reproj_px: float = np.nan                    # mean corner reprojection error


def measure(img, cam, layout=DEFAULT_LAYOUT, det=None):
    """Detect markers in a grey image and estimate the pad pose relative to the body.
    Uses only the image and the known camera + layout (what a real drone has)."""
    det = det or detector()
    corners, ids, _ = det.detectMarkers(img)
    out = PadMeasurement()
    if ids is None:
        return out
    cp = layout.corners_pad()
    found = []                                            # (id, 4 x 2 corners, mean side px)
    for c, i in zip(corners, ids.ravel()):
        if int(i) in cp:
            c = c.reshape(4, 2)
            side = np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1).mean()
            found.append((int(i), c, side))
    if not found:
        return out
    # C0: markers only a few pixels wide have noisy corners and spoil the pose of the
    # good ones. Use only markers >= MIN_SIDE_PX wide; if none is that big, use them all
    # (a rough pose is better than none).
    good = [f for f in found if f[2] >= MIN_SIDE_PX]
    out.confident = bool(good)
    good = good or found
    out.ids = [f[0] for f in good]
    obj = [cp[f[0]] for f in good]
    imgp = [f[1] for f in good]
    obj = np.concatenate(obj).astype(np.float64)
    imgp = np.concatenate(imgp).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, cam.K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return out
    proj, _ = cv2.projectPoints(obj, rvec, tvec, cam.K, None)
    out.reproj_px = float(np.linalg.norm(proj.reshape(-1, 2) - imgp, axis=1).mean())
    Rcb = cam_to_body(cam)
    R_cam_pad = cv2.Rodrigues(rvec)[0]
    out.p_body = OFFSET_B + Rcb @ tvec.ravel()
    out.R_body_pad = Rcb @ R_cam_pad
    out.ok = True
    return out
