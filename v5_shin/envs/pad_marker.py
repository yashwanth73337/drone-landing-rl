"""ArUco pad texture + textured quad asset (V5 Block 3, SPEC §4).

DICT_4X4_50, id 0. The black marker square is 1.2 m, centred on the 1.5 m pad,
with a 0.15 m white margin. ArUco needs a quiet zone to be detectable, and
the EKF+RL baseline (H) detects it. A marker filling the pad edge-to-edge
would have none, so this refines the earlier SPEC wording.
Both files are generated deterministically on first use and are tracked in git.
"""
import os

import cv2
import numpy as np

from .platform import PAD_SIZE

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
MARKER_SIZE = 1.2          # m, outer black square
DICT_ID = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
TEX_PX = 768               # ~2 mm/px. Width must be a multiple of 4 and the image RGB:
                           # a 750 px grayscale texture renders sheared under EGL (GL row alignment)
TEX_PATH = os.path.join(ASSETS, "aruco_pad.png")
OBJ_PATH = os.path.join(ASSETS, "pad_box.obj")

# Marker corners in the PAD frame (x forward, y left), ArUco order
# TL, TR, BR, BL of the marker image. The image top is at pad +x, and the image
# left is at pad +y (verified by tests/test_block3_camera.py).
_h = MARKER_SIZE / 2
MARKER_CORNERS_PAD = np.array([[_h, _h], [_h, -_h], [-_h, -_h], [-_h, _h]])


def ensure_assets():
    if not os.path.exists(TEX_PATH):
        d = cv2.aruco.getPredefinedDictionary(DICT_ID)
        m_px = int(round(TEX_PX * MARKER_SIZE / PAD_SIZE))       # 614 px
        marker = cv2.aruco.generateImageMarker(d, MARKER_ID, m_px, borderBits=1)
        img = np.full((TEX_PX, TEX_PX), 255, np.uint8)
        o = (TEX_PX - m_px) // 2
        img[o:o + m_px, o:o + m_px] = marker
        cv2.imwrite(TEX_PATH, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    if not os.path.exists(OBJ_PATH):
        _write_box_obj(OBJ_PATH)
    return TEX_PATH, OBJ_PATH


def _write_box_obj(path):
    """Platform visual: ONE closed box mesh (1.5 x 1.5 x PAD_TOP_Z, centred at the
    body origin), with the marker texture on the top face only. The side and
    bottom faces sample a white margin texel. A separate marker quad above the
    box z-fights under EGL's coarse depth buffer (found in Block 3: stripes and
    lost detection beyond 3 m altitude), so there must be no coplanar surfaces."""
    from .platform import PAD_TOP_Z
    h, t = PAD_SIZE / 2, PAD_TOP_Z / 2
    W = (0.01, 0.01)                                  # white margin texel
    faces = [  # (4 vertices CCW seen from outside, 4 uvs)
        ([(-h, h, t), (-h, -h, t), (h, -h, t), (h, h, t)],       # top (+z): marker
         [(0, 0), (1, 0), (1, 1), (0, 1)]),                      # image top -> +x, left -> +y
        ([(-h, -h, -t), (-h, h, -t), (h, h, -t), (h, -h, -t)], [W] * 4),   # bottom
        ([(h, -h, -t), (h, h, -t), (h, h, t), (h, -h, t)], [W] * 4),       # +x
        ([(-h, h, -t), (-h, -h, -t), (-h, -h, t), (-h, h, t)], [W] * 4),   # -x
        ([(h, h, -t), (-h, h, -t), (-h, h, t), (h, h, t)], [W] * 4),       # +y
        ([(-h, -h, -t), (h, -h, -t), (h, -h, t), (-h, -h, t)], [W] * 4),   # -y
    ]
    with open(path, "w") as f:
        f.write("# V5 platform box, marker on the top face\n")
        for i, (vs, uvs) in enumerate(faces):
            for v in vs:
                f.write("v %g %g %g\n" % v)
            for uv in uvs:
                f.write("vt %g %g\n" % uv)
            e1 = np.subtract(vs[1], vs[0]); e2 = np.subtract(vs[2], vs[0])
            n = np.cross(e1, e2); n = n / np.linalg.norm(n)
            f.write("vn %g %g %g\n" % tuple(n))
            b, ni = 4 * i, i + 1
            f.write(f"f {b+1}/{b+1}/{ni} {b+2}/{b+2}/{ni} {b+3}/{b+3}/{ni}\n")
            f.write(f"f {b+1}/{b+1}/{ni} {b+3}/{b+3}/{ni} {b+4}/{b+4}/{ni}\n")


def detector():
    """Subpixel corner refinement on (OpenCV's default is none). Used by the tests
    and, later, by the EKF+RL baseline's detector."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT_ID), params)
