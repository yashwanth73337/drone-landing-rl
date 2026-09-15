"""
Resolution feasibility test, run WITHOUT PyBullet.

Question: at the apparent marker sizes this camera geometry produces, can
cv2.aruco actually decode the marker?

For a nadir camera with horizontal FOV f at height h above a marker of side
s, the marker spans

    px = s / (2 * h * tan(f/2)) * W

pixels in a W-wide image. With f = 90 deg this is simply px = s/(2h) * W.
Dividing by 6 gives pixels per ArUco cell, which is what the detector's
bit-sampling actually cares about.

This synthesises exactly that image -- marker pasted into a white frame at
the computed pixel size -- and runs the real detector on it. It cannot tell
us about lighting, perspective or rasterisation artefacts (that needs the
renderer), but it puts a hard floor under the resolution choice: if the
detector fails here, on a clean noise-free nadir image, it cannot possibly
succeed in simulation.
"""

import numpy as np
import cv2

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
MARKER_ID = 0


def marker_px(marker_size, height, fov_deg, width_px):
    half = np.tan(np.radians(fov_deg) / 2.0)
    return marker_size / (2.0 * height * half) * width_px


def synth_and_detect(frame_w, frame_h, marker_px_size, offset_frac=0.0):
    """Paste a marker of the given pixel size into a white frame, detect it."""
    m = int(round(marker_px_size))
    if m < 2:
        return False, 0
    if m > min(frame_w, frame_h):
        return False, m  # overflows the frame: nothing to detect

    img = np.full((frame_h, frame_w), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(DICT, MARKER_ID, m, 1)

    cx = int(frame_w / 2 + offset_frac * frame_w / 2)
    cy = int(frame_h / 2)
    x0 = int(cx - m / 2)
    y0 = int(cy - m / 2)
    if x0 < 0 or y0 < 0 or x0 + m > frame_w or y0 + m > frame_h:
        return False, m

    img[y0:y0 + m, x0:x0 + m] = marker
    _, ids, _ = DETECTOR.detectMarkers(img)
    ok = ids is not None and MARKER_ID in ids.flatten()
    return bool(ok), m


def main():
    marker_size = 0.25
    fov = 90.0
    heights = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20,
               0.175, 0.15, 0.135, 0.125]
    resolutions = [64, 96, 128, 160, 192, 256]

    print("=" * 78)
    print("  ARUCO RESOLUTION FEASIBILITY  (synthetic, no renderer)")
    print("=" * 78)
    print(f"  marker {marker_size} m, nadir camera, {fov:.0f} deg FOV, "
          f"square sensor")
    print(f"  geometric dropout height = "
          f"{marker_size / 2 / np.tan(np.radians(fov) / 2):.4f} m")
    print()

    header = "  h(m)  " + "".join(f"{r:>12}" for r in resolutions)
    print(header)
    print("  " + "-" * (len(header) - 2))

    first_fail = {r: None for r in resolutions}
    for h in heights:
        cells = []
        for r in resolutions:
            px = marker_px(marker_size, h, fov, r)
            ok, m = synth_and_detect(r, r, px)
            cells.append(f"{'OK ' if ok else '-- '}{px / 6:5.1f}p/c")
            if not ok and first_fail[r] is None and px <= r:
                first_fail[r] = h
        print(f"  {h:5.3f} " + "".join(f"{c:>12}" for c in cells))

    print()
    print("  cell = pixels per ArUco cell (marker spans 6 cells).")
    print("  'OK' = cv2.aruco decoded it. '--' = it did not.")
    print()
    print("  Highest height at which detection FAILS while the marker is")
    print("  still geometrically inside the frame:")
    for r in resolutions:
        f = first_fail[r]
        print(f"    {r:4d} px : "
              + ("never (resolution-limited nowhere)" if f is None
                 else f"{f:.3f} m"))

    # Worst-case lateral placement: marker pushed to the frame edge, where
    # the detector has the least surrounding quiet zone.
    print()
    print("  Off-centre check at h = 0.50 m (worst spawn offset present in")
    print("  the curriculum maps to roughly 0.6 of the half-frame):")
    for r in resolutions:
        px = marker_px(marker_size, 0.50, fov, r)
        ok_c, _ = synth_and_detect(r, r, px, offset_frac=0.0)
        ok_o, _ = synth_and_detect(r, r, px, offset_frac=0.55)
        print(f"    {r:4d} px : centred {'OK' if ok_c else '--'}   "
              f"offset {'OK' if ok_o else '--'}")


if __name__ == "__main__":
    main()
