"""
VisionLanderAviary
==================

Stage V1 of the vision-based landing work.

This subclasses LanderAIAviary and adds a DOWNWARD-FACING CAMERA plus a
VISUAL-ONLY ArUco marker on the landing platform. It changes NOTHING about
the physics, the observation, the action, the reward, or the success logic.

    LanderAIAviary        (untouched, still the trusted baseline)
        |
        +-- VisionLanderAviary   (this file: adds sensing, changes no dynamics)

WHAT THIS FILE IS FOR
------------------------------------------------------------------------------
V1 answers one question with measurements rather than assumption:

    While the EXISTING privileged-state policy flies the EXISTING task,
    how much of the trajectory would a real downward camera actually be
    able to see, and where exactly does it lose the marker?

The learned visual encoder, the LSTM relative-state estimator, the asymmetric
privileged critic and the active-perception reward from Shin et al. are all
OUT OF SCOPE here. None of them appear in this file.


THE CRITICAL INVARIANT
------------------------------------------------------------------------------
_computeObs() is NOT overridden. This environment returns the parent's exact
15-D privileged observation, so the trusted TD3 checkpoints
(moving_025_seed{1,2,3}_final.zip) load and behave identically.

ALL camera output is published through _computeInfo(). Nothing the camera
produces can reach the policy. This is deliberate: V1 is an observational
study of a policy that is not being changed.

Three further precautions keep the physics bit-identical:

  1. The drone URDF is NOT modified. The camera is a view/projection matrix
     computed per step from the drone's pose -- it is not a body.
  2. The ArUco marker is a SEPARATE body created with
     baseCollisionShapeIndex=-1 and baseMass=0. It has no collision geometry
     at all, so it cannot generate a contact point, and it is never passed to
     _hasValidLandingContact() (which tests DRONE_IDS[0] against PLAT_ID only).
  3. Rendering consumes no RNG draws, so the seeded episode stream is
     unchanged.

Equivalence against the parent class is verified empirically, not assumed --
see scripts/fov_diagnostic_v1.py --mode equivalence.


CAMERA MODEL AND WHY IT DIFFERS FROM THE PAPER
------------------------------------------------------------------------------
Shin et al. mount the camera at a 60 deg downward pitch FROM THE FORWARD AXIS
(i.e. 30 deg off nadir). That is not portable to this environment, and the
reason is structural rather than a matter of taste:

    Their action is a_t = [vx, vy, vz, wz] -- it contains YAW RATE, and their
    reward's single largest weight (w = 2.0, Table III) is a yaw-rate penalty.
    A tilted camera works for them because the policy can actively steer where
    the camera looks.

    LanderAIAviary's action is delta_p_t = 0.1 * c_t, three position deltas.
    DSLPIDControl is given target_rpy = 0, so yaw is pinned. A tilted camera
    would therefore create a PERMANENT blind direction fixed in the world
    frame: the pad would be systematically easier to see when it drifts one
    way than the other, and the policy would have no means of correcting it.

    That asymmetry would contaminate every FOV statistic measured here.

DELIBERATE DEVIATION 1: the camera points straight down (nadir), rigidly
attached to the drone body, so body tilt does move the footprint.

DELIBERATE DEVIATION 2: the image is SQUARE (default 128x128, 90 deg FOV on
both axes). Shin et al. use 512x320, 90 deg horizontal. A non-square image has
a narrower vertical FOV, which would make "does the pad stay in frame" depend
on WHICH DIRECTION the platform happened to spawn in -- and the spawn azimuth
is uniform on [0, 2pi) and entirely uncontrolled. A square sensor removes that
nuisance variable. The FOV value itself (90 deg) is taken from the paper.

DELIBERATE DEVIATION 3: grayscale is produced by converting the rendered RGB,
not by a native grayscale sensor. Functionally equivalent for ArUco.


THE MARKER SIZE IS A DESIGN CHOICE, NOT A MEASUREMENT
------------------------------------------------------------------------------
For a nadir camera the height at which a square marker of side s stops being
fully visible has a closed form (ignoring tilt):

    h_drop = (s/2 + d_xy) / tan(FOV/2)

With FOV = 90 deg this is simply h_drop = s/2 + d_xy.

So the "marker leaves the frame at height h" result is, to first order, pure
geometry -- it is a consequence of the marker size WE choose, not a property
of ArUco, of the policy, or of the physics. Strand A's measured 0.15-0.20 m
dropout is reproduced by putting its own configuration into this formula.

marker_size is therefore an explicit, documented experimental parameter. Any
later claim of the form "our method survives visual dropout" has to state the
marker size it was measured at, or it is not a claim about anything.

Defaults here: pad 0.5 m (inherited, unchanged), marker 0.25 m, FOV 90 deg,
giving h_drop ~ 0.125 m for a centred approach -- about 25% of the 0.5 m
descent, with the marker comfortably inside the frame at spawn even at the
widest curriculum offset (0.30 m + 0.125 m = 0.425 m < 0.5 m half-width).


WHAT LANDS IN info[]
------------------------------------------------------------------------------
Geometric (analytic pinhole projection, always on, no rendering cost):

    cam_u, cam_v              marker centre in normalised image coords,
                              (0,0) = principal point, edges at +/-1
    cam_centre_in_frame       marker centre inside the image rectangle
    cam_corners_in_frame      how many of the 4 outer corners are in frame
    cam_fully_in_frame        all 4 corners in frame (ArUco needs this)
    cam_visible_behind        marker is behind the camera / past near plane
    cam_px_per_cell           projected marker width / 6, in pixels
    cam_loss_cause            None | 'behind' | 'too_close' | 'lateral'
                              | 'tilt' -- see _classifyLossCause()

Rendered (only when camera_mode='render'):

    cam_aruco_detected        cv2.aruco decoded the marker this step
    cam_aruco_pos_err         ||solvePnP relative position - ground truth||
    cam_image                 HxW uint8 grayscale (only if keep_image=True)
"""

import numpy as np
import pybullet as p
import cv2

from envs.LanderAIAviary import LanderAIAviary


class VisionLanderAviary(LanderAIAviary):
    """LanderAIAviary + downward camera + visual-only ArUco marker.

    Physics, observation, action, reward and success logic are inherited
    unchanged. See module docstring for the invariant this relies on.
    """

    # ------------------------------------------------------------------
    # Loss-cause labels for _classifyLossCause()
    # ------------------------------------------------------------------
    LOSS_NONE = None
    LOSS_BEHIND = 'behind'        # marker at or behind the near plane
    LOSS_TOO_CLOSE = 'too_close'  # marker overflows the frame from proximity
    LOSS_LATERAL = 'lateral'      # marker pushed out sideways by offset
    LOSS_TILT = 'tilt'            # would have been in frame if drone were level

    def __init__(self,
                 *args,
                 # --- camera geometry -------------------------------------
                 # DEVIATION 1 from Shin et al.: nadir, not 60 deg from the
                 # forward axis. Rationale in the module docstring.
                 camera_res=(128, 128),
                 # DEVIATION 2: square sensor. FOV value itself is the
                 # paper's (90 deg horizontal).
                 camera_fov_deg: float = 90.0,
                 # Camera sits on the drone's underside, in the BODY frame,
                 # so it tilts with the airframe.
                 camera_offset_body=(0.0, 0.0, -0.01),
                 camera_near: float = 0.003,
                 camera_far: float = 10.0,
                 # --- marker ----------------------------------------------
                 # EXPERIMENTAL PARAMETER, not a paper value. Sets h_drop.
                 marker_size: float = 0.25,
                 marker_id: int = 0,
                 marker_dict: int = cv2.aruco.DICT_4X4_50,
                 # Height of the visual marker plane above the pad's top
                 # surface. Purely cosmetic; the body has no collision shape.
                 marker_lift: float = 0.001,
                 # White backing square, as a multiple of marker_size, giving
                 # ArUco the quiet zone it needs regardless of pad colour.
                 marker_quiet_zone: float = 1.4,
                 enable_marker: bool = True,
                 # --- camera mode -----------------------------------------
                 #   'off'      no camera at all (parent behaviour + marker)
                 #   'geometry' analytic projection only -- cheap, exact,
                 #              no renderer, no detector
                 #   'render'   also rasterise and run cv2.aruco
                 camera_mode: str = 'geometry',
                 keep_image: bool = False,
                 **kwargs):

        if camera_mode not in ('off', 'geometry', 'render'):
            raise ValueError(
                f"camera_mode must be 'off', 'geometry' or 'render', "
                f"got {camera_mode!r}"
            )

        self.CAM_RES = np.asarray(camera_res, dtype=int)
        self.CAM_FOV_DEG = float(camera_fov_deg)
        self.CAM_OFFSET_BODY = np.asarray(camera_offset_body, dtype=float)
        self.CAM_NEAR = float(camera_near)
        self.CAM_FAR = float(camera_far)
        self.CAMERA_MODE = camera_mode
        self.KEEP_IMAGE = bool(keep_image)

        self.MARKER_SIZE = float(marker_size)
        self.MARKER_ID = int(marker_id)
        self.MARKER_DICT_ID = marker_dict
        self.MARKER_LIFT = float(marker_lift)
        self.MARKER_QUIET_ZONE = float(marker_quiet_zone)
        self.ENABLE_MARKER = bool(enable_marker)

        # Height of the BLACK CELL SURFACE above the pad's top face:
        # marker_lift, plus the backing plate, plus the cell boxes' own
        # half-thickness. Named because every height reported by this class
        # has to declare its datum. info['height_above_pad'] is measured to
        # the PAD TOP, so anything compared against it must add this.
        # Getting this wrong is a 0.9 mm systematic bias in every dropout
        # height -- small, but it is exactly the kind of silent datum error
        # that made the ArUco corner-scale bug hard to find.
        self.MARKER_PLANE_OFFSET = self.MARKER_LIFT + 0.0009

        # Body handle for the visual-only marker. Reset every episode because
        # BaseAviary calls p.resetSimulation() during _housekeeping().
        self.MARKER_BODY = None

        # Pinhole intrinsics. PyBullet's computeProjectionMatrixFOV takes the
        # VERTICAL fov; we keep the sensor square so both axes match.
        w, h = int(self.CAM_RES[0]), int(self.CAM_RES[1])
        half = np.tan(np.radians(self.CAM_FOV_DEG) / 2.0)
        self.CAM_FX = (w / 2.0) / half
        self.CAM_FY = (h / 2.0) / (half * (h / w))
        self.CAM_CX = w / 2.0
        self.CAM_CY = h / 2.0
        self.CAM_K = np.array([
            [self.CAM_FX, 0.0, self.CAM_CX],
            [0.0, self.CAM_FY, self.CAM_CY],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        # Rendering is noise-free, so zero distortion is the honest model.
        self.CAM_DIST = np.zeros(5, dtype=np.float64)

        self._aruco_dict = cv2.aruco.getPredefinedDictionary(
            self.MARKER_DICT_ID
        )
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(
            self._aruco_dict, self._aruco_params
        )

        self._last_cam_info = {}

        super().__init__(*args, **kwargs)

    # ==================================================================
    # VISUAL-ONLY MARKER
    # ==================================================================

    def _markerCellGrid(self):
        """Return the 6x6 (4x4 payload + 1-cell border) ArUco bit grid.

        0 = black cell, 1 = white cell.
        """
        cells = 4 + 2  # DICT_4X4_* payload plus a one-cell black border
        img = cv2.aruco.generateImageMarker(
            self._aruco_dict, self.MARKER_ID, cells, 1
        )
        return (img > 127).astype(int)

    def _markerRectangles(self):
        """Merge the marker's black cells into as few rectangles as possible.

        Run-length encodes each row, then merges vertically-stacked runs that
        share the same column span. For DICT_4X4_50 id 0 this takes 28 unit
        cells down to a handful of rectangles, which keeps the rendered body
        small without changing a single pixel of the pattern.

        Returns a list of (cx, cy, half_w, half_h) in the marker frame.
        """
        grid = self._markerCellGrid()
        n = grid.shape[0]
        cell = self.MARKER_SIZE / n
        half = self.MARKER_SIZE / 2.0

        # Horizontal runs of black cells, per row.
        runs = []   # (row, col_start, col_end_exclusive)
        for r in range(n):
            c = 0
            while c < n:
                if grid[r, c] == 0:
                    c0 = c
                    while c < n and grid[r, c] == 0:
                        c += 1
                    runs.append([r, r + 1, c0, c])
                else:
                    c += 1

        # Merge runs that sit directly on top of one another.
        merged = []
        for run in runs:
            placed = False
            for m in merged:
                if m[2] == run[2] and m[3] == run[3] and m[1] == run[0]:
                    m[1] = run[1]
                    placed = True
                    break
            if not placed:
                merged.append(list(run))

        # AXIS MAPPING -- this is easy to get wrong and the failure is
        # silent. The camera places world +x along the image's vertical axis
        # and world -y along its horizontal axis. To make the rendered
        # pattern match cv2.aruco's reference bitmap, the marker's ROW index
        # must run down the image (world x) and its COLUMN index across it
        # (world y). Mapping columns to x instead transposes the pattern,
        # which is a mirror -- and ArUco is invariant to the four rotations
        # but NOT to reflection, so a transposed marker renders perfectly,
        # looks completely correct to the eye, and never decodes.
        rects = []
        for r0, r1, c0, c1 in merged:
            ext_x = (r1 - r0) * cell      # rows span world x
            ext_y = (c1 - c0) * cell      # columns span world y
            cx = half - (r0 + (r1 - r0) / 2.0) * cell
            cy = half - (c0 + (c1 - c0) / 2.0) * cell
            rects.append((cx, cy, ext_x / 2.0, ext_y / 2.0))
        return rects

    def _createMarkerBody(self, position):
        """Create the visual-only marker as ONE body.

        Implementation note, found the hard way: p.createVisualShapeArray()
        SEGFAULTS above 16 child shapes. It is not an exception, it is a hard
        crash of the interpreter, and the marker needs more shapes than that
        even after merging. The body is therefore assembled as a base visual
        shape plus fixed-joint LINKS, which has no such cap.

        Every link is created with collisionShapeIndex = -1 and mass 0, so
        the body carries NO collision geometry whatsoever and cannot appear
        in getContactPoints() for any query. This is what keeps the physics
        identical to LanderAIAviary.
        """
        rects = self._markerRectangles()
        backing = self.MARKER_SIZE * self.MARKER_QUIET_ZONE / 2.0

        base_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[backing, backing, 0.0004],
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            physicsClientId=self.CLIENT,
        )

        link_visuals = []
        link_positions = []
        for cx, cy, hw, hh in rects:
            link_visuals.append(p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[hw, hh, 0.0003],
                rgbaColor=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.CLIENT,
            ))
            link_positions.append([cx, cy, 0.0006])

        n = len(link_visuals)
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,      # <-- no collision geometry
            baseVisualShapeIndex=base_visual,
            basePosition=position,
            linkMasses=[0.0] * n,
            linkCollisionShapeIndices=[-1] * n,   # <-- nor on any link
            linkVisualShapeIndices=link_visuals,
            linkPositions=link_positions,
            linkOrientations=[[0, 0, 0, 1]] * n,
            linkInertialFramePositions=[[0, 0, 0]] * n,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
            linkParentIndices=[0] * n,
            linkJointTypes=[p.JOINT_FIXED] * n,
            linkJointAxis=[[0, 0, 1]] * n,
            physicsClientId=self.CLIENT,
        )

    def _updateMarker(self):
        """Place / move the visual-only marker on the pad's top surface."""
        if not self.ENABLE_MARKER:
            return

        target = self._getLandingTargetPos()
        pos = [float(target[0]), float(target[1]),
               float(target[2]) + self.MARKER_LIFT]

        if self.MARKER_BODY is None:
            self.MARKER_BODY = self._createMarkerBody(pos)
        else:
            p.resetBasePositionAndOrientation(
                self.MARKER_BODY, pos, [0, 0, 0, 1],
                physicsClientId=self.CLIENT,
            )

    def _updatePlatform(self, step_offset=0):
        """Parent's platform update, then drag the marker along with it.

        The parent method is called unchanged; the marker is a pure rider.
        """
        super()._updatePlatform(step_offset=step_offset)
        self._updateMarker()

    def _markerCornersWorld(self):
        """The 4 outer corners of the marker's black square, in world frame.

        These are the corners ArUco actually keys on -- using the inner
        payload square instead was a real bug earlier in this project and
        inflated pose error by roughly 12x.
        """
        target = self._getLandingTargetPos()
        z = target[2] + self.MARKER_PLANE_OFFSET
        h = self.MARKER_SIZE / 2.0
        return np.array([
            [target[0] - h, target[1] + h, z],
            [target[0] + h, target[1] + h, z],
            [target[0] + h, target[1] - h, z],
            [target[0] - h, target[1] - h, z],
        ], dtype=np.float64)

    def _markerCornersObject(self):
        """Same 4 corners in the marker's own frame, for solvePnP."""
        h = self.MARKER_SIZE / 2.0
        return np.array([
            [-h, h, 0.0],
            [h, h, 0.0],
            [h, -h, 0.0],
            [-h, -h, 0.0],
        ], dtype=np.float64)

    # ==================================================================
    # CAMERA GEOMETRY
    # ==================================================================

    def _cameraPose(self, level=False):
        """Return (eye, R_wc) for the downward camera.

        R_wc maps CAMERA-frame coordinates into the world frame, using the
        standard vision convention: camera x right, y down, z forward (i.e.
        along the viewing direction).

        level=True computes the same pose with roll and pitch zeroed, which
        is what _classifyLossCause() uses to decide whether a loss of sight
        was caused by body tilt or would have happened anyway.
        """
        state = self._getDroneStateVector(0)
        pos = np.asarray(state[0:3], dtype=float)

        if level:
            yaw = float(state[9])
            quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        else:
            quat = state[3:7]

        R_body = np.array(
            p.getMatrixFromQuaternion(quat), dtype=float
        ).reshape(3, 3)

        eye = pos + R_body @ self.CAM_OFFSET_BODY

        # Camera looks along the body's -z axis (straight down when level).
        forward = R_body @ np.array([0.0, 0.0, -1.0])
        # Body +x becomes "up" in the image, so image rows run along the
        # drone's forward axis. Yaw is pinned at 0 in this environment, so
        # this is effectively world +x.
        up = R_body @ np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up)

        right /= np.linalg.norm(right)
        # Vision convention has +y pointing DOWN the image.
        down = -up / np.linalg.norm(up)
        forward = forward / np.linalg.norm(forward)

        R_wc = np.column_stack([right, down, forward])
        return eye, R_wc

    def _projectPoints(self, pts_world, level=False):
        """Project world points to pixel coords. Returns (uv, z_cam)."""
        eye, R_wc = self._cameraPose(level=level)
        pts_cam = (pts_world - eye) @ R_wc        # world -> camera
        z = pts_cam[:, 2]

        safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.CAM_FX * pts_cam[:, 0] / safe_z + self.CAM_CX
        v = self.CAM_FY * pts_cam[:, 1] / safe_z + self.CAM_CY
        return np.column_stack([u, v]), z

    def _inFrame(self, uv, z):
        """Boolean mask: point is in front of the near plane AND on sensor."""
        w, h = float(self.CAM_RES[0]), float(self.CAM_RES[1])
        return (
            (z > self.CAM_NEAR)
            & (uv[:, 0] >= 0.0) & (uv[:, 0] <= w)
            & (uv[:, 1] >= 0.0) & (uv[:, 1] <= h)
        )

    def _classifyLossCause(self, corners_world, n_in_frame, centre_z):
        """Why is the marker not fully visible?

        Strand A reported a blind fraction but never separated the causes,
        which left "the camera lost the pad" doing a lot of unexamined work.
        The four cases have completely different implications:

          behind      geometrically degenerate; drone is at/below the marker
          too_close   the marker OVERFLOWS the frame -- this is the terminal
                      approach, and it is unavoidable for a fixed marker size
          lateral     the marker is off to the side -- a tracking failure,
                      and the thing an active-perception reward could fix
          tilt        in frame if the drone were level, out of frame because
                      it is not -- an aggressive-manoeuvre failure
        """
        if n_in_frame == 4:
            return self.LOSS_NONE
        if centre_z <= self.CAM_NEAR:
            return self.LOSS_BEHIND

        # Would a perfectly level drone at the same position have seen it?
        uv_level, z_level = self._projectPoints(corners_world, level=True)
        n_level = int(np.sum(self._inFrame(uv_level, z_level)))
        if n_level == 4:
            return self.LOSS_TILT

        # Level too. Is the marker overflowing the frame (centre still well
        # inside, corners outside) or pushed off to one side?
        centre = corners_world.mean(axis=0).reshape(1, 3)
        uv_c, z_c = self._projectPoints(centre, level=True)
        if bool(self._inFrame(uv_c, z_c)[0]):
            return self.LOSS_TOO_CLOSE
        return self.LOSS_LATERAL

    def _geometricCameraInfo(self):
        """Analytic visibility metrics. No rendering, no detector."""
        corners = self._markerCornersWorld()
        centre = corners.mean(axis=0).reshape(1, 3)

        uv_c, z_c = self._projectPoints(corners)
        in_frame = self._inFrame(uv_c, z_c)
        n_in = int(np.sum(in_frame))

        uv_ctr, z_ctr = self._projectPoints(centre)
        centre_in = bool(self._inFrame(uv_ctr, z_ctr)[0])

        # Normalised image coords: 0 at the principal point, +/-1 at the edge.
        nx = float((uv_ctr[0, 0] - self.CAM_CX) / (self.CAM_RES[0] / 2.0))
        ny = float((uv_ctr[0, 1] - self.CAM_CY) / (self.CAM_RES[1] / 2.0))

        # Apparent marker size, as pixels per ArUco cell. Below roughly 3-4
        # the detector cannot resolve the 6x6 pattern regardless of framing,
        # so this separates "out of frame" from "too small to decode".
        span = float(np.max(uv_c[:, 0]) - np.min(uv_c[:, 0]))
        px_per_cell = abs(span) / 6.0 if np.isfinite(span) else 0.0

        cause = self._classifyLossCause(corners, n_in, float(z_ctr[0]))

        return {
            'cam_u': nx,
            'cam_v': ny,
            'cam_centre_in_frame': centre_in,
            'cam_corners_in_frame': n_in,
            'cam_fully_in_frame': bool(n_in == 4),
            'cam_visible_behind': bool(float(z_ctr[0]) <= self.CAM_NEAR),
            'cam_px_per_cell': px_per_cell,
            'cam_loss_cause': cause,
            'cam_range': float(z_ctr[0]),
        }

    # ==================================================================
    # RENDERING + DETECTION
    # ==================================================================

    def _renderCamera(self):
        """Rasterise the downward view and return an HxW uint8 grayscale.

        Uses the TinyRenderer explicitly so results do not depend on whether
        a GPU/EGL path happens to be available on a given machine -- the
        laptop and the lab desktop have different GPUs.
        """
        eye, R_wc = self._cameraPose()
        forward = R_wc[:, 2]
        down = R_wc[:, 1]

        view = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=(eye + forward).tolist(),
            cameraUpVector=(-down).tolist(),
            physicsClientId=self.CLIENT,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=self.CAM_FOV_DEG,
            aspect=float(self.CAM_RES[0]) / float(self.CAM_RES[1]),
            nearVal=self.CAM_NEAR,
            farVal=self.CAM_FAR,
        )
        w, h, rgb, _, _ = p.getCameraImage(
            width=int(self.CAM_RES[0]),
            height=int(self.CAM_RES[1]),
            viewMatrix=view,
            projectionMatrix=proj,
            # Shadows off: a shadow crossing the marker changes local
            # contrast and makes detection depend on sun angle, which is not
            # what V1 is measuring.
            shadow=0,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.CLIENT,
        )
        rgb = np.reshape(np.asarray(rgb, dtype=np.uint8), (h, w, 4))
        return cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)

    def _detectMarker(self, gray):
        """Run cv2.aruco + solvePnP. Returns (detected, position_error_m)."""
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        if ids is None or self.MARKER_ID not in ids.flatten():
            return False, float('nan')

        idx = int(np.where(ids.flatten() == self.MARKER_ID)[0][0])
        img_pts = corners[idx].reshape(4, 2).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            self._markerCornersObject(), img_pts,
            self.CAM_K, self.CAM_DIST,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return True, float('nan')

        # Ground truth: marker centre in the camera frame.
        eye, R_wc = self._cameraPose()
        truth_cam = (self._markerCornersWorld().mean(axis=0) - eye) @ R_wc
        err = float(np.linalg.norm(tvec.reshape(3) - truth_cam))
        return True, err

    # ==================================================================
    # INFO -- the ONLY channel camera data uses.
    # _computeObs() is deliberately NOT overridden.
    # ==================================================================

    def _computeInfo(self):
        info = super()._computeInfo()

        if self.CAMERA_MODE == 'off':
            return info

        info.update(self._geometricCameraInfo())

        if self.CAMERA_MODE == 'render':
            gray = self._renderCamera()
            detected, err = self._detectMarker(gray)
            info['cam_aruco_detected'] = bool(detected)
            info['cam_aruco_pos_err'] = err
            if self.KEEP_IMAGE:
                info['cam_image'] = gray

        self._last_cam_info = info
        return info

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self, seed=None, options=None):
        # BaseAviary._housekeeping() calls p.resetSimulation(), which destroys
        # every body including the marker. Drop the stale handle BEFORE the
        # parent runs, since the parent's reset path reaches _updatePlatform.
        self.MARKER_BODY = None
        self._last_cam_info = {}
        return super().reset(seed=seed, options=options)

    # ==================================================================
    # ANALYTIC PREDICTION -- used by the diagnostic to check the
    # instrumentation against closed-form geometry.
    # ==================================================================

    def predictedDropoutHeight(self, horizontal_offset=0.0):
        """Closed-form height at which the marker stops being fully visible.

            h_drop = (s/2 + d_xy) / tan(FOV/2)

        Level drone, nadir camera, marker centred on the optical axis in the
        limit d_xy -> 0. This is the number that makes the point that
        "dropout height" is a property of the marker size we chose.
        """
        half_fov = np.radians(self.CAM_FOV_DEG) / 2.0
        h_above_marker = (
            (self.MARKER_SIZE / 2.0 + horizontal_offset) / np.tan(half_fov)
        )
        # Returned in the SAME datum as info['height_above_pad'], which is
        # measured from the pad's top face to the DRONE CENTRE. Two datum
        # corrections are needed to get there from the camera-to-marker
        # distance above:
        #   + MARKER_PLANE_OFFSET : marker surface sits above the pad face
        #   - CAM_OFFSET_BODY[2]  : the camera sits BELOW the drone centre
        # Omitting the second one leaves a constant 1 cm bias in every
        # predicted dropout height.
        return float(
            h_above_marker
            + self.MARKER_PLANE_OFFSET
            - float(self.CAM_OFFSET_BODY[2])
        )
