"""Forward-pitched grayscale pinhole camera (V5 Block 3).

Paper (p.5543, p.5546): grayscale, 512x320, 90 deg HORIZONTAL FOV, mounted
at a 60 deg downward pitch from the forward axis.
Body frame: x forward, y left, z up. Optical axis in body = Ry(+60 deg) x_b
= [cos60, 0, -sin60]. Image 'up' = [sin60, 0, cos60].
Exactly one render per policy step (counted; the tests assert it).
"""
import pkgutil

import numpy as np
import pybullet as p

WIDTH, HEIGHT = 512, 320           # [paper]
HFOV_DEG = 90.0                    # [paper] horizontal
PITCH_DEG = 60.0                   # [paper] below the forward axis
OFFSET_B = np.array([0.15, 0.0, 0.0])  # m, [unspecified]: just ahead of the 0.25 m visual body
NEAR, FAR = 0.05, 100.0


def vfov_deg(width=WIDTH, height=HEIGHT, hfov_deg=HFOV_DEG):
    return np.degrees(2 * np.arctan(np.tan(np.radians(hfov_deg) / 2) * height / width))


def load_egl(client):
    """Load the EGL renderer plugin into a DIRECT client. Returns the plugin id or -1."""
    egl = pkgutil.get_loader("eglRenderer")
    if egl is None:
        return -1
    return p.loadPlugin(egl.get_filename(), "_eglRendererPlugin", physicsClientId=client)


class Camera:
    def __init__(self, client, width=WIDTH, height=HEIGHT, renderer="egl"):
        self.cid = client
        self.w, self.h = width, height
        # renderer='egl' requires make_client(egl=True), i.e. the plugin is loaded
        # before any body is created. Otherwise frames come out blank.
        self.renderer = renderer
        self._pflag = (p.ER_BULLET_HARDWARE_OPENGL if renderer == "egl"
                       else p.ER_TINY_RENDERER)
        self.vfov = vfov_deg(width, height)
        self.proj = p.computeProjectionMatrixFOV(self.vfov, width / height, NEAR, FAR)
        self.fx = (width / 2) / np.tan(np.radians(HFOV_DEG) / 2)
        self.fy = (height / 2) / np.tan(np.radians(self.vfov) / 2)
        self.cx, self.cy = width / 2, height / 2
        a = np.radians(PITCH_DEG)
        self.axis_b = np.array([np.cos(a), 0.0, -np.sin(a)])
        self.up_b = np.array([np.sin(a), 0.0, np.cos(a)])
        self.left_b = np.cross(self.up_b, self.axis_b)   # image -u direction
        self.n_renders = 0

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def pose_world(self, R, pos):
        eye = pos + R @ OFFSET_B
        return eye, R @ self.axis_b, R @ self.up_b

    def render(self, R, pos, light_direction=None):
        """R: body->world, pos: body origin (world). Returns uint8 gray (h, w)."""
        eye, fwd, up = self.pose_world(R, np.asarray(pos, float))
        view = p.computeViewMatrix(eye.tolist(), (eye + fwd).tolist(), up.tolist())
        kw = {}
        if light_direction is not None:
            kw["lightDirection"] = list(light_direction)
        _, _, rgba, _, _ = p.getCameraImage(
            self.w, self.h, view, self.proj, renderer=self._pflag,
            physicsClientId=self.cid, **kw)
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(self.h, self.w, 4)[..., :3]
        self.n_renders += 1
        # ITU-R BT.601 luma, same weights as cv2.cvtColor(RGB2GRAY)
        gray = rgb @ np.array([0.299, 0.587, 0.114])
        return np.clip(np.rint(gray), 0, 255).astype(np.uint8)

    def project(self, pts_world, R, pos):
        """Analytic pinhole projection -> (N, 2) pixels (u right, v down) and depth."""
        eye, fwd, up = self.pose_world(R, np.asarray(pos, float))
        right = np.cross(fwd, up)
        d = np.atleast_2d(pts_world) - eye
        z = d @ fwd
        u = self.cx + self.fx * (d @ right) / z
        v = self.cy - self.fy * (d @ up) / z
        return np.stack([u, v], 1), z
