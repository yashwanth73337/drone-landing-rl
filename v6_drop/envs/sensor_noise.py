"""v6_drop N1 (10 Nov): SENSOR NOISE for an evaluation-only robustness test.

Why: our simulator gives the drone perfect values (exact speed, exact orientation, a clean
camera image). A real drone's estimates are never exact. This module adds realistic errors to
what the DRONE senses: the student's inputs, the drop shield and the touch detector. The
REFEREE (world.py landing checks, impact, outcome) keeps using the exact truth.

Noise (levels decided 10 Nov by Yashwanth, "yes..ok"):
  level    speed (each axis)   orientation (each axis)   camera pixels (grey 0-255)
  low      0.02 m/s            0.5 deg                   2
  medium   0.05 m/s            1.0 deg                   5
  high     0.10 m/s            2.0 deg                   10
- Speed and orientation errors DRIFT SLOWLY (first-order Gauss-Markov process, time constant
  TAU = 0.5 s, updated every 0.01 s physics step): a real estimator (EKF) smooths its errors,
  so they are correlated in time, not new white noise at 100 Hz. The value above is the
  standard deviation of the error at any moment.
  x[k+1] = a x[k] + sqrt(1 - a^2) sigma e,  a = exp(-dt / TAU),  e ~ N(0, 1).
- Camera noise is new in every frame and every pixel (sensor grain), added to the full
  512 x 320 image before ArUco and before the student's 256 x 160 picture.
- Repeatable: reseed(seed) gives the same noise for the same flight seed.
The student is NOT retrained: it never saw noise in training (a sim-to-real style check).
"""
import numpy as np

LEVELS = {"none": dict(v=0.0, tilt_deg=0.0, pix=0.0),
          "low": dict(v=0.02, tilt_deg=0.5, pix=2.0),
          "medium": dict(v=0.05, tilt_deg=1.0, pix=5.0),
          "high": dict(v=0.10, tilt_deg=2.0, pix=10.0)}
TAU = 0.5            # s, correlation time of the speed and orientation errors


def rotvec_to_R(w):
    """Rotation matrix of a small rotation vector w (rad), Rodrigues formula."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def R_to_quat(R):
    """Quaternion (x, y, z, w) of a rotation matrix (PyBullet order)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


class SensorNoise:
    def __init__(self, level="medium", v=None, tilt_deg=None, pix=None, tau=TAU, dt=0.01, seed=0):
        cfg = dict(LEVELS[level])
        if v is not None:
            cfg["v"] = v
        if tilt_deg is not None:
            cfg["tilt_deg"] = tilt_deg
        if pix is not None:
            cfg["pix"] = pix
        self.level = level
        self.sv, self.st, self.sp = float(cfg["v"]), np.radians(float(cfg["tilt_deg"])), float(cfg["pix"])
        self.tau, self.dt = float(tau), float(dt)
        self.a = float(np.exp(-self.dt / self.tau))
        self.b = float(np.sqrt(1.0 - self.a * self.a))
        self.reseed(seed)

    def reseed(self, seed):
        self.rng = np.random.default_rng([int(seed), 7_770_001])     # its own stream
        self.reset()

    def reset(self):
        """New episode: start the drifting errors at a random value of the right size."""
        self.v_err = self.rng.normal(0.0, 1.0, 3) * self.sv
        self.r_err = self.rng.normal(0.0, 1.0, 3) * self.st

    def advance(self):
        """One physics step (0.01 s)."""
        self.v_err = self.a * self.v_err + self.b * self.sv * self.rng.normal(0.0, 1.0, 3)
        self.r_err = self.a * self.r_err + self.b * self.st * self.rng.normal(0.0, 1.0, 3)

    def velocity(self, v_world):
        """The drone's own (noisy) velocity estimate, world frame."""
        return np.asarray(v_world, float) + self.v_err

    def rotation(self, R):
        """The drone's own (noisy) orientation estimate: a small rotation in the body frame."""
        return np.asarray(R, float) @ rotvec_to_R(self.r_err)

    def image(self, img):
        """Camera grain: Gaussian noise per pixel, new every frame."""
        if self.sp <= 0:
            return img
        n = self.rng.normal(0.0, self.sp, img.shape)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

    def config(self):
        return dict(level=self.level, v=self.sv, tilt_deg=float(np.degrees(self.st)), pix=self.sp, tau=self.tau)
