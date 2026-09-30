"""Domain randomisation and sensor noise (V5 Block 6; SPEC §5, §8, Table II).

Every group can be switched off (tests, ablations); the default is all on, as
in the paper. Sampling "when" follows Table II:
  env init       controller gains                        (LandingSim.__init__)
  episode start  initial v, w; ground texture, scale, brightness; RGB scaling; light
  each step      external force and torque (held for the 10 substeps of a policy step)
"""
from dataclasses import dataclass

import numpy as np
import pybullet as p

# ---- Table II ------------------------------------------------------------------
F_EXT_MAX = 0.75                  # N per axis, world frame        [paper; frame unspecified]
M_EXT_MAX = 4e-3                  # N m per axis, body frame        [D7: printed 4e3]
V0_MAX = 1.0                      # m/s per axis, world frame       [paper]
W0_MAX = np.radians(10.0)         # rad/s per axis, body frame      [paper]
TEX_IDS = 50                      # [paper]
TEX_SCALE = (0.4, 1.2)            # [paper]
BRIGHTNESS = (0.5, 1.0)           # ground                          [paper]
RGB_SCALE = (0.5, 1.0)            # per channel, whole image before grayscale [interpretation]
LIGHT_PHI_DEG = (45.0, 135.0)     # light elevation in the world x-z plane      [interpretation]
# ---- sensors (p.5546) ----------------------------------------------------------------
VEL_NOISE_STD = 0.05              # m/s per axis, body frame        [paper]
ATT_NOISE_STD = np.radians(0.5)   # rad per axis (small-angle)      [paper]


@dataclass
class DRConfig:
    forces: bool = True
    initial_state: bool = True
    visual: bool = True
    sensor_noise: bool = True

    @classmethod
    def off(cls):
        return cls(False, False, False, False)


def sample_episode_visual(rng):
    phi = np.radians(rng.uniform(*LIGHT_PHI_DEG))
    return dict(tex=int(rng.integers(TEX_IDS)), scale=float(rng.uniform(*TEX_SCALE)),
                brightness=float(rng.uniform(*BRIGHTNESS)),
                rgb_scale=rng.uniform(*RGB_SCALE, 3),
                light_dir=np.array([np.cos(phi), 0.0, np.sin(phi)]))


def default_visual():
    return dict(tex=0, scale=1.0, brightness=1.0, rgb_scale=None,
                light_dir=np.array([0.0, 0.0, 1.0]))   # EGL light state is sticky: always pass one


def sample_initial_rates(rng):
    return rng.uniform(-V0_MAX, V0_MAX, 3), rng.uniform(-W0_MAX, W0_MAX, 3)


def sample_disturbance(rng):
    return rng.uniform(-F_EXT_MAX, F_EXT_MAX, 3), rng.uniform(-M_EXT_MAX, M_EXT_MAX, 3)


# ---- sensors ---------------------------------------------------------------------
def quat_mul(a, b):
    """Hamilton product, [x, y, z, w] order (PyBullet)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def rotvec_to_quat(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.array([0.5 * r[0], 0.5 * r[1], 0.5 * r[2], 1.0])
    ax = r / th
    return np.array([*(ax * np.sin(th / 2)), np.cos(th / 2)])


def canonical(q):
    """Unit quaternion with qw >= 0 (q and -q are the same rotation) [unspecified]."""
    q = q / np.linalg.norm(q)
    return -q if q[3] < 0 else q


def measure(state, rng, noise=True):
    """u_t = [v_body (3), q (4, [x, y, z, w], qw >= 0)], with paper noise if enabled."""
    v_b = state["R"].T @ state["v"]
    q = state["quat"]
    if noise:
        v_b = v_b + rng.normal(0.0, VEL_NOISE_STD, 3)
        q = quat_mul(q, rotvec_to_quat(rng.normal(0.0, ATT_NOISE_STD, 3)))   # body-frame error
    return np.concatenate([v_b, canonical(np.asarray(q, float))])
