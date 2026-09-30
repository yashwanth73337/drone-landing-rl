"""Ground plane with our own textures and Table II visual DR (V5 Blocks 3 + 6).

Rules found in Block 3 (EGL):
  - no URDF-embedded textures anywhere (they shift loadTexture ids);
  - textures are RGB with width % 4 == 0.
Rules found in Block 6:
  - a body without a visual is drawn from its collision shape (a plane gets a default
    checker texture), so the collision plane carries a hidden dummy visual;
  - PyBullet never frees visual shapes, so re-creating the ground each episode
    leaks ~40 KB/episode. Texture scale s is therefore realised with
    N_SCALES pre-built visual quads (meshScale s, same UVs, so the texture tile's
    physical size is proportional to s). Only the active quad sits at z = 0; the
    rest are parked below the far plane. s is discretised to 0.05 steps [deviation].
"""
import os

import cv2
import numpy as np
import pybullet as p

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
TEX_DIR = os.path.join(ASSETS, "ground_tex")
TILE_M = 2.0                          # m per texture repeat at s = 1
SCALES = np.round(np.arange(0.40, 1.2001, 0.05), 2)   # Table II s in [0.4, 1.2]
BASE_HALF = 100.0 / SCALES.min()      # mesh half-size so the ground is >= 100 m at s = 0.4
N_TEX = 50                            # Table II texture ids 1..50
TEX_PX = 256
PARK = [0.0, 0.0, -1000.0]


# ---------------------------------------------------------------- textures
def checker_texture(path, px=256, cells=8):
    if not os.path.exists(path):
        k = px // cells
        img = ((np.indices((px, px)) // k).sum(0) % 2 * 90 + 120).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return path


def _periodic_noise(rng, px, beta):
    """Tileable 1/f^beta noise via the FFT (periodic by construction), in [0, 1]."""
    f = np.fft.fftfreq(px)
    fr = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    fr[0, 0] = 1.0
    spec = (rng.normal(size=(px, px)) + 1j * rng.normal(size=(px, px))) / fr ** beta
    spec[0, 0] = 0
    img = np.real(np.fft.ifft2(spec))
    img -= img.min()
    return img / max(img.max(), 1e-9)


def make_texture(i, px=TEX_PX):
    """Texture i in 0..49, deterministic from seed 1300 + i. Families cycle over
    noise (grass/dirt/asphalt), tiles/checker, stripes (road), and blobs."""
    rng = np.random.default_rng(1300 + i)
    fam = i % 5
    yy, xx = np.indices((px, px))
    if fam in (0, 1):                                   # fractal noise, two colours
        m = _periodic_noise(rng, px, beta=rng.uniform(0.8, 1.8))
    elif fam == 2:                                      # checker / tiles + noise
        k = px // rng.choice([2, 4, 8, 16])
        m = ((xx // k + yy // k) % 2).astype(float)
        m = 0.8 * m + 0.2 * _periodic_noise(rng, px, 1.5)
    elif fam == 3:                                      # stripes (road markings)
        period = px // rng.choice([4, 8, 16])
        m = ((xx % period) < period * rng.uniform(0.1, 0.5)).astype(float)
        if rng.random() < 0.5:
            m = m.T
        m = 0.85 * m + 0.15 * _periodic_noise(rng, px, 1.2)
    else:                                               # blobs (patchy ground)
        m = (_periodic_noise(rng, px, 2.2) > rng.uniform(0.4, 0.6)).astype(float)
        m = 0.8 * m + 0.2 * _periodic_noise(rng, px, 1.0)
    c0, c1 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    img = (c0[None, None] * (1 - m[..., None]) + c1[None, None] * m[..., None])
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)       # RGB


def ensure_textures():
    os.makedirs(TEX_DIR, exist_ok=True)
    paths = []
    for i in range(N_TEX):
        path = os.path.join(TEX_DIR, f"tex_{i + 1:02d}.png")
        if not os.path.exists(path):
            cv2.imwrite(path, cv2.cvtColor(make_texture(i), cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


def _quad_obj(path, half, repeats):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(f"v {-half} {-half} 0\nv {half} {-half} 0\nv {half} {half} 0\nv {-half} {half} 0\n")
            f.write(f"vt 0 0\nvt {repeats} 0\nvt {repeats} {repeats}\nvt 0 {repeats}\n")
            f.write("vn 0 0 1\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n")
    return path


# ---------------------------------------------------------------- ground
class Ground:
    """self.body: the collision plane (used for contact checks).
    Visual DR: set_appearance(tex_index, scale, brightness)."""

    def __init__(self, client, randomize=False):
        self.cid = client
        self.randomize = randomize
        col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
        # A body with NO visual is drawn from its collision shape, and a PyBullet plane
        # gets a default checker texture, which z-fights with our quad at z = 0 (found in
        # Block 6: the texture-scale test read a mix of both patterns). So give the
        # collision plane a tiny transparent visual 50 m underground.
        dummy = p.createVisualShape(p.GEOM_SPHERE, radius=1e-3, rgbaColor=[0, 0, 0, 0],
                                    visualFramePosition=[0, 0, -50], physicsClientId=client)
        self.body = p.createMultiBody(0, col, dummy, [0, 0, 0], physicsClientId=client)
        repeats = int(round(2 * BASE_HALF / TILE_M))
        scales = SCALES if randomize else np.array([1.0])
        self.scales = scales
        self.visuals = []
        for s in scales:
            # One OBJ file PER scale with pre-scaled vertices and the same UV repeats.
            # (Found in Block 6: EGL shares one mesh between visual shapes loaded from
            # the same file, so meshScale variants rendered blank; TinyRenderer did not.)
            obj = _quad_obj(os.path.join(ASSETS, f"ground_quad_s{int(round(s * 100)):03d}.obj"),
                            BASE_HALF * s, repeats)
            vis = p.createVisualShape(p.GEOM_MESH, fileName=obj, physicsClientId=client)
            self.visuals.append(p.createMultiBody(0, -1, vis, PARK, physicsClientId=client))
        default = checker_texture(os.path.join(ASSETS, "ground_checker.png"))
        self.tex_ids = [p.loadTexture(default, physicsClientId=client)]
        if randomize:   # preload once: textures are never freed either
            self.tex_ids = [p.loadTexture(t, physicsClientId=client) for t in ensure_textures()]
        self.active = None
        self.set_appearance(0, 1.0, 1.0)

    def set_appearance(self, tex_index, scale, brightness):
        k = int(np.argmin(np.abs(self.scales - scale)))
        if self.active is not None and self.active != k:
            p.resetBasePositionAndOrientation(self.visuals[self.active], PARK, [0, 0, 0, 1],
                                              physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(self.visuals[k], [0, 0, 0], [0, 0, 0, 1],
                                          physicsClientId=self.cid)
        self.active = k
        b = float(brightness)
        p.changeVisualShape(self.visuals[k], -1, textureUniqueId=self.tex_ids[tex_index],
                            rgbaColor=[b, b, b, 1], physicsClientId=self.cid)
        self.appearance = dict(tex=int(tex_index), scale=float(self.scales[k]), brightness=b)
        return self.appearance

    def set_texture(self, path):
        """Test helper (Block 3): load a texture and apply it to the active visual."""
        tid = p.loadTexture(path, physicsClientId=self.cid)
        p.changeVisualShape(self.visuals[self.active], -1, textureUniqueId=tid,
                            rgbaColor=[1, 1, 1, 1], physicsClientId=self.cid)
        return tid
