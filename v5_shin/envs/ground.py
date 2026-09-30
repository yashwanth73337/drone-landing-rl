"""Ground plane with our own textures (V5 Block 3; texture DR arrives in Block 6).

Rule, found in Block 3: under the EGL renderer, textures that URDFs load
internally (e.g. pybullet_data plane.urdf) shift the texture ids returned by
p.loadTexture, so our textures render wrong. V5 therefore uses NO
URDF-embedded textures: every texture comes from p.loadTexture.
Textures must be RGB with a width that is a multiple of 4.
"""
import os

import cv2
import numpy as np
import pybullet as p

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
GROUND_HALF = 100.0     # m, 200 x 200 m quad
TILE_M = 2.0            # m of ground per texture repeat (Table II scale s multiplies this in Block 6)


def checker_texture(path, px=256, cells=8):
    if not os.path.exists(path):
        k = px // cells
        img = ((np.indices((px, px)) // k).sum(0) % 2 * 90 + 120).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return path


def _quad_obj(path, half, repeats):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(f"v {-half} {-half} 0\nv {half} {-half} 0\nv {half} {half} 0\nv {-half} {half} 0\n")
            f.write(f"vt 0 0\nvt {repeats} 0\nvt {repeats} {repeats}\nvt 0 {repeats}\n")
            f.write("vn 0 0 1\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n")
    return path


class Ground:
    def __init__(self, client, texture_path=None):
        self.cid = client
        obj = _quad_obj(os.path.join(ASSETS, "ground_quad.obj"), GROUND_HALF,
                        int(2 * GROUND_HALF / TILE_M))
        col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
        vis = p.createVisualShape(p.GEOM_MESH, fileName=obj, physicsClientId=client)
        self.body = p.createMultiBody(0, col, vis, [0, 0, 0], physicsClientId=client)
        self.set_texture(texture_path or checker_texture(os.path.join(ASSETS, "ground_checker.png")))

    def set_texture(self, path):
        tid = p.loadTexture(path, physicsClientId=self.cid)
        p.changeVisualShape(self.body, -1, textureUniqueId=tid, rgbaColor=[1, 1, 1, 1],
                            physicsClientId=self.cid)
        return tid
