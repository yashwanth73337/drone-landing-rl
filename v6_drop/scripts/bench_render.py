"""v6_drop diagnostic: why did rendering get slower after C0 (36 -> 17 decisions/s on the
laptop)? Changes NOTHING in the repo. Each variant runs in its own fresh process (one EGL
World per process), renders the drone's camera 100 times from the same pose, and prints
the mean time per render.

Variants:
  legs        current drone (assets/lmf2_legs.urdf: 15 visual parts)
  cube        V5 drone (assets/lmf2.urdf: 1 visual part)
  legs_1vis   legged collision shapes, but only ONE visual part (the body box)
  legs_away   current drone, but parked 50 m away (drawn only if it is in some view)

Run (from the repo root):
  python -m v6_drop.scripts.bench_render
"""
import multiprocessing as mp
import os
import re
import tempfile
import time

N = 100


def one_visual_urdf(src):
    """Copy of src with every <visual> block except the first removed."""
    text = open(src).read()
    blocks = list(re.finditer(r"<visual>.*?</visual>", text, flags=re.S))
    for b in reversed(blocks[1:]):
        text = text[:b.start()] + text[b.end():]
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path, len(blocks)


def run(variant, q):
    import numpy as np
    import pybullet as p
    from v6_drop.envs import lmf2_params as P
    from v6_drop.envs import quad as Q
    from v6_drop.envs import world as W
    from v6_drop.envs.platform_s1 import PAD_TOP_Z
    if variant == "cube":
        Q.URDF = Q.URDF_V5_CUBE
    elif variant == "legs_1vis":
        Q.URDF, _ = one_visual_urdf(Q.URDF)
    w = W.World(renderer="egl", seed=1)
    try:
        pose = [-2.0, 0.0, PAD_TOP_Z + 2.0]
        w.reset(pad_speed=0.0, spawn=dict(pos=pose, yaw=0.0))
        R = np.eye(3)
        if variant == "legs_away":
            p.resetBasePositionAndOrientation(w.quad.body, [50, 50, 50], [0, 0, 0, 1],
                                              physicsClientId=w.cid)
        n_vis = len(p.getVisualShapeData(w.quad.body, physicsClientId=w.cid))
        for _ in range(5):                                   # warm-up
            w.cam.render(R, np.array(pose))
        t0 = time.time()
        for _ in range(N):
            w.cam.render(R, np.array(pose))
        q.put((variant, n_vis, (time.time() - t0) / N * 1000))
    finally:
        w.close()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for variant in ("legs", "cube", "legs_1vis", "legs_away"):
        pr = ctx.Process(target=run, args=(variant, q))
        pr.start()
        pr.join()
        v, n_vis, ms = q.get()
        print(f"[bench] {v:10s} visual parts {n_vis:2d}  render {ms:6.1f} ms  "
              f"(~{1000 / ms:.0f} renders/s)", flush=True)


if __name__ == "__main__":
    main()
