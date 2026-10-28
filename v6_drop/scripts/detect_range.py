"""v6_drop B4: detection-range study for the multi-marker pad.

Renders the real textured pad with the 60 deg camera and runs OpenCV ArUco
detection + one solvePnP, for each candidate layout, in four situations:

  approach   Shin-like Stage 1 starts: drone centre 2-5 m above the pad top,
             horizontal offset U(-3, 3) m on x and y, yaw = travel direction +-10 deg,
             level, pad centre in view (rejection sampling, as Shin et al.).
  chase      the same starts moved 4 m further back (worst-case gap growth while
             the drone speeds up gently to a 4 m/s pad); counted only when the pad
             centre is still in view.
  descent    straight line from 4 m behind / 3 m above to the drop point (feet 5 cm
             above the pad centre), 40 points, tilt -10 / 0 / +10 deg.
  drop       feet 2 / 5 / 10 cm above the pad (body centre P.REST_HEIGHT + that),
             tilt -10 / 0 / +10 deg, offsets -0.15 / 0 / +0.15 m along and across: 81 cases.

Columns: detected = at least one layout marker found and a pose computed;
big / small = which kind was found; err = |estimated - true| pad-centre position
in the body frame (cm), median and 95th percentile.

Run (from the repo root):
  python -m v6_drop.scripts.detect_range --renderer egl
"""
import argparse
import time

import numpy as np
import pybullet as p

from v6_drop.envs import lmf2_params as P
from v6_drop.envs import pad_board as PB
from v6_drop.envs.camera import Camera
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.envs.quad import make_client

FOV_MARGIN_PX = 10


def euler_R(roll, pitch, yaw):
    q = p.getQuaternionFromEuler([roll, pitch, yaw])
    return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)


def add_ground(cid, grey=0.45):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[60, 60, 0.01],
                              rgbaColor=[grey, grey, grey, 1], physicsClientId=cid)
    p.createMultiBody(0, -1, vis, [0, 0, -0.01], physicsClientId=cid)


class Scene:
    def __init__(self, layout, renderer):
        self.cid = make_client(egl=(renderer == "egl"))
        add_ground(self.cid)
        self.layout = layout
        self.plat = PB.textured_platform(self.cid, layout)
        self.plat.reset(np.random.default_rng(0), v=0.0)      # pad centre at origin, heading +x
        self.cam = Camera(self.cid, renderer=renderer)
        self.det = PB.detector()

    def close(self):
        p.disconnect(self.cid)

    def pad_in_view(self, R, pos):
        uv, d = self.cam.project(self.plat.pad_center(), R, pos)
        u, v = uv[0]
        return d[0] > 0 and FOV_MARGIN_PX <= u <= self.cam.w - FOV_MARGIN_PX \
            and FOV_MARGIN_PX <= v <= self.cam.h - FOV_MARGIN_PX

    def shot(self, R, pos):
        img = self.cam.render(R, pos)
        m = PB.measure(img, self.cam, self.layout, self.det)
        truth = R.T @ (self.plat.pad_center() - pos)
        err = np.linalg.norm(m.p_body - truth) if m.ok else np.nan
        big = self.layout.big_id in m.ids
        small = any(i != self.layout.big_id for i in m.ids)
        return dict(ok=m.ok, big=big, small=small, err=err, n=len(m.ids),
                    dist=np.linalg.norm(self.plat.pad_center() - pos))


def sample_starts(scene, rng, n, back=0.0):
    out, tries = [], 0
    while len(out) < n and tries < 100 * n:
        tries += 1
        H = rng.uniform(2.0, 5.0)
        dx, dy = rng.uniform(-3.0, 3.0, 2)
        yaw = rng.uniform(-np.radians(10), np.radians(10))
        R = euler_R(0, 0, yaw)
        pos = np.array([dx, dy, PAD_TOP_Z + H])
        if not scene.pad_in_view(R, pos):
            continue
        pos2 = pos - np.array([back, 0, 0])
        out.append((R, pos2, scene.pad_in_view(R, pos2)))
    return out


def summarize(rows):
    ok = np.array([r["ok"] for r in rows])
    err = np.array([r["err"] for r in rows if r["ok"]]) * 100
    s = dict(n=len(rows), det=ok.mean() * 100 if len(rows) else np.nan,
             big=np.mean([r["big"] for r in rows]) * 100 if rows else np.nan,
             small=np.mean([r["small"] for r in rows]) * 100 if rows else np.nan,
             med=np.median(err) if len(err) else np.nan,
             p95=np.percentile(err, 95) if len(err) else np.nan,
             dmax=max(r["dist"] for r in rows) if rows else np.nan)
    return s


def fmt(name, s):
    return (f"  {name:9s} n={s['n']:4d}  detected {s['det']:5.1f}%  big {s['big']:5.1f}%  "
            f"small {s['small']:5.1f}%  err median {s['med']:5.1f} cm  p95 {s['p95']:6.1f} cm  "
            f"max dist {s['dmax']:.1f} m")


def run_layout(layout, renderer, n_starts, seed):
    sc = Scene(layout, renderer)
    rng = np.random.default_rng(seed)
    res = {}
    try:
        res["approach"] = [sc.shot(R, pos) for R, pos, _ in sample_starts(sc, rng, n_starts)]
        rng = np.random.default_rng(seed)                     # same starts, moved back 4 m
        ch = sample_starts(sc, rng, n_starts, back=4.0)
        res["chase"] = [sc.shot(R, pos) for R, pos, vis in ch if vis]
        res["chase_inview"] = np.mean([vis for _, _, vis in ch]) * 100
        desc = []
        a, b = np.array([-4.0, 0, PAD_TOP_Z + 3.0]), np.array([0.0, 0, PAD_TOP_Z + P.REST_HEIGHT + 0.05])
        for pitch in (-10, 0, 10):
            for f in np.linspace(0, 1, 40):
                desc.append(sc.shot(euler_R(0, np.radians(pitch), 0), a + f * (b - a)))
        res["descent"] = desc
        drop = []
        for H in (P.REST_HEIGHT + 0.02, P.REST_HEIGHT + 0.05, P.REST_HEIGHT + 0.10):
            for pitch in (-10, 0, 10):
                for dx in (-0.15, 0, 0.15):
                    for dy in (-0.15, 0, 0.15):
                        drop.append(sc.shot(euler_R(0, np.radians(pitch), 0),
                                            np.array([dx, dy, PAD_TOP_Z + H])))
        res["drop"] = drop
    finally:
        sc.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderer", choices=["egl", "tiny"], default="egl")
    ap.add_argument("--layouts", nargs="+", default=list(PB.LAYOUTS))
    ap.add_argument("--starts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()
    for name in a.layouts:
        t0 = time.time()
        L = PB.LAYOUTS[name]
        res = run_layout(L, a.renderer, a.starts, a.seed)
        print(f"\n[{name}] big {L.big_size} m, small {L.small_size} m x {len(L.small_ids())}, "
              f"renderer {a.renderer}  ({time.time() - t0:.0f} s)")
        for k in ("approach", "chase", "descent", "drop"):
            print(fmt(k, summarize(res[k])))
        print(f"  chase: pad centre still in view in {res['chase_inview']:.1f}% of moved-back starts")
        d = res["descent"]
        gaps = 0
        for i in range(3):
            seq = [r["ok"] for r in d[40 * i:40 * (i + 1)]]
            gaps += sum(1 for k in range(40) if not seq[k])
        print(f"  descent: frames with NO detection: {gaps} of 120")


if __name__ == "__main__":
    main()
