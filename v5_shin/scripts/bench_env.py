"""V5 throughput benchmark (SPEC §1 gate: run before any training).

Each worker process runs the Block 1-3 physics loop: 10 physics substeps per
policy step (controller + motors + platform), one camera render, and one
platform perturbation. Setup is excluded from the timing. The script asserts
exactly one render per policy step.

Configurations: renderer x resolution mode x number of parallel envs.
  res modes: 512x320          native render (paper resolution)
             512x320->256x160 render native, cv2 INTER_AREA downsample (SPEC D9)
             256x160          render directly at 256x160 (reference only)

Usage (from the repo root):
  python -m v5_shin.scripts.bench_env                       # full grid
  python -m v5_shin.scripts.bench_env --envs 1 4 --steps 100  # quick
Writes v5_shin/runs/bench_<timestamp>/bench.csv and config.json.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import platform as pyplatform
import time

import numpy as np

RES_MODES = {
    "512x320": ((512, 320), None),
    "512x320->256x160": ((512, 320), (256, 160)),
    "256x160": ((256, 160), None),
}


def worker(args):
    renderer, res_mode, steps, seed = args
    import cv2
    import pybullet as p
    from v5_shin.envs import lmf2_params as P
    from v5_shin.envs.camera import Camera
    from v5_shin.envs.ground import Ground
    from v5_shin.envs.platform import PAD_TOP_Z, Platform
    from v5_shin.envs.quad import LMF2Quad, make_client

    (w, h), down = RES_MODES[res_mode]
    rng = np.random.default_rng(seed)
    cid = make_client(egl=(renderer == "egl"))
    Ground(cid)
    plat = Platform(cid)
    quad = LMF2Quad(cid, rng, P.sample_gains(rng))
    cam = Camera(cid, width=w, height=h, renderer=renderer)
    plat.reset(rng, 1.0)
    quad.reset_pose([-3.0, 0.0, PAD_TOP_Z + 5.0])
    cmd = np.zeros(4)
    t_phys = t_rend = 0.0
    t0 = time.perf_counter()
    for k in range(steps):
        cmd = 0.9 * cmd + 0.1 * rng.uniform(-1, 1, 4) * [3, 3, 0.5, 0.5]
        a = time.perf_counter()
        for _ in range(P.SUBSTEPS):
            quad.apply_control(cmd)
            plat.integrate()
            p.stepSimulation(physicsClientId=cid)
        b = time.perf_counter()
        s = quad.state()
        img = cam.render(s["R"], s["pos"])
        if down is not None:
            img = cv2.resize(img, down, interpolation=cv2.INTER_AREA)
        c = time.perf_counter()
        plat.perturb(rng)
        t_phys += b - a
        t_rend += c - b
    elapsed = time.perf_counter() - t0
    assert cam.n_renders == steps, (cam.n_renders, steps)
    p.disconnect(cid)
    return dict(elapsed=elapsed, t_phys=t_phys, t_rend=t_rend, steps=steps)


def run_config(renderer, res_mode, n_envs, steps):
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_envs) as pool:
        res = pool.map(worker, [(renderer, res_mode, steps, 1000 + i) for i in range(n_envs)])
    wall = max(r["elapsed"] for r in res)
    total = sum(r["steps"] for r in res)
    return dict(
        renderer=renderer, res=res_mode, n_envs=n_envs, steps_per_env=steps,
        agg_sps=total / wall,
        per_env_sps=np.mean([r["steps"] / r["elapsed"] for r in res]),
        phys_ms=1e3 * np.mean([r["t_phys"] / r["steps"] for r in res]),
        render_ms=1e3 * np.mean([r["t_rend"] / r["steps"] for r in res]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderers", nargs="+", default=["egl", "tiny"])
    ap.add_argument("--res", nargs="+", default=list(RES_MODES))
    ap.add_argument("--envs", nargs="+", type=int, default=[1, 8, 16])
    ap.add_argument("--steps", type=int, default=300, help="policy steps per env (300 = 1 episode)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import cv2
    import pybullet
    try:
        import torch
        tv, gpu = torch.__version__, (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        tv, gpu = None, None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "runs", f"bench_{stamp}")
    os.makedirs(out, exist_ok=True)
    cfg = dict(vars(a), host=pyplatform.node(), cpu_count=os.cpu_count(),
               python=pyplatform.python_version(), numpy=np.__version__,
               cv2=cv2.__version__, torch=tv, pybullet_api=pybullet.getAPIVersion(),
               pybullet_numpy=bool(pybullet.isNumpyEnabled()), gpu=gpu)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    rows = []
    print(f"{'renderer':8} {'res':18} {'envs':>4} {'agg steps/s':>11} {'per-env':>8} "
          f"{'phys ms':>8} {'render ms':>9} {'ep/hour':>9}")
    for r in a.renderers:
        for res in a.res:
            if r == "tiny" and res != "512x320->256x160" and len(a.res) > 1:
                continue      # tiny is the fallback; bench only the D9 mode
            for n in a.envs:
                row = run_config(r, res, n, a.steps)
                row["episodes_per_hour_300step"] = row["agg_sps"] * 3600 / 300
                rows.append(row)
                print(f"{r:8} {res:18} {n:>4} {row['agg_sps']:>11.1f} {row['per_env_sps']:>8.1f} "
                      f"{row['phys_ms']:>8.2f} {row['render_ms']:>9.2f} "
                      f"{row['episodes_per_hour_300step']:>9.0f}", flush=True)
    with open(os.path.join(out, "bench.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}/bench.csv")
    print("Paper reference: ~3.5 h on an RTX 4090; Fig. 5 spans ~125k episodes (<= 300 steps each).")


if __name__ == "__main__":
    main()
