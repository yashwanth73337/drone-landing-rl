"""Block 7 probe: can the CNN front-end, fed the DOWNSAMPLED image, recover where the
pad is? Supervised regression of the true body-frame relative pad position from a
single frame, on held-out spawns. This checks that preprocessing + CNN preserve the
information the estimator will need. It is not part of the paper's method.

Frames: LandingSim spawns (Table I, full DR, c = 1), plus a random 0-30 policy steps of
the oracle so that close-range views are included. Seeds from 20000 (perception base).

Usage:
  python -m v5_shin.scripts.probe_encoder --frames 3000 --epochs 30          # desktop GPU
  python -m v5_shin.scripts.probe_encoder --frames 600 --epochs 15 --quick   # CPU sanity
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from v5_shin.envs.landing_sim import LandingSim
from v5_shin.policies.encoder import EMBED_DIM, ImageEncoder, downsample, n_params
from v5_shin.policies.oracle import OracleController


def collect(n, seed, renderer="tiny", egl=False, only_in_view=True, dr=True):
    """Only frames with the pad centre in the image are kept by default: a single frame
    cannot locate an unseen pad (that is the LSTM's job). The in-view fraction of all
    visited frames is reported."""
    from v5_shin.envs.dr import DRConfig
    sim = LandingSim(seed=seed, renderer=renderer, egl=egl, dr=DRConfig() if dr else DRConfig.off())
    rng = np.random.default_rng(seed + 1)
    oracle = OracleController()
    X, Y, VIS = [], [], []
    seen = kept = 0
    try:
        while len(X) < n:
            sim.reset(c=1.0)
            for _ in range(int(rng.integers(0, 31))):
                rel_p, rel_v, s = sim.true_relative_state()
                yaw = np.arctan2(s["R"][1, 0], s["R"][0, 0])
                te, tr, _ = sim.step(oracle(rel_p, rel_v, sim.plat.velocity(), yaw))
                if te or tr:
                    break
            if sim.done:
                continue
            rel_p, _, s = sim.true_relative_state()
            uv, z = sim.cam.project(sim.plat.pad_center(), s["R"], s["pos"])
            vis = bool(z[0] > 0 and 0 <= uv[0, 0] < 512 and 0 <= uv[0, 1] < 320)
            seen += 1
            if only_in_view and not vis:
                continue
            kept += 1
            X.append(downsample(sim.render()))
            Y.append(s["R"].T @ rel_p)                        # body-frame pad position
            VIS.append(vis)
    finally:
        sim.close()
    return np.stack(X), np.array(Y, np.float32), np.array(VIS), kept / max(seen, 1)


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--renderer", default="egl", choices=["egl", "tiny"])
    ap.add_argument("--quick", action="store_true", help="tiny renderer, CPU-friendly")
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--no-dr", action="store_true", help="diagnostic: Table II DR off")
    a = ap.parse_args()
    if a.quick:
        a.renderer = "tiny"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    t0 = time.time()
    X, Y, VIS, frac_vis = collect(a.frames, a.seed, a.renderer, a.renderer == "egl", dr=not a.no_dr)
    t_col = time.time() - t0
    n_tr = int(0.8 * len(X))
    Xt = torch.tensor(X, dtype=torch.uint8, device=dev)
    mu, sd = Y[:n_tr].mean(0), Y[:n_tr].std(0)
    Yt = torch.tensor((Y - mu) / sd, device=dev)
    enc = ImageEncoder().to(dev)
    head = nn.Linear(EMBED_DIM, 3).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=3e-4)
    bs = 64
    for ep in range(a.epochs):
        perm = torch.randperm(n_tr, device=dev)
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            loss = ((head(enc(Xt[idx])) - Yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    def predict(lo, hi):
        with torch.no_grad():
            p = torch.cat([head(enc(Xt[i:min(i + 256, hi)])) for i in range(lo, hi, 256)])
        return p.cpu().numpy() * sd + mu
    pred = predict(n_tr, len(X))
    pred_tr = predict(0, n_tr)
    yte, vte = Y[n_tr:], VIS[n_tr:]
    err = np.linalg.norm(pred - yte, axis=1)
    out = dict(
        frames=len(X), train=n_tr, test=len(X) - n_tr, epochs=a.epochs, renderer=a.renderer,
        device=dev, encoder_params=n_params(enc), collect_s=round(t_col, 1),
        total_s=round(time.time() - t0, 1),
        test_r2_xyz=[round(float(v), 3) for v in r2(yte, pred)],
        train_r2_xyz=[round(float(v), 3) for v in r2(Y[:n_tr], pred_tr)],
        dr=not a.no_dr, grad_steps=a.epochs * int(np.ceil(n_tr / bs)),
        test_err_median_m=round(float(np.median(err)), 3),
        test_err_p90_m=round(float(np.percentile(err, 90)), 3),
        frac_visited_frames_pad_in_view=round(float(frac_vis), 3),
        target_std_m=[round(float(v), 3) for v in sd],
    )
    run = os.path.join(os.path.dirname(__file__), "..", "runs",
                       f"probe_encoder_{a.renderer}_n{a.frames}_e{a.epochs}{'_nodr' if a.no_dr else ''}_s{a.seed}")
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
