"""Block 7: image front-end (downsample + CNN encoder -> l_t in R^512).

The learnability check is an overfit test: a small set of rendered frames must be
fit almost perfectly (train R^2 > 0.9 on the body-frame pad position). If images
and labels were misaligned, or gradients did not reach the input layers, this would
fail. Held-out accuracy is data-limited and is measured by scripts/probe_encoder.py.

Run:  python -m pytest v5_shin/tests/test_block7_encoder.py -v -s
"""
import numpy as np
import pytest
import torch

from v5_shin.envs.dr import DRConfig
from v5_shin.envs.landing_sim import LandingSim
from v5_shin.policies.encoder import EMBED_DIM, OBS_H, OBS_W, ImageEncoder, downsample, n_params


def test_downsample_is_2x2_mean():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (320, 512), dtype=np.uint8)
    d = downsample(img)
    ref = img.reshape(160, 2, 256, 2).astype(float).mean((1, 3))
    assert d.shape == (OBS_H, OBS_W) and d.dtype == np.uint8
    assert np.abs(d - ref).max() <= 0.5 + 1e-9


def test_encoder_shapes_and_input_handling():
    torch.manual_seed(0)
    enc = ImageEncoder().eval()
    x = torch.randint(0, 256, (4, OBS_H, OBS_W), dtype=torch.uint8)
    y3, y4 = enc(x), enc(x.unsqueeze(1))
    print(f"\n[encoder] params {n_params(enc):,}, conv output {enc.n_flat}")
    assert y3.shape == (4, EMBED_DIM) and torch.isfinite(y3).all()
    assert torch.equal(y3, y4)
    xi = ImageEncoder.to_input(torch.tensor([[[0, 255]]], dtype=torch.uint8))
    assert xi.min() == -0.5 and xi.max() == 0.5


def test_no_batch_dependence():
    """No BatchNorm or similar: encoding frame by frame equals encoding the batch."""
    torch.manual_seed(0)
    enc = ImageEncoder().eval()
    x = torch.randint(0, 256, (5, OBS_H, OBS_W), dtype=torch.uint8)
    with torch.no_grad():
        batch = enc(x)
        single = torch.cat([enc(x[i:i + 1]) for i in range(5)])
    assert torch.allclose(batch, single, atol=1e-5)


def test_gradients_reach_every_parameter():
    torch.manual_seed(0)
    enc = ImageEncoder()
    x = torch.randint(0, 256, (2, OBS_H, OBS_W), dtype=torch.uint8)
    enc(x).pow(2).mean().backward()
    dead = [n for n, p in enc.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, dead


def test_overfits_rendered_frames():
    torch.manual_seed(0)
    sim = LandingSim(seed=20000, dr=DRConfig.off())
    X, Y = [], []
    try:
        while len(X) < 48:
            sim.reset(c=1.0)
            rel_p, _, s = sim.true_relative_state()
            X.append(downsample(sim.render()))
            Y.append(s["R"].T @ rel_p)
    finally:
        sim.close()
    X = torch.tensor(np.stack(X))
    Y = np.array(Y, np.float32)
    mu, sd = Y.mean(0), Y.std(0)
    Yt = torch.tensor((Y - mu) / sd)
    enc, head = ImageEncoder(), torch.nn.Linear(EMBED_DIM, 3)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=3e-4)
    for _ in range(120):
        idx = torch.randint(0, len(X), (16,))
        loss = ((head(enc(X[idx])) - Yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = head(enc(X)).numpy()
    r2 = 1 - ((pred - Yt.numpy()) ** 2).sum(0) / (Yt.numpy() ** 2).sum(0)
    print(f"\n[overfit] train R^2 xyz {np.round(r2, 3)} after 120 steps on 48 frames")
    assert (r2 > 0.9).all()
