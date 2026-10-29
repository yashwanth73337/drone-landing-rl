"""v6_drop Step 6d: low-phase loss weighting (train_bc --low-weight) and the exam's drop-gate
numbers (eval_student.gate_stats).

Checks:
  1. feet_height() from 'priv' equals the true feet height in the simulator
  2. --low-weight 1 gives exactly the old move loss; W > 1 gives the hand-computed weighted
     mean (low steps count W times); masked (warm-up/padding) steps never count
  3. the low-step error statistics count only low, unmasked steps
  4. gate_stats: longest gate run, longest window stay and sinking speed on a made-up log
  5. the exam names files after the checkpoint (last.pt -> exam_*, other -> <stem>_exam_*)

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s6d_lowweight.py -v -s
"""
import numpy as np
import pytest
import torch

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.learn.student import Student
from v6_drop.scripts import eval_student as ES
from v6_drop.scripts import train_bc as TB


def test_feet_height_from_priv():
    env = E.LandingEnv(renderer="tiny", seed=0, headless=True)
    try:
        obs, _ = env.reset(seed=100005)
        for _ in range(5):
            s = env.world.true_state()
            hf = s["pos"][2] - PAD_TOP_Z - P.REST_HEIGHT
            got = TB.feet_height(torch.as_tensor(obs["priv"])).item()
            assert got == pytest.approx(hf, abs=1e-5)
            obs, *_ = env.step({"move": np.zeros(4, np.float32), "drop": 0})
    finally:
        env.close()


def make_batch(T=6, B=2):
    g = torch.Generator().manual_seed(0)
    priv = torch.zeros(T, B, 8)
    hf = torch.tensor([[2.0, 0.1], [1.0, 0.05], [0.5, 0.2], [0.29, 1.0], [0.31, 0.02], [0.1, 3.0]])
    priv[..., 7] = (hf + P.REST_HEIGHT) / float(E.PRIV_SCALE[7])
    return dict(image=torch.randint(0, 255, (T, B, 160, 256), dtype=torch.uint8, generator=g),
                state=torch.randn(T, B, 11, generator=g), priv=priv,
                move=torch.rand(T, B, 4, generator=g) * 2 - 1, drop=torch.zeros(T, B, dtype=torch.uint8),
                mask=torch.tensor([[0, 1], [1, 1], [1, 1], [1, 1], [1, 0], [1, 1]], dtype=torch.float32)), hf


def test_low_weight_loss():
    torch.manual_seed(0)
    m = Student().eval()
    b, hf = make_batch()
    with torch.no_grad():
        move, _, _, _ = m(b["image"], b["state"])
    se = ((move - b["move"]) ** 2).mean(-1)
    mask = b["mask"]
    low = (hf < TB.LOW_H).float() * mask
    with torch.no_grad():
        l1, _, st = TB.losses(m, b, False, "cpu", 1.0)
        l5, _, _ = TB.losses(m, b, False, "cpu", 5.0)
    assert l1.item() == pytest.approx(((se * mask).sum() / mask.sum()).item(), rel=1e-5)
    w = mask + 4.0 * low
    assert l5.item() == pytest.approx(((se * w).sum() / w.sum()).item(), rel=1e-5)
    assert st["n_low"].item() == low.sum().item() == 5        # feet < 0.3 m and unmasked: 5 steps
    ab = ((move - b["move"]).abs() * low.unsqueeze(-1)).sum((0, 1))
    assert torch.allclose(st["move_abs_low"], ab, atol=1e-5)


def test_gate_stats():
    base = dict(d=0.03, hf=0.03, vrel=0.05, vz_rel=0.0, tilt=1.0)
    log = [dict(base, hf=0.5)] * 3 + [dict(base)] * 2 + [dict(base, d=0.2)] + [dict(base)] * 4 \
        + [dict(base, hf=0.02 + 0.005, vz_rel=-0.2)] + [dict(base, hf=0.0)]
    g_run, w_stay, w_sink = ES.gate_stats(log)
    assert g_run == 4 and w_stay == 8
    assert w_sink == pytest.approx((0.2 + 0.0 * 7) / 8)
    assert ES.gate_stats([dict(base, hf=1.0)]) == (0, 0, None)


def test_exam_file_names(tmp_path, monkeypatch):
    original = E.LandingEnv
    monkeypatch.setattr(ES.E, "LandingEnv", lambda renderer, seed: original("tiny", seed))
    torch.manual_seed(0)
    for name, pre in (("last.pt", ""), ("best.pt", "best_")):
        ck = tmp_path / name
        torch.save(dict(model=Student().state_dict(), size=256, epoch=0), ck)
        m, _ = ES.load(str(ck), "cpu")
        ES.exam(m, str(ck), 1, "cpu")
        for f in (f"{pre}exam_1.csv", f"{pre}exam_1_summary.txt", f"{pre}exam_paths.png"):
            assert (tmp_path / f).exists(), f
    txt = (tmp_path / "exam_1_summary.txt").read_text()
    print("\n" + txt)
    assert "drop gate held" in txt
