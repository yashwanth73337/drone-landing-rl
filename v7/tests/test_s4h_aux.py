"""v7 S4h tests (21 Nov 2026): behaviour cloning with the helper task (v7/scripts/train_aux_v7.py).

  1. the move and drop losses are EXACTLY those of v6 train_bc.losses (same model, same batch)
  2. the helper head: (T, B, 256) -> (T, B, 8); its loss is the masked mean squared error of
     'priv'; zero mask -> zero helper loss (the blind mask also covers the helper)
  3. the helper loss reaches the CAMERA network (gradient on the first conv layer)
  4. a few steps on one batch: the helper loss falls (it can learn)
  5. the script (2 tiny epochs on CPU, from a PLAIN checkpoint): last.pt has a plain Student
     under "model" (eval_student.load reads it and the student flies one decision) and the
     helper under "aux"; --init from that helper checkpoint loads both; an existing run name
     is refused
Data: 2 demo attempts recorded with the view teacher into a temporary folder, then the V6
dataset builder (about 1 min).
Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -s v7/tests/test_s4h_aux.py
"""
import json

import numpy as np
import pytest
import torch

from v6_drop.learn.dataset import batch, build
from v6_drop.learn.student import Student
from v6_drop.scripts import train_bc as B6
from v7.scripts import record_demos_v7 as R
from v7.scripts import train_aux_v7 as A
from v7.scripts import train_bc_v7 as T7


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("aux")
    R.main(["--teacher", "view", "--attempts", "2", "--workers", "1", "--out", str(root / "demos")])
    build(str(root / "demos"), str(root / "bc_data"))
    T7.use_v7_numbers()
    return root


@pytest.fixture(scope="module")
def demos(data):
    return T7.DemosV7([str(data / "bc_data")])


@pytest.fixture(scope="module")
def plain_ckpt(data):
    torch.manual_seed(0)
    p = data / "plain.pt"
    torch.save(dict(model=Student(256).state_dict(), size=256), p)
    return str(p)


def one_batch(demos, n=4):
    pairs = demos.clip_starts("train", np.random.default_rng(0))[:n]
    return B6.to_t(batch(demos, pairs), "cpu")


# ---------------------------------------------------------------- 1. same move / drop loss
def test_move_drop_identical(demos):
    torch.manual_seed(1)
    model, aux = Student(256), A.AuxHead(256)
    b = one_batch(demos)
    lm6, ld6, st6 = B6.losses(model, b, False, "cpu", 1.0)
    lm, ld, la, st = A.losses_aux(model, aux, b, False, "cpu", 1.0)
    assert torch.equal(lm, lm6) and torch.equal(ld, ld6)
    for k in st6:
        assert torch.equal(st[k], st6[k]), k
    assert la.item() > 0


# ---------------------------------------------------------------- 2. helper head and mask
def test_head_and_mask(demos):
    aux = A.AuxHead(256)
    assert aux(torch.zeros(5, 3, 256)).shape == (5, 3, 8)
    assert demos.a["priv"].shape[1] == A.PRIV_DIM
    b = one_batch(demos)
    b["mask"] = torch.zeros_like(b["mask"])
    _, _, la, _ = A.losses_aux(Student(256), aux, b, False, "cpu")
    assert la.item() == 0.0


# ---------------------------------------------------------------- 3, 4. it trains the eyes
def test_helper_gradient_and_learning(demos):
    torch.manual_seed(2)
    model, aux = Student(256), A.AuxHead(256)
    b = one_batch(demos, n=2)
    _, _, la, _ = A.losses_aux(model, aux, b, False, "cpu")
    la.backward()
    g = model.eyes.conv[0].weight.grad
    assert g is not None and g.abs().sum().item() > 0
    opt = torch.optim.Adam(list(model.parameters()) + list(aux.parameters()), lr=1e-3)
    first = None
    for _ in range(25):
        opt.zero_grad()
        _, _, la, _ = A.losses_aux(model, aux, b, False, "cpu")
        la.backward()
        opt.step()
        first = la.item() if first is None else first
    print(f"\nhelper loss on one batch: {first:.4f} -> {la.item():.4f}")
    assert la.item() < 0.5 * first


# ---------------------------------------------------------------- 5. the script
def test_script(data, plain_ckpt):
    from v6_drop.scripts.eval_student import load
    runs = data / "runs"
    common = ["--data", str(data / "bc_data"), "--epochs", "2", "--max-clips", "3", "--batch", "2",
              "--device", "cpu", "--runs", str(runs), "--blind-mask", "3"]
    A.main(["--name", "a1", "--init", plain_ckpt] + common)
    ck = torch.load(runs / "a1" / "last.pt", weights_only=False)
    assert set(ck) >= {"model", "aux", "size", "epoch", "val_loss", "config"}
    assert set(ck["model"]) == set(Student(256).state_dict())          # a plain Student
    cfg = json.load(open(runs / "a1" / "config.json"))
    assert cfg["init_aux"] is False and cfg["aux_weight"] == A.AUX_W
    assert cfg["masked_share"] is not None
    assert (runs / "a1" / "log.csv").read_text().count("\n") == 3      # header + 2 epochs
    model, _ = load(str(runs / "a1" / "last.pt"), "cpu")               # eval / DAgger loader
    obs = dict(image=np.zeros((160, 256), np.uint8), state=np.zeros(11, np.float32))
    act, hc, p = model.act(obs)
    assert act["move"].shape == (4,) and 0.0 <= p <= 1.0
    A.main(["--name", "a2", "--init", str(runs / "a1" / "last.pt")] + common)
    assert json.load(open(runs / "a2" / "config.json"))["init_aux"] is True
    with pytest.raises(SystemExit):
        A.main(["--name", "a1", "--init", plain_ckpt] + common)
