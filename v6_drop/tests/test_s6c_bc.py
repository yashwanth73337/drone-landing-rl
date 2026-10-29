"""v6_drop Step 6c: dataset, student network, behaviour-cloning training and the exam.

Uses a tiny demo set recorded inside the test (3 attempts, TinyRenderer), so it does not
need results/demos.

Checks:
  1. build: images go to a memory-mapped file; episodes are contiguous and in time order;
     every picture equals the recorded one; validation = episodes with id % 10 == 0
  2. clips: left-aligned, up to 16 warm-up steps without loss, 32 loss steps, and over one
     epoch every step gets its loss exactly once
  3. student: output shapes; the drop chance starts at 0.5%; running a sequence at once
     equals running it step by step (memory carried correctly); act() gives a valid action
  4. learning works: on two clips, 40 training steps cut the move loss by > 70%
  5. train_bc runs end to end and writes best.pt, last.pt, log.csv, config.json
  6. the exam runs and writes its table, summary and figure

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s6c_bc.py -v -s
"""
import os
import sys

import numpy as np
import pytest
import torch

from v6_drop.envs import landing_env as E
from v6_drop.experts.teacher import Teacher
from v6_drop.experts.troubles import Troubles
from v6_drop.learn import dataset as D
from v6_drop.learn.student import Student, n_params
from v6_drop.scripts import record_demos as RD


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("bc")
    src, dst = root / "demos", root / "bc_data"
    os.makedirs(src)
    env = E.LandingEnv(renderer="tiny", seed=0)
    recs = []
    try:
        for e in (0, 3, 10):                                   # ep 0 and 10 -> validation
            rec, row = RD.fly_and_record(env, Teacher(), Troubles(), e)
            assert row["kept"]
            buf = dict(rec, ep=[e] * len(rec["t"]))
            RD.save_chunk(buf, str(src / f"demos_w0_0_{e:03d}.npz"))
            recs.append((e, rec))
    finally:
        env.close()
    n, m = D.build(str(src), str(dst))
    return D.Demos(str(dst)), recs, n


# ---------------------------------------------------------------- 1. build
def test_build(data):
    d, recs, n = data
    print(f"\n[data] {n} samples, episodes {d.ep.tolist()}, lengths {d.length.tolist()}")
    assert isinstance(d.images, np.memmap) and d.images.shape == (n, 160, 256)
    assert d.ep.tolist() == [0, 3, 10]
    for (e, rec), start, length in zip(recs, d.start, d.length):
        assert length == len(rec["t"])
        assert np.array_equal(d.images[start:start + length], np.stack(rec["image"]))
        assert np.array_equal(d.a["t"][start:start + length], np.arange(length))
        assert np.array_equal(d.a["move"][start:start + length], np.stack(rec["move"]))
    assert d.episodes("val").tolist() == [0, 2] and d.episodes("train").tolist() == [1]


# ---------------------------------------------------------------- 2. clips
def test_clips(data):
    d, _, _ = data
    e = 1
    c = d.clip(e, 40)                                          # mid-episode: full warm-up
    assert c["image"].shape == (D.BURN + D.CLIP, 160, 256)
    assert c["mask"][:D.BURN].sum() == 0 and c["mask"][D.BURN:].sum() == D.CLIP
    r0 = d.start[e]
    assert np.array_equal(c["image"][0], d.images[r0 + 40 - D.BURN])
    c0 = d.clip(e, -10)                                        # first, short clip
    assert c0["mask"][:22].sum() == 22 and c0["mask"][22:].sum() == 0
    assert np.array_equal(c0["image"][0], d.images[r0])        # starts at the episode start
    covered = np.zeros(d.length[e])
    for ee, s in d.clip_starts("train", np.random.default_rng(3)):
        lo, hi = max(s, 0), min(s + D.CLIP, d.length[ee])
        covered[lo:hi] += 1
    assert np.all(covered == 1)


# ---------------------------------------------------------------- 3. student
def test_student_shapes_and_memory():
    torch.manual_seed(0)
    m = Student()
    T, B = 6, 2
    img = torch.randint(0, 255, (T, B, 160, 256), dtype=torch.uint8)
    st = torch.randn(T, B, 11)
    move, logit, feats, hc = m(img, st)
    print(f"\n[student] {n_params(m):,} parameters; start drop chance {torch.sigmoid(logit).mean():.4f}")
    assert move.shape == (T, B, 4) and logit.shape == (T, B) and feats.shape == (T, B, 256)
    assert torch.all(move.abs() <= 1)
    assert torch.sigmoid(logit).mean().item() == pytest.approx(0.005, rel=0.3)
    hc2, steps = None, []
    for t in range(T):
        mv, _, _, hc2 = m(img[t:t + 1], st[t:t + 1], hc2)
        steps.append(mv)
    assert torch.allclose(torch.cat(steps), move, atol=1e-5)
    obs = {"image": img[0, 0].numpy(), "state": st[0, 0].numpy()}
    act, hc3, p = m.act(obs)
    assert act["move"].shape == (4,) and act["drop"] in (0, 1) and 0 < p < 1


# ---------------------------------------------------------------- 4. learning works
def test_overfits_two_clips(data):
    from v6_drop.scripts.train_bc import losses
    d, _, _ = data
    torch.manual_seed(0)
    m = Student()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    b = {k: torch.as_tensor(v) for k, v in D.batch(d, [(1, 0), (1, 32)]).items()}
    first = None
    for i in range(40):
        l_move, l_drop, _ = losses(m, b, False, "cpu")
        if first is None:
            first = l_move.item()
        opt.zero_grad()
        (l_move + l_drop).backward()
        opt.step()
    print(f"\n[overfit] move loss {first:.4f} -> {l_move.item():.4f}")
    assert l_move.item() < 0.3 * first


# ---------------------------------------------------------------- 5. train_bc end to end
def test_train_bc_script(data, tmp_path, monkeypatch):
    from v6_drop.scripts import train_bc as TB
    d, _, _ = data
    monkeypatch.setattr(TB, "RESULTS", str(tmp_path))
    data_dir = os.path.dirname(d.images.filename)
    monkeypatch.setattr(sys, "argv", ["train_bc", "--name", "t", "--data", data_dir, "--epochs", "2",
                                      "--batch", "2", "--max-clips", "2", "--device", "cpu"])
    TB.main()
    run = tmp_path / "runs" / "t"
    for f in ("best.pt", "last.pt", "log.csv", "config.json"):
        assert os.path.exists(run / f)
    ck = torch.load(run / "best.pt", weights_only=False)
    assert ck["size"] == 256 and ck["epoch"] in (1, 2)


# ---------------------------------------------------------------- 6. exam
def test_exam_runs(tmp_path, monkeypatch):
    from v6_drop.scripts import eval_student as ES
    ck = tmp_path / "s.pt"
    torch.save(dict(model=Student().state_dict(), size=256, epoch=0), ck)
    original = E.LandingEnv
    monkeypatch.setattr(ES.E, "LandingEnv", lambda renderer, seed: original("tiny", seed))
    m, _ = ES.load(str(ck), "cpu")
    ES.exam(m, str(ck), 2, "cpu")
    for f in ("exam_2.csv", "exam_2_summary.txt", "exam_paths.png"):
        assert os.path.exists(tmp_path / f)
