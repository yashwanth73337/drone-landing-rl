"""v6_drop Step 6d: DAgger (the student flies, the teacher labels).

Uses an UNTRAINED student (fixed torch seed) and TinyRenderer, so it needs no results/.

Checks:
  1. ids and seeds: round r, flight i -> episode r x 10000 + i, seed 200000 + id; never a
     test seed (9000..) or a demo seed (100000..)
  2. the STUDENT flies: re-running the student on the saved pictures/sensor numbers from an
     empty memory gives exactly the saved 'flown' moves
  3. the labels are the TEACHER's answers in the states the student visited: replaying the
     flown actions from the same seed with a fresh teacher gives the saved labels
  4. the same flight twice records identical data
  5. dagger_collect end to end (1 worker, 2 flights): files, flights.csv, summary; the
     folder builds with learn/dataset.py; a second run into the same folder is refused
  6. Demos reads several parts as one: global rows are right, clips from the second part
     show its own pictures, validation = id % 10 == 0 over all parts, repeated ids refused
  7. train_bc with --init and two --data folders: with lr 0 the saved student equals the
     starting student (so --init really loads it); config lists both parts

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s6d_dagger.py -v -s
"""
import os
import shutil
import sys

import numpy as np
import pytest
import torch

from v6_drop.envs import landing_env as E
from v6_drop.experts.teacher import Teacher
from v6_drop.experts.troubles import Troubles
from v6_drop.learn import dataset as D
from v6_drop.learn.student import Student
from v6_drop.scripts import dagger_collect as DC
from v6_drop.scripts import record_demos as RD

EP = DC.episode_id(1, 3)                 # 10003, seed 210003


def student(path=None):
    torch.manual_seed(0)
    m = Student().eval()
    if path is not None:
        torch.save(dict(model=m.state_dict(), size=256, epoch=0), path)
    return m


@pytest.fixture(scope="module")
def env():
    torch.set_num_threads(1)
    e = E.LandingEnv(renderer="tiny", seed=0)
    yield e
    e.close()


@pytest.fixture(scope="module")
def flight(env):
    rec, row = DC.fly_dagger(env, student(), Teacher(), EP)
    print(f"\n[flight] {row}")
    return rec, row


# ---------------------------------------------------------------- 1. ids and seeds
def test_ids_and_seeds():
    assert DC.episode_id(1, 0) == 10000 and DC.episode_id(3, 299) == 30299
    seeds = [DC.SEED0 + DC.episode_id(r, i) for r in (1, 2, 3, 4) for i in (0, 9999)]
    assert min(seeds) == 210000
    assert all(s >= RD.SEED0 + 100000 for s in seeds)          # beyond demos 100000..
    with pytest.raises(AssertionError):
        DC.episode_id(1, 10000)


# ---------------------------------------------------------------- 2. the student flies
def test_student_flies(flight):
    rec, row = flight
    n = len(rec["t"])
    assert n == row["steps"] and all(len(v) == n for v in rec.values())
    assert rec["ep"] == [EP] * n and rec["t"] == list(range(n))
    m = student()
    img = torch.as_tensor(np.stack(rec["image"]))[:, None]
    st = torch.as_tensor(np.stack(rec["state"]))[:, None]
    with torch.no_grad():
        move, _, _, _ = m(img, st)
    assert np.allclose(move[:, 0].numpy(), np.stack(rec["flown"]), atol=1e-5)
    assert not np.allclose(np.stack(rec["flown"]), np.stack(rec["move"]), atol=0.05)


# ---------------------------------------------------------------- 3. teacher labels
def test_labels_are_teacher_answers(env, flight):
    rec, _ = flight
    env.reset(seed=DC.SEED0 + EP)
    t = Teacher()
    for k in range(len(rec["t"])):
        label, _ = t.act(env)
        assert np.allclose(label["move"], rec["move"][k], atol=1e-6), f"step {k}"
        assert label["drop"] == rec["drop"][k]
        env.step({"move": rec["flown"][k], "drop": 0})


# ---------------------------------------------------------------- 4. reproducible
def test_same_flight_twice(env, flight):
    rec, row = flight
    rec2, row2 = DC.fly_dagger(env, student(), Teacher(), EP)
    assert row2 == row
    for k in rec:
        assert np.array_equal(np.asarray(rec[k]), np.asarray(rec2[k])), k


# ---------------------------------------------------------------- 5. end to end
@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    root = tmp_path_factory.mktemp("dag")
    ck = root / "s.pt"
    student(ck)
    out = root / "r1"
    argv = ["dagger_collect", "--ckpt", str(ck), "--round", "1", "--flights", "2", "--workers", "1",
            "--renderer", "tiny", "--out", str(out)]
    old = sys.argv
    sys.argv = argv
    try:
        DC.main()
    finally:
        sys.argv = old
    return root, out, argv


def test_collect_end_to_end(collected):
    root, out, argv = collected
    assert os.path.exists(out / "flights.csv") and os.path.exists(out / "summary.txt")
    files = sorted(os.listdir(out))
    assert any(f.startswith("dagger_w0_") and f.endswith(".npz") for f in files)
    print("\n" + open(out / "summary.txt").read())
    n, m = D.build(str(out), str(root / "built_r1"))
    d = D.Demos(str(root / "built_r1"))
    assert m == 2 and d.ep.tolist() == [10000, 10001] and len(d.images) == n
    old = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit):
            DC.main()
    finally:
        sys.argv = old


# ---------------------------------------------------------------- 6. several parts
@pytest.fixture(scope="module")
def two_parts(env, flight, collected, tmp_path_factory):
    root = tmp_path_factory.mktemp("parts")
    src = root / "demos"
    os.makedirs(src)
    rec, row = RD.fly_and_record(env, Teacher(), Troubles(), 3)
    assert row["kept"]
    RD.save_chunk(dict(rec, ep=[3] * len(rec["t"])), str(src / "demos_w0_0_000.npz"))
    D.build(str(src), str(root / "demo_data"))
    croot = collected[0]
    return D.Demos([str(root / "demo_data"), str(croot / "built_r1")]), rec


def test_demos_several_parts(two_parts, collected):
    d, demo_rec = two_parts
    n0 = len(demo_rec["t"])
    assert d.ep.tolist() == [3, 10000, 10001]
    assert len(d.images) == len(d.a["t"]) == int(d.length.sum())
    assert d.start[1] == n0
    part2 = D.Demos(str(collected[0] / "built_r1"))
    e = 2                                             # episode 10001, second part
    r0 = int(d.start[e])
    assert np.array_equal(d.images[r0:r0 + 5], part2.images[part2.start[1]:part2.start[1] + 5])
    assert np.array_equal(d.images[r0], part2.images[part2.start[1]])
    assert np.array_equal(d.images[np.array([0, r0])], np.stack([demo_rec["image"][0], d.images[r0]]))
    c = d.clip(e, 0)
    assert np.array_equal(c["image"][0], part2.images[part2.start[1]])
    assert np.array_equal(c["move"][0], part2.a["move"][part2.start[1]])
    assert d.episodes("val").tolist() == [1] and d.episodes("train").tolist() == [0, 2]
    with pytest.raises(AssertionError):
        D.Demos([str(collected[0] / "built_r1")] * 2)


# ---------------------------------------------------------------- 7. train_bc --init
def test_train_bc_init_and_parts(two_parts, tmp_path, monkeypatch):
    from v6_drop.scripts import train_bc as TB
    d, _ = two_parts
    ck = tmp_path / "init.pt"
    m0 = student(ck)
    monkeypatch.setattr(TB, "RESULTS", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["train_bc", "--name", "t", "--data", *d.paths, "--init", str(ck),
                                      "--epochs", "1", "--batch", "2", "--max-clips", "2",
                                      "--device", "cpu", "--lr", "0"])
    TB.main()
    best = torch.load(tmp_path / "runs" / "t" / "best.pt", weights_only=False)
    for k, v in m0.state_dict().items():
        assert torch.equal(best["model"][k], v), k
    assert [p["episodes"] for p in best["config"]["parts"]] == [1, 2]
    assert best["config"]["init"] == str(ck)
