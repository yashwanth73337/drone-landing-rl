"""v7 S4a tests (20 Nov 2026): the DAgger collector.

  1. episode ids: round r flight i -> r x 10000 + i, never the demo ids 0..1199;
     seeds 600000 + id
  2. one DAgger flight with an UNTRAINED student: the student flies (flown = its own move),
     the label is the teacher's answer in the same state, every key has one entry per step
  3. the whole script (1 worker, 2 flights) writes .npz + flights.csv + summary.txt, refuses
     to overwrite, and the V6 dataset builder + Demos read demos and DAgger parts TOGETHER
Run:  python -m pytest -q -s v7/tests/test_s4a_dagger.py
"""
import numpy as np
import pytest
import torch

from v6_drop.learn.student import Student
from v7 import config as C
from v7.experts.teacher_v7 import TeacherV7
from v7.scripts import dagger_collect_v7 as D


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    torch.manual_seed(0)
    p = tmp_path_factory.mktemp("ck") / "untrained.pt"
    torch.save(dict(model=Student(256).state_dict(), size=256), p)
    return str(p)


def test_episode_ids():
    assert D.episode_id(1, 0) == 10000 and D.episode_id(3, 299) == 30299
    assert D.SEED0 + D.episode_id(1, 0) == 610000
    with pytest.raises(AssertionError):
        D.episode_id(1, 10000)
    with pytest.raises(AssertionError):
        D.episode_id(0, 5)                           # round 0 would collide with the demos


def test_one_flight(ckpt):
    from v6_drop.scripts.eval_student import load
    model, _ = load(ckpt, "cpu")
    env = C.make_env(renderer="egl", seed=0)
    try:
        rec, row = D.fly_dagger(env, model, TeacherV7(), D.episode_id(1, 0))
    finally:
        env.close()
    n = row["steps"]
    print(f"\n[dagger] untrained student: {row}")
    assert row["seed"] == 610000 and all(len(rec[k]) == n for k in D.KEYS)
    assert set(rec["ep"]) == {10000} and rec["t"] == list(range(n))
    move, flown = np.array(rec["move"]), np.array(rec["flown"])
    assert np.abs(move - flown).max() > 0.1          # label (teacher) != what the student flew
    assert row["err_vx"] > 0


def test_script_and_dataset(ckpt, tmp_path):
    from v6_drop.learn.dataset import Demos, build
    from v6_drop.scripts.record_demos import save_chunk
    from v7.experts.troubles_v7 import TroublesV7
    from v7.scripts import record_demos_v7 as R
    out = tmp_path / "r1"
    D.main(["--ckpt", ckpt, "--round", "1", "--flights", "2", "--workers", "1", "--out", str(out)])
    assert (out / "summary.txt").exists() and (out / "flights.csv").exists()
    assert len(list(out.glob("dagger_*.npz"))) == 1
    with pytest.raises(SystemExit):
        D.main(["--ckpt", ckpt, "--round", "1", "--flights", "2", "--workers", "1", "--out", str(out)])
    env = C.make_env(renderer="egl", seed=0)                 # one demo episode (id 1)
    try:
        rec, row = R.fly_and_record(env, TeacherV7(), TroublesV7(), 1)
    finally:
        env.close()
    buf = {k: list(rec[k]) for k in R.KEYS}
    buf["ep"] = [1] * row["steps"]
    (tmp_path / "demos").mkdir()
    save_chunk(buf, str(tmp_path / "demos" / "demos_w0_0_000.npz"))
    build(str(tmp_path / "demos"), str(tmp_path / "bc_data"))
    build(str(out), str(tmp_path / "dagger_data_r1"))
    d = Demos([str(tmp_path / "bc_data"), str(tmp_path / "dagger_data_r1")])
    assert sorted(d.ep.tolist()) == [1, 10000, 10001]
    assert d.val.tolist().count(True) == 1                  # 10000 % 10 == 0 -> validation
