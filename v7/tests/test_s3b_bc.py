"""v7 S3b tests (20 Nov 2026): BC with the V7 numbers.

  1. feet height from 'priv' uses the S500's 0.22 m
  2. error reports use the V7 scale (+-10 m/s): a 0.1 move error is 1.0 m/s
  3. end to end: 2 recorded V7 demos (one train, one validation episode) -> V6 dataset
     builder -> 1 short BC epoch on the CPU -> best.pt, last.pt, log.csv, config.json in
     <results>/runs/<name>; the V6 trainer's own numbers are V7 while it runs
Run:  python -m pytest -q -s v7/tests/test_s3b_bc.py
"""
import csv
import json
import os

import numpy as np
import pytest
import torch

from v6_drop.scripts import train_bc as B6
from v7 import config as C
from v7.envs.s500 import REST_HEIGHT
from v7.scripts import train_bc_v7 as T7


def test_feet_height_022():
    priv = np.zeros((2, 8), np.float32)
    priv[:, 7] = (REST_HEIGHT + np.array([0.05, 1.0])) / 5.0
    assert np.allclose(T7.feet_height(priv), [0.05, 1.0], atol=1e-6)


def test_error_report_scale():
    T7.use_v7_numbers()
    n = torch.tensor(10.0)
    acc = dict(move_abs=torch.full((4,), 1.0), n=n, move_abs_low=torch.full((4,), 1.0),
               n_low=n, drops_hit=torch.tensor(1.0), drops=torch.tensor(1.0),
               false_drop=torch.tensor(0.0), nondrop=torch.tensor(9.0))
    s = B6.summarize(acc)
    assert s["move_mae_vx"] == pytest.approx(1.0) and s["move_mae_vz"] == pytest.approx(0.1)
    assert s["move_mae_yaw_deg"] == pytest.approx(6.0, abs=1e-4)     # 0.1 x 60 deg/s


def test_end_to_end(tmp_path, monkeypatch):
    from v6_drop.learn.dataset import build
    from v6_drop.scripts.record_demos import save_chunk
    from v7.experts.teacher_v7 import TeacherV7
    from v7.experts.troubles_v7 import TroublesV7
    from v7.scripts import record_demos_v7 as R
    env = C.make_env(renderer="egl", seed=0)
    buf = {k: [] for k in R.KEYS + ("ep",)}
    try:
        for e in (1, 10):                                  # 10 % 10 == 0 -> validation
            rec, row = R.fly_and_record(env, TeacherV7(), TroublesV7(), e)
            assert row["kept"] == 1
            for k in R.KEYS:
                buf[k] += rec[k]
            buf["ep"] += [e] * row["steps"]
    finally:
        env.close()
    src, data = tmp_path / "demos", tmp_path / "bc_data"
    src.mkdir()
    save_chunk(buf, str(src / "demos_w0_0_000.npz"))
    build(str(src), str(data))
    monkeypatch.setattr(T7, "RESULTS", str(tmp_path))
    T7.main(["--name", "t", "--data", str(data), "--epochs", "1", "--max-clips", "3",
             "--device", "cpu", "--batch", "2"])
    out = tmp_path / "runs" / "t"
    for f in ("best.pt", "last.pt", "log.csv", "config.json"):
        assert (out / f).exists(), f
    cfg = json.load(open(out / "config.json"))
    assert cfg["train_episodes"] == 1 and cfg["val_episodes"] == 1 and cfg["drop_labels"] >= 2
    assert cfg["epochs"] == 1 and cfg["size"] == 256 and cfg["lr"] == 3e-4
    row = next(csv.DictReader(open(out / "log.csv")))
    print(f"\n[bc] epoch 1: val MAE vx {row['val_move_mae_vx']} m/s, drop recall {row['val_drop_recall']}")
    assert float(row["val_move_mae_vx"]) >= 0
    assert np.array_equal(B6.ACT_SCALE, T7.ACT_SCALE) and B6.feet_height is T7.feet_height
    assert os.path.normpath(B6.RESULTS) == os.path.normpath(str(tmp_path))
