"""v7 S4p tests (24 Nov 2026): step R reward + PPO for V7 (v7/scripts/train_ppo_v7.py).

  1. end marks (step R): success exactly V6; slide_off / tip_over / not_still -8; hard landing
     -8 - 2 (impact - 1) floored at -10 (1.0 -> -8, 1.2 -> -8.4, 1.5 -> -9, >= 2.0 -> -10);
     crash / lost / flip -10; timeout 0
  2. the ORDER the change is for: worst success > timeout > unsafe >= hard >= crash, and with
     the time cost of a full 60 s timeout (-6) a tip-over at 20 s scores LOWER than the timeout
  3. RewardV7 (what the env pays) uses the new end marks
  4. PPO settings: noise in V7 policy units (0.2 m/s / 10 = 0.02 ...), training seeds
     3,000,000 + 100,000 x worker + k, dev exam = 4 paths x n seeds from 30000
  5. a tiny real run on CPU (1 worker, 16 steps per round, 3 rounds, an exam every PPO round
     with 1 flight per path): starts from a student checkpoint WITH a helper head; writes
     last.pt / r<round>.pt ("model" = plain Student, eval_student loads it), metrics.csv
     (3 rows), exam csv (4 rows, one per path); --resume continues at the next round; an
     existing run name is refused for --init
Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -s v7/tests/test_s4p_ppo_v7.py
"""
import csv
import json

import numpy as np
import pytest
import torch

from v6_drop.envs import reward as R6
from v6_drop.learn.student import Student
from v7.envs import reward_v7 as R7
from v7.scripts import train_ppo_v7 as P7

TD_OK = dict(land_xy=np.array([0.1, 0.0]), impact=0.5, reason="drop", touched_before_cut=False)


# ---------------------------------------------------------------- 1. end marks
def test_end_marks():
    for air in (False, True):
        assert R7.end_mark("success", TD_OK, air) == R6.end_mark("success", TD_OK, air)
    for o in ("slide_off", "tip_over", "not_still"):
        assert R7.end_mark(o, {}) == -8.0
    hard = lambda v: R7.end_mark("hard_landing", dict(impact=v))
    assert hard(1.0) == pytest.approx(-8.0)
    assert hard(1.2) == pytest.approx(-8.4)
    assert hard(1.5) == pytest.approx(-9.0)
    assert hard(2.0) == pytest.approx(-10.0)
    assert hard(6.0) == pytest.approx(-10.0)
    for o in ("crash_platform", "crash_ground", "flip", "lost"):
        assert R7.end_mark(o, {}) == -10.0
    assert R7.end_mark("timeout", {}) == 0.0
    assert R6.END_MARKS["tip_over"] == -3.0                  # the V6 reward is untouched


# ---------------------------------------------------------------- 2. the order
def test_order():
    worst_success = R7.end_mark("success", dict(land_xy=np.array([0.75, 0.0]), impact=1.0,
                                                reason="touch_detect", touched_before_cut=True))
    assert worst_success == pytest.approx(10.0)
    timeout = R7.end_mark("timeout", {})
    unsafe = R7.end_mark("tip_over", {})
    hard = R7.end_mark("hard_landing", dict(impact=1.01))
    crash = R7.end_mark("lost", {})
    assert worst_success > timeout > unsafe >= hard >= crash
    time_cost = lambda s: -R6.TIME_PENALTY * 10 * s           # 10 decisions per second
    assert unsafe + time_cost(20) < timeout + time_cost(60)  # tip-over at 20 s < safe hover to 60 s
    assert R6.end_mark("tip_over", {}) + time_cost(20) > timeout + time_cost(60)   # (the V6 problem)


# ---------------------------------------------------------------- 3. the env's reward
def test_reward_v7_uses_new_marks():
    rw = R7.RewardV7("true")
    total, terms = rw.step(None, None, "tip_over", {})
    assert terms["end"] == -8.0 and total == pytest.approx(-8.0 - R6.TIME_PENALTY)
    total, terms = rw.step(None, None, "hard_landing", dict(impact=1.5))
    assert terms["end"] == pytest.approx(-9.0)


# ---------------------------------------------------------------- 4. settings
def test_settings():
    s = P7.sigma_norm()
    np.testing.assert_allclose(s, [0.02, 0.02, 0.05, 0.05], atol=1e-6)
    assert P7.train_seed(0, 0) == 3_000_000 and P7.train_seed(2, 5) == 3_200_005
    t = P7.exam_tasks(3)
    assert len(t) == 12 and t[0] == ("random", 30000) and t[-1] == ("corners", 30002)
    assert set(m for m, _ in t) == set(P7.PATHS)


# ---------------------------------------------------------------- 5. a tiny run
def test_tiny_run(tmp_path):
    from v6_drop.scripts.eval_student import load
    torch.manual_seed(0)
    init = tmp_path / "student_aux.pt"
    torch.save(dict(model=Student(256).state_dict(), aux={"net.0.weight": torch.zeros(1)}, size=256), init)
    common = ["--workers", "1", "--steps", "16", "--critic-min", "1", "--critic-max", "1",
              "--exam-every", "1", "--exam-n", "1", "--guard", "0", "--device", "cpu",
              "--out-root", str(tmp_path / "runs"), "--mb-chunks", "1", "--chunk", "8"]
    P7.main(["--init", str(init), "--name", "t", "--rounds", "3"] + common)
    out = tmp_path / "runs" / "t"
    ck = torch.load(out / "last.pt", weights_only=False)
    assert ck["round"] == 2 and ck["ppo_start"] == 1 and ck["steps"] == 3 * 16
    assert set(ck["model"]) == set(Student(256).state_dict())
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert [r["phase"] for r in rows] == ["critic", "ppo", "ppo"]
    ex = list(csv.DictReader(open(out / "exam_r0002.csv")))
    assert sorted(r["mode"] for r in ex) == sorted(P7.PATHS)
    assert (out / "r0001.pt").exists()
    model, _ = load(str(out / "r0002.pt"), "cpu")                  # eval_student_v7's loader
    cfg = json.load(open(out / "config.json"))
    assert cfg["version"] == "v7" and cfg["gamma"] == 0.99
    P7.main(["--resume", str(out / "last.pt"), "--name", "t", "--rounds", "4"] + common)
    assert torch.load(out / "last.pt", weights_only=False)["round"] == 3
    with pytest.raises(SystemExit):
        P7.main(["--init", str(init), "--name", "t", "--rounds", "1"] + common)
