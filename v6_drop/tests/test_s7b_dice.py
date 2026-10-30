"""v6_drop Step 7b: the dice exam (scripts/dice_exam.py).

Uses an UNTRAINED student (fixed torch seed) and TinyRenderer, so it needs no results/.

Checks:
  1. the noise width in policy units: 0.2 m/s on vx -> 0.2 / 6; 3 deg/s yaw -> 3 / 60
  2. choose(): clean = the exam rule (mean move, drop if p > 0.5, no random draws); move
     noise has the right width per output and is clipped to [-1, 1]; the drop coin falls
     with chance p; 'coin' marks only drops the clean rule would not make (p <= 0.5)
  3. mode clean flies exactly like the real exam (eval_student.fly): same outcome, same
     number of steps
  4. reproducible: the same (seed, mode) twice gives the same row; dice change the flight
  5. a student whose drop chance is 0.3 everywhere: the coin drops it within a few steps
     (coin_drop True, drop chance 0.3 recorded), clean never drops, coin_risk = 1 - 0.7^steps
  6. end to end (1 worker, 1 seed, modes clean + both): flights.csv, summary.txt with both
     modes and the decisions/s line

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s7b_dice.py -v -s
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

from v6_drop.envs import landing_env as E
from v6_drop.envs import world as W
from v6_drop.learn.student import Student
from v6_drop.scripts import dice_exam as DX
from v6_drop.scripts import eval_student as ES


def student(path=None, p_drop=None):
    torch.manual_seed(0)
    m = Student().eval()
    if p_drop is not None:
        last = m.drop_head[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.constant_(last.bias, math.log(p_drop / (1 - p_drop)))
    if path is not None:
        torch.save(dict(model=m.state_dict(), size=256, epoch=0), path)
    return m


@pytest.fixture(scope="module")
def env():
    torch.set_num_threads(1)
    e = E.LandingEnv(renderer="tiny", seed=0)
    yield e
    e.close()


@pytest.fixture
def short(monkeypatch):
    monkeypatch.setattr(W, "HORIZON", 40)


# ---------------------------------------------------------------- 1. units
def test_sigma_norm():
    s = DX.sigma_norm((0.2, 0.2, 0.05, 3.0))
    assert s == pytest.approx([0.2 / 6, 0.2 / 6, 0.05 / 1.0, 3.0 / 60.0])


# ---------------------------------------------------------------- 2. choose()
def test_choose_modes():
    sig = DX.sigma_norm()
    mean = np.array([0.1, -0.2, 0.0, 0.3])
    rng = np.random.default_rng(0)
    for p in (0.2, 0.7):
        mv, dr, coin = DX.choose(mean, p, "clean", sig, rng)
        assert np.array_equal(mv, mean.astype(np.float32)) and dr == int(p > 0.5) and not coin
    state = rng.bit_generator.state
    DX.choose(mean, 0.2, "clean", sig, rng)
    assert rng.bit_generator.state == state                      # clean draws nothing
    moves = np.array([DX.choose(mean, 0.2, "move", sig, rng)[0] for _ in range(20000)])
    print(f"\n[noise] std {np.round(moves.std(0), 4).tolist()} vs sigma {np.round(sig, 4).tolist()}")
    assert moves.std(0) == pytest.approx(sig, rel=0.03)
    assert moves.mean(0) == pytest.approx(mean, abs=0.002)
    edge = np.array([DX.choose([0.999, 0, 0, 0], 0.0, "move", sig, rng)[0][0] for _ in range(2000)])
    assert edge.max() <= 1.0 and (edge == 1.0).any()             # clipped like the env
    for p in (0.3, 0.7):
        res = [DX.choose(mean, p, "drop", sig, rng) for _ in range(20000)]
        freq = np.mean([d for _, d, _ in res])
        print(f"[coin] p {p}: dropped {freq:.3f} of the steps")
        assert freq == pytest.approx(p, abs=0.01)
        assert all(np.array_equal(m, mean.astype(np.float32)) for m, _, _ in res[:50])
        assert all(c == (d == 1 and p <= 0.5) for _, d, c in res)


# ---------------------------------------------------------------- 3. clean = the exam
def test_clean_equals_exam(env, short):
    m = student()
    log, info, _, _ = ES.fly(env, m, 9000, "cpu")
    row = DX.fly_dice(env, m, 9000, "clean", DX.sigma_norm())
    print(f"\n[clean] exam {info['outcome']} {len(log)} steps; dice clean {row['outcome']} {row['steps']} steps")
    assert row["outcome"] == info["outcome"] and row["steps"] == len(log)


# ---------------------------------------------------------------- 4. reproducible
def test_reproducible(env, short):
    m = student()
    a = DX.fly_dice(env, m, 9001, "both", DX.sigma_norm())
    b = DX.fly_dice(env, m, 9001, "both", DX.sigma_norm())
    c = DX.fly_dice(env, m, 9001, "clean", DX.sigma_norm())
    print(f"\n[repeat] both {a['ret']} / {b['ret']}; clean {c['ret']}")
    assert a == b
    assert a["ret"] != c["ret"]


# ---------------------------------------------------------------- 5. the coin
def test_coin_drop(env, short):
    m = student(p_drop=0.3)
    d = DX.fly_dice(env, m, 9002, "drop", DX.sigma_norm())
    c = DX.fly_dice(env, m, 9002, "clean", DX.sigma_norm())
    print(f"\n[coin] drop mode: {d['outcome']} by {d['reason']} after {d['steps']} steps, feet "
          f"{d['feet_at_drop_cm']} cm, p {d['drop_chance_at_drop']}; clean: {c['outcome']} {c['steps']} steps, "
          f"coin risk {c['coin_risk']}")
    assert d["reason"] == "drop" and d["coin_drop"] and d["steps"] <= 20
    assert d["drop_chance_at_drop"] == pytest.approx(0.3, abs=1e-3) and d["feet_at_drop_cm"] is not None
    assert c["reason"] != "drop" and not c["coin_drop"]
    assert c["coin_risk"] == pytest.approx(1 - 0.7 ** c["steps"], abs=1e-3)


# ---------------------------------------------------------------- 6. end to end
def test_end_to_end(tmp_path):
    ck = tmp_path / "s.pt"
    student(ck)
    old = sys.argv
    sys.argv = ["dice_exam", "--ckpt", str(ck), "--n", "1", "--workers", "1", "--modes", "clean", "both",
                "--renderer", "tiny"]
    try:
        DX.main()
    finally:
        sys.argv = old
    out = DX.out_dir(str(ck), 1)
    assert out.endswith("s_dice_exam_1")
    txt = open(os.path.join(out, "summary.txt")).read()
    assert "== clean" in txt and "== both" in txt and "decisions/s" in txt and "== move" not in txt
    rows = open(os.path.join(out, "flights.csv")).read().strip().splitlines()
    assert len(rows) == 3 and rows[1].split(",")[1] == "clean" and rows[2].split(",")[1] == "both"
