"""v6_drop Step 8: the final evaluation (scripts/final_eval.py).

Uses untrained networks and TinyRenderer, so it needs no results/.

Checks:
  1. Wilson intervals = the values taught in 8.1 (100/100 -> 96.3-100; 350/500 -> 65.8-73.9)
  2. rule of three: 0 failures in 500 -> < 0.60 %; ~ 3/n
  3. McNemar exact test = hand values (6 vs 0 -> p 0.03125; 5 vs 5 -> 1; 80 vs 20 -> tiny;
     seeds where both agree do not matter)
  4. paired difference: mean and CI by hand
  5. describe(): counts, bands and impact numbers from hand-made rows
  6. end to end (1 worker, 2 students, 1 seed): files, main table, pair line; a second
     run into the same folder is REFUSED (the test is run once)

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s8_final.py -v -s
"""
import math
import os

import numpy as np
import pytest
import torch

from v6_drop.learn.student import Student
from v6_drop.scripts import final_eval as FE


def test_wilson_values():
    lo, hi = FE.wilson(100, 100)
    assert (round(lo, 1), round(hi, 1)) == (96.3, 100.0)
    lo, hi = FE.wilson(350, 500)
    assert (round(lo, 1), round(hi, 1)) == (65.8, 73.9)
    lo, hi = FE.wilson(297, 300)
    assert (round(lo, 1), round(hi, 1)) == (97.1, 99.7)


def test_rule_of_three():
    assert FE.rule_of_three(500) == pytest.approx(0.597, abs=0.001)
    for n in (100, 300, 1000):
        assert FE.rule_of_three(n) == pytest.approx(300 / n, rel=0.02)


def test_mcnemar():
    a = [1] * 6 + [0] * 0 + [1] * 50
    b = [0] * 6 + [1] * 50
    assert FE.mcnemar(a, b) == (6, 0, pytest.approx(0.03125))
    a = [1] * 5 + [0] * 5 + [1] * 100
    b = [0] * 5 + [1] * 5 + [1] * 100
    assert FE.mcnemar(a, b)[2] == pytest.approx(1.0)
    a = [1] * 80 + [0] * 20 + [1] * 300 + [0] * 100
    b = [0] * 80 + [1] * 20 + [1] * 300 + [0] * 100
    oa, ob, p = FE.mcnemar(a, b)
    print(f"\n[mcnemar] 80 vs 20 disagreements (400 agreements ignored): p = {p:.2e}")
    assert (oa, ob) == (80, 20) and p < 1e-8
    a2, b2 = a[:100], b[:100]                     # the same disagreements, no agreements
    assert FE.mcnemar(a2, b2)[2] == pytest.approx(p)


def test_paired_diff():
    m, lo, hi = FE.paired_diff([5, 6, 7, 8], [4, 5, 7, 6])
    d = np.array([-1, -1, 0, -2.0])
    hw = 1.96 * d.std(ddof=1) / 2
    assert (m, lo, hi) == pytest.approx((-1.0, -1 - hw, -1 + hw))


def row(seed, outcome, reason=None, impact=None, off=None, pad=0.5, t=6.0, vetoes=0):
    return dict(seed=seed, outcome=outcome, reason=reason, impact=impact, offset_cm=off, pad_speed=pad,
                time_s=t, vetoes=vetoes)


def test_describe():
    rows = [row(1, "success", "drop", 0.6, 10, 0.5, 6), row(2, "success", "drop", 0.8, 20, 1.5, 7),
            row(3, "success", "auto_cut", 0.2, 30, 2.5, 8), row(4, "timeout", pad=3.5, t=30, vetoes=4),
            row(5, "hard_landing", "drop", 1.3, pad=3.9, t=5)]
    d = FE.describe(rows)
    assert (d["n"], d["safe"], d["drop"], d["auto_cut"]) == (5, 3, 2, 1)
    assert d["outcomes"] == {"success": 3, "timeout": 1, "hard_landing": 1}
    assert d["impact_mean"] == pytest.approx((0.6 + 0.8 + 0.2) / 3) and d["impact_max"] == pytest.approx(0.8)
    assert d["offset_mean"] == pytest.approx(20) and d["vetoes"] == 4 and d["fail_bound"] is None
    assert [(b["n"], b["safe"], b["drop"]) for b in d["bands"]] == [(1, 1, 1), (1, 1, 1), (1, 1, 0), (2, 0, 0)]
    assert d["time_mean"] == pytest.approx((6 + 7 + 8 + 30 + 5) / 5)
    ok = FE.describe([row(1, "success", "drop", 0.5, 5)] * 3)
    assert ok["fail_bound"] == pytest.approx(FE.rule_of_three(3))


def test_end_to_end(tmp_path):
    torch.manual_seed(0)
    ck = tmp_path / "s.pt"
    torch.save(dict(model=Student().state_dict(), size=256, epoch=0), ck)
    args = ["--tag", "t", "--n", "1", "--seed0", "9500", "--workers", "1", "--renderer", "tiny",
            "--out-root", str(tmp_path / "out"), "--students", f"dagger_r4={ck}", f"dagger_r4_shield={ck}:0.8"]
    FE.main(args)
    out = tmp_path / "out" / "t"
    txt = open(out / "summary.txt").read()
    print("\n" + txt)
    assert (out / "flights_dagger_r4.csv").exists() and (out / "flights_dagger_r4_shield.csv").exists()
    assert "dagger_r4_shield   0.8" in txt and "dagger_r4 -> dagger_r4_shield" in txt
    with pytest.raises(SystemExit):
        FE.main(args)                              # run ONCE
