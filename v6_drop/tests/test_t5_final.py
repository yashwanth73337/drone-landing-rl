"""v6_drop T5 (9 Nov): the final evaluation with the drone's own touch detector
(scripts/final_eval.py --touch sensor --touch-hold N --pairs a,b ...).

Uses untrained networks and TinyRenderer, so it needs no results/.

Checks:
  1. describe(): touch numbers (air drops, drops after a touch, detector, slow, missed,
     false alarms, hard touches, delay) from hand-made sensor rows; perfect rows: touch None
  2. end to end, sensor mode: header says ONBOARD TOUCH DETECTOR, per-student touch line,
     the chosen pair with the air-drop McNemar, touch columns in flights_*.csv
  3. end to end, perfect mode: none of the T5 lines (Step 8 output unchanged)

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_t5_final.py -v -s
"""
import csv

import pytest
import torch

from v6_drop.learn.student import Student
from v6_drop.scripts import final_eval as FE


def srow(seed, outcome, reason=None, touched=False, delay=None, missed=False, fa=False, touches=0):
    return dict(seed=seed, outcome=outcome, reason=reason, impact=0.5 if outcome == "success" else None,
                offset_cm=10.0 if outcome == "success" else None, pad_speed=1.5, time_s=6.0, vetoes=0,
                touches=touches, missed=missed, touched_before_cut=touched, detect_delay_s=delay, false_alarm=fa)


def test_describe_touch():
    rows = [srow(1, "success", "drop"),                                        # air drop
            srow(2, "success", "drop"),                                        # air drop
            srow(3, "success", "drop", touched=True, delay=0.03, touches=1),   # drop after a touch
            srow(4, "success", "touch_detect", touched=True, delay=0.12, touches=1),
            srow(5, "success", "touch_detect", touched=True, delay=0.14, touches=2),
            srow(6, "success", "drop", touched=True, delay=4.9, touches=2),    # touch-and-go
            srow(7, "timeout", touches=1, missed=True),
            srow(8, "hard_landing", "touch", touched=True, touches=1)]
    d = FE.describe(rows)
    t = d["touch"]
    print(f"\n[t5] {t}")
    assert (d["safe"], d["drop"]) == (6, 4)
    assert (t["air"], t["after"], t["detector"], t["slow"], t["missed"], t["false_alarm"], t["hard_touch"]) == \
           (2, 2, 2, 1, 1, 0, 1)
    assert t["delay_med"] == pytest.approx(0.13) and t["delay_max"] == pytest.approx(0.14)
    assert t["air_ci"] == pytest.approx(FE.wilson(2, 8))
    plain = [{k: v for k, v in r.items() if k not in ("touches", "missed", "touched_before_cut",
                                                      "detect_delay_s", "false_alarm")} for r in rows[:2]]
    assert FE.describe(plain)["touch"] is None


@pytest.fixture(scope="module")
def ck(tmp_path_factory):
    torch.manual_seed(0)
    p = tmp_path_factory.mktemp("ck") / "s.pt"
    torch.save(dict(model=Student().state_dict(), size=256, epoch=0), p)
    return p


def run(tmp_path, ck, extra):
    args = ["--tag", "t", "--n", "1", "--seed0", "9500", "--workers", "1", "--renderer", "tiny",
            "--out-root", str(tmp_path / "out"), "--students", f"a={ck}:0.8", f"b={ck}:0.8"] + extra
    FE.main(args)
    out = tmp_path / "out" / "t"
    return open(out / "summary.txt").read(), list(csv.DictReader(open(out / "flights_a.csv")))


def test_end_to_end_sensor(tmp_path, ck):
    txt, rows = run(tmp_path, ck, ["--touch", "sensor", "--pairs", "a,b"])
    print("\n" + txt)
    assert "ONBOARD TOUCH DETECTOR (hold 10 steps = 0.10 s)" in txt
    assert txt.count("touch detector: air drops") == 2
    assert "a -> b:" in txt and "air drop only-a" in txt
    assert {"touches", "missed", "touched_before_cut", "detect_delay_s", "false_alarm"} <= set(rows[0])


def test_end_to_end_perfect_unchanged(tmp_path, ck):
    txt, rows = run(tmp_path, ck, [])
    assert "ONBOARD" not in txt and "touch detector:" not in txt and "air drop only" not in txt
    assert "touches" not in rows[0]
