"""v7 S2b tests (20 Nov 2026): the teacher study script.

  1. Wilson 95 % interval: known values
  2. one task in this process: record fields, view shares in [0, 1], safe landings
  3. the whole script with 1 worker and 1 flight per path: files written, the summary
     re-reads them, the flight-time limit rule (longest safe flight x 1.3 -> 45 / 50 / 60 s)
  4. --pitch 45 reaches the env's camera (images stage), file name ends in _p45
Run:  python -m pytest -q -s v7/tests/test_s2b_study.py
"""
import json

import pytest

from v7.scripts import teacher_study_v7 as S


def test_wilson():
    lo, hi = S.wilson(98, 100)
    assert lo == pytest.approx(0.9300, abs=2e-4) and hi == pytest.approx(0.9945, abs=2e-4)
    lo, hi = S.wilson(200, 200)
    assert hi == 1.0 and lo == pytest.approx(0.9812, abs=2e-4)
    assert S.wilson(0, 0) == (0.0, 1.0)


def test_one_task_in_process():
    recs = S.run_task(dict(stage="grid", rate=0.3, vmax=9.0, mode="corners", seeds=[32000]))
    r = recs[0]
    keys = {"stage", "rate", "vmax", "mode", "seed", "outcome", "t", "pad_v_start", "pad_v_mean",
            "pad_v_max", "waits", "drops_asked", "vetoes", "reason", "impact", "view60",
            "view45", "blind60_s", "blind45_s"}
    assert keys <= set(r) and r["pitch"] == 60.0
    assert 0.0 <= r["view60"] <= 1.0 and 0.0 <= r["view45"] <= 1.0
    print(f"\n[task] {r}")
    assert r["outcome"] == "success" and r["mode"] == "corners" and r["pad_v_max"] <= 9.0


def test_script_end_to_end(tmp_path):
    base = str(tmp_path / "grid")
    recs = S.main(["--stage", "grid", "--rate", "0.3", "--vmax", "9", "--n", "1",
                   "--workers", "1", "--out", base])
    assert len(recs) == 4 and {r["mode"] for r in recs} == set(S.PATHS)
    with open(base + ".jsonl") as f:
        assert len([json.loads(x) for x in f]) == 4
    table = open(base + ".txt").read()
    assert "pad top speed 9 m/s" in table and "flight-time limit" in table
    again = S.main(["--stage", "grid", "--out", base, "--summary-only"])
    assert again == recs


def test_pitch_option(tmp_path, monkeypatch):
    seen = {}
    real = S.run_task

    def spy(task):
        seen["pitch"] = task["pitch"]
        return real(dict(task, stage="grid"))         # headless: fast; pitch still passed
    monkeypatch.setattr(S, "run_task", spy)
    monkeypatch.setattr(S, "OUT_DIR", str(tmp_path))
    recs = S.main(["--stage", "images", "--pitch", "45", "--n", "1", "--paths", "straight",
                   "--workers", "1"])
    assert seen["pitch"] == 45.0 and recs[0]["pitch"] == 45.0
    assert (tmp_path / "images_r0.3_v9_p45.jsonl").exists()
