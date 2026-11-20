"""v7 S3c tests (20 Nov 2026): the student dev test script.

  1. Wilson interval (same numbers as the S2b study)
  2. end to end with an UNTRAINED student (random weights), 1 flight on the straight path,
     1 worker: the env has the student rules (45 deg, shield 0.8, touch detector), the record
     has every field, the files are written and the summary re-reads them
Run:  python -m pytest -q -s v7/tests/test_s3c_eval.py
"""
import json

import pytest
import torch

from v6_drop.learn.student import Student
from v7.scripts import eval_student_v7 as EV


def test_wilson():
    lo, hi = EV.wilson(98, 100)
    assert lo == pytest.approx(0.9300, abs=2e-4) and hi == pytest.approx(0.9945, abs=2e-4)


def test_end_to_end(tmp_path, monkeypatch):
    torch.manual_seed(0)
    ck = tmp_path / "untrained.pt"
    torch.save(dict(model=Student(256).state_dict(), size=256), ck)
    seen_env = {}
    from v7 import config as C
    real = C.make_env

    def spy(**kw):
        e = real(**kw)
        seen_env.update(pitch=e.world.cam.pitch_deg, shield=e.drop_shield, touch=e.world.touch,
                        v_max=e.v_max, rate=float(e.rate_lim[0]))
        return e
    monkeypatch.setattr(C, "make_env", spy)
    recs = EV.main(["--ckpt", str(ck), "--n", "1", "--paths", "straight", "--workers", "1",
                    "--device", "cpu"])
    assert seen_env == dict(pitch=45.0, shield=0.8, touch="sensor", v_max=9.0, rate=0.3)
    r = recs[0]
    print(f"\n[dev] untrained student: {r}")
    keys = {"mode", "seed", "outcome", "t", "landed_by", "impact", "pad_v_start", "pad_v_mean",
            "pad_v_max", "presses", "blocks", "p_drop_max", "seen"}
    assert keys <= set(r) and r["mode"] == "straight" and r["seed"] == 30000
    base = tmp_path / "dev_untrained_1"
    assert len([json.loads(x) for x in open(str(base) + ".jsonl")]) == 1
    txt = open(str(base) + ".txt").read()
    assert "ALL" in txt and "by pad top speed" in txt
    assert EV.main(["--ckpt", str(ck), "--n", "1", "--summary-only"]) == recs
