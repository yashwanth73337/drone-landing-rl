"""v6_drop T3 (8 Nov): PPO training and its dev exams with the drone's own touch detector
(scripts/train_ppo.py --touch sensor, scripts/dice_exam.py touch columns).

Checks:
  1. dice exam rows get touch columns ONLY in sensor mode (perfect: the old row exactly)
  2. the exam line adds detector / drop-after-touch / slow / missed / false-alarm counts
     only when the rows have touch columns
  3. train_ppo --touch sensor end to end: config, workers really fly in sensor mode (exam csv
     has the touch columns), metrics get det_pct / aft_pct; a perfect run has neither

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_t3_train.py -v -s
"""
import csv
import json

import torch

from v6_drop.envs import landing_env as E
from v6_drop.learn.student import Student
from v6_drop.scripts import dice_exam as DX
from v6_drop.scripts import train_ppo as TP

TOUCH_COLS = {"touches", "missed", "touched_before_cut", "detect_delay_s", "false_alarm"}


def test_dice_rows_touch_columns_only_in_sensor_mode():
    torch.manual_seed(0)
    m = Student().eval()
    sig = DX.sigma_norm(DX.SIGMA)
    rows = {}
    for touch in ("perfect", "sensor"):
        env = E.LandingEnv(renderer="tiny", seed=0, **TP.touch_kw(touch, 10))
        try:
            rows[touch] = DX.fly_dice(env, m, 9000, "clean", sig)
        finally:
            env.close()
    print(f"\n[dice] perfect keys {len(rows['perfect'])}, sensor keys {len(rows['sensor'])}, "
          f"outcome {rows['sensor']['outcome']}, touches {rows['sensor']['touches']}")
    assert not TOUCH_COLS & set(rows["perfect"])
    assert TOUCH_COLS <= set(rows["sensor"])
    same = {k: v for k, v in rows["sensor"].items() if k not in TOUCH_COLS}
    assert set(same) == set(rows["perfect"])
    assert TP.touch_kw("perfect", 10) == {}
    assert TP.touch_kw("sensor", 30) == dict(touch="sensor", touch_cfg=dict(hold_steps=30))


def test_exam_line_counts():
    base = dict(outcome="success", ret=1.0)
    old = [dict(base, reason="drop"), dict(base, reason="auto_cut")]
    line, k = TP.exam_line(old, 9)
    assert k == 2 and "detector" not in line and "auto-cut 1" in line
    new = [dict(base, reason="drop", touches=0, missed=False, touched_before_cut=False, detect_delay_s=None, false_alarm=False),
           dict(base, reason="drop", touches=1, missed=False, touched_before_cut=True, detect_delay_s=0.03, false_alarm=False),
           dict(base, reason="drop", touches=2, missed=False, touched_before_cut=True, detect_delay_s=4.9, false_alarm=False),
           dict(base, reason="touch_detect", touches=1, missed=False, touched_before_cut=True, detect_delay_s=0.11,
                false_alarm=False)]
    line, k = TP.exam_line(new, 9)
    print("\n" + line)
    assert k == 4 and "detector 1, drop after touch 2, slow 1, missed 0, false alarms 0" in line


def _train(tmp_path, name, extra):
    torch.manual_seed(0)
    init = tmp_path / "init.pt"
    torch.save(dict(model=Student().state_dict(), size=256, epoch=0), init)
    TP.main(["--init", str(init), "--name", name, "--out-root", str(tmp_path), "--workers", "1",
             "--steps", "16", "--chunk", "8", "--mb-chunks", "2", "--rounds", "2", "--critic-min", "1",
             "--critic-max", "1", "--exam-every", "1", "--exam-n", "1", "--renderer", "tiny",
             "--device", "cpu", "--actor-warmup", "1", "--guard", "0"] + extra)
    run = tmp_path / name
    return (json.load(open(run / "config.json")), list(csv.DictReader(open(run / "metrics.csv"))),
            list(csv.DictReader(open(run / "exam_r0001.csv"))))


def test_train_ppo_sensor_end_to_end(tmp_path, capsys):
    cfg, metrics, exam = _train(tmp_path, "s", ["--touch", "sensor", "--touch-hold", "10"])
    out = capsys.readouterr().out
    print(f"\n[t3] config touch {cfg['touch']} hold {cfg['touch_hold']}; metrics keys det_pct "
          f"{'det_pct' in metrics[0]}; exam columns {sorted(TOUCH_COLS & set(exam[0]))}")
    assert cfg["touch"] == "sensor" and cfg["touch_hold"] == 10
    assert "det_pct" in metrics[0] and "aft_pct" in metrics[0]
    assert TOUCH_COLS <= set(exam[0])
    assert "[exam] round 1" in out and "| detector" in out and " det " in out


def test_train_ppo_perfect_unchanged(tmp_path, capsys):
    cfg, metrics, exam = _train(tmp_path, "p", [])
    out = capsys.readouterr().out
    assert cfg["touch"] == "perfect"
    assert "det_pct" not in metrics[0] and not TOUCH_COLS & set(exam[0])
    assert "| detector" not in out and " det " not in out
