"""v6_drop Step 7e: the camera-based DROP SHIELD (envs/landing_env.py, drop_shield=L).

Uses TinyRenderer and untrained networks, so it needs no results/.

Checks:
  1. the shield's feet height (ArUco + solvePnP + own orientation) matches the truth at
     drop range (2-15 cm) within 3 mm; with the drone still, predicted impact = sqrt(2 g h)
  2. a drop asked at 10 cm is BLOCKED (episode goes on, info shield_veto); without the
     shield the same drop is a hard landing
  3. a drop asked at 2 cm is carried out: a safe drop, and the real impact matches the
     shield's prediction (within 0.05 m/s)
  4. the sink speed counts: at 2 cm, sinking 0.6 m/s -> predicted 0.87 > 0.8 -> blocked
  5. no confident camera measurement -> blocked
  6. dice_exam counts blocked presses (vetoes) and a blocked press is not a drop
  7. train_ppo --shield: the workers use it (vetoes column, config)

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s7e_shield.py -v -s
"""
import csv
import json
import math

import numpy as np
import pytest
import torch

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs import world as W
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.learn.student import Student
from v6_drop.scripts import dice_exam as DX
from v6_drop.scripts import train_ppo as TP

STILL = {"move": np.zeros(4, np.float32), "drop": 0}
DROP = {"move": np.zeros(4, np.float32), "drop": 1}


@pytest.fixture(scope="module")
def env():
    e = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal", drop_shield=0.8)
    yield e
    e.close()


@pytest.fixture(scope="module")
def env_off():
    e = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal")
    yield e
    e.close()


def place(e, feet, vz=0.0):
    pos = [0.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + feet]
    e.reset(seed=1, options=dict(pad_speed=0.0, spawn=dict(pos=pos, yaw=0.0, v=[0.0, 0.0, vz])))


# ---------------------------------------------------------------- 1. height
def test_shield_height_matches_truth(env):
    for feet in (0.02, 0.03, 0.05, 0.08, 0.15):
        place(env, feet)
        ok, pred, h = env.shield_check()
        true = env.world.true_state()["pos"][2] - PAD_TOP_Z - P.REST_HEIGHT
        print(f"\n[height] true feet {100 * true:.2f} cm, shield {100 * h:.2f} cm, predicted impact "
              f"{pred:.3f} m/s, allowed {ok}")
        assert h == pytest.approx(true, abs=0.003)
        assert pred == pytest.approx(math.sqrt(2 * P.G * h), abs=1e-6)
        assert ok == (pred <= 0.8)


# ---------------------------------------------------------------- 2-3. block / allow
def test_high_drop_blocked(env, env_off):
    place(env, 0.10)
    _, _, term, trunc, info = env.step(DROP)
    print(f"\n[block] 10 cm: veto {info['shield_veto']}, predicted {info['shield_pred']:.2f} m/s, "
          f"episode ended {term}")
    assert info["drop_requested"] and info["shield_veto"] and not info["drop"] and not term
    place(env_off, 0.10)
    _, _, term, _, info = env_off.step(DROP)
    print(f"[no shield] 10 cm: {info['outcome']} at {info['touchdown']['impact']:.2f} m/s")
    assert term and info["outcome"] == "hard_landing" and not info["shield_veto"]


def test_low_drop_allowed(env):
    place(env, 0.02)
    _, _, term, _, info = env.step(DROP)
    td = info["touchdown"]
    print(f"\n[allow] 2 cm: {info['outcome']} by {td['reason']}, impact {td['impact']:.3f} vs "
          f"predicted {info['shield_pred']:.3f} m/s")
    assert term and info["drop"] and not info["shield_veto"]
    assert info["outcome"] == "success" and td["reason"] == "drop"
    assert td["impact"] == pytest.approx(info["shield_pred"], abs=0.05)


# ---------------------------------------------------------------- 4-5. sink, no measurement
def test_sink_speed_counts(env):
    place(env, 0.02, vz=-0.6)
    ok, pred, h = env.shield_check()
    print(f"\n[sink] 2 cm sinking 0.6 m/s: predicted {pred:.3f} m/s, allowed {ok}")
    assert pred == pytest.approx(math.sqrt(0.36 + 2 * P.G * h), abs=1e-6) and not ok


def test_no_measurement_blocks(env):
    place(env, 0.02)
    env.last_meas = None
    _, _, term, _, info = env.step(DROP)
    assert info["shield_veto"] and info["shield_pred"] is None and not term


# ---------------------------------------------------------------- 6. dice exam
def drop_happy(p=0.3, path=None):
    torch.manual_seed(0)
    m = Student().eval()
    torch.nn.init.zeros_(m.drop_head[-1].weight)
    torch.nn.init.constant_(m.drop_head[-1].bias, math.log(p / (1 - p)))
    if path is not None:
        torch.save(dict(model=m.state_dict(), size=256, epoch=0), path)
    return m


def test_dice_exam_counts_vetoes(env, monkeypatch):
    monkeypatch.setattr(W, "HORIZON", 30)
    row = DX.fly_dice(env, drop_happy(), 9002, "drop", DX.sigma_norm())
    print(f"\n[dice] {row['outcome']} after {row['steps']} steps, vetoes {row['vetoes']}, "
          f"reason {row['reason']}")
    assert row["vetoes"] > 0 and row["reason"] != "drop" and row["feet_at_drop_cm"] is None
    assert DX.out_dir("/x/last.pt", 100, 0.8).endswith("dice_exam_100_shield80")


# ---------------------------------------------------------------- 7. train_ppo
def test_train_ppo_uses_shield(tmp_path):
    init = tmp_path / "init.pt"
    drop_happy(0.3, init)
    TP.main(["--init", str(init), "--name", "s", "--out-root", str(tmp_path), "--workers", "1",
             "--steps", "16", "--chunk", "8", "--mb-chunks", "2", "--rounds", "1", "--critic-min", "1",
             "--renderer", "tiny", "--device", "cpu", "--shield", "0.8"])
    rows = list(csv.DictReader(open(tmp_path / "s" / "metrics.csv")))
    cfg = json.load(open(tmp_path / "s" / "config.json"))
    print(f"\n[ppo] vetoes in the round: {rows[0]['vetoes']}; config shield {cfg['shield']}")
    assert cfg["shield"] == 0.8 and int(rows[0]["vetoes"]) > 0
