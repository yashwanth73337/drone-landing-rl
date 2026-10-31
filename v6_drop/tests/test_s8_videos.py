"""v6_drop Step 8.3: videos for the professor (eval_student --videos with the shield, slow
motion of the motors-off fall, seed picking).

Uses TinyRenderer and untrained networks, so it needs no results/.

Checks:
  1. filming does NOT change the flight: the same seed with and without frames gives the
     same outcome, impact and number of steps (the frame hook only renders)
  2. the motors-off part is filmed: 25 slow-motion frames + one per 0.1 s of the check
  3. the drop note: no shield -> 'motors off'; shield at 10 cm -> BLOCKED; at 2 cm -> ALLOWED;
     no press -> nothing
  4. seed picking from a final_eval CSV: first drop per pad-speed band + the fastest pad;
     failures

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s8_videos.py -v -s
"""
import csv
import math

import numpy as np
import pytest
import torch

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.learn.student import Student
from v6_drop.scripts import eval_student as ES


@pytest.fixture(scope="module")
def env():
    e = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal")
    yield e
    e.close()


def dropper():
    """Untrained student that presses drop at once (p = 0.99): every flight ends by a drop."""
    torch.manual_seed(0)
    m = Student().eval()
    torch.nn.init.zeros_(m.drop_head[-1].weight)
    torch.nn.init.constant_(m.drop_head[-1].bias, math.log(0.99 / 0.01))
    return m


def test_filming_does_not_change_the_flight(env):
    m = dropper()
    log0, info0, _, pics0 = ES.fly(env, m, 9500, "cpu", frames=False)
    log1, info1, _, pics1 = ES.fly(env, m, 9500, "cpu", frames=True)
    td0, td1 = info0["touchdown"], info1["touchdown"]
    n_off = int(round(2.0 / P.PHYSICS_DT))
    print(f"\n[film] {info0['outcome']} by {td0['reason']}, impact {td0['impact']:.3f} vs {td1['impact']:.3f}; "
          f"steps {len(log0)} vs {len(log1)}; frames {len(pics1)}")
    assert info0["outcome"] == info1["outcome"] and len(log0) == len(log1) == 1
    assert td0["impact"] == pytest.approx(td1["impact"], abs=1e-9) and td0["reason"] == "drop"
    assert pics0 == [] and env.world.frame_hook is None
    k_max = env.world._off_k                     # motors-off physics steps that really ran
    expect = len(log1) + min(k_max, ES.SLOWMO_STEPS) + sum(1 for k in range(ES.SLOWMO_STEPS + 1, k_max + 1)
                                                           if k % 10 == 0) + 20
    assert len(pics1) == expect and 0 < k_max <= n_off


def place(e, feet):
    pos = [0.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + feet]
    e.reset(seed=1, options=dict(pad_speed=0.0, spawn=dict(pos=pos, yaw=0.0, v=[0.0, 0.0, 0.0])))


def test_drop_note(env):
    press, none = {"drop": 1}, {"drop": 0}
    env.drop_shield = None
    place(env, 0.10)
    assert ES.drop_note(env, none) == (None, None)
    assert ES.drop_note(env, press)[0] == "DROP PRESSED -> motors off"
    env.drop_shield = 0.8
    try:
        place(env, 0.10)
        blocked = ES.drop_note(env, press)[0]
        place(env, 0.02)
        allowed = ES.drop_note(env, press)[0]
    finally:
        env.drop_shield = None
    print(f"\n[note] 10 cm: {blocked}\n[note] 2 cm: {allowed}")
    assert "BLOCKED" in blocked and "ALLOWED" in allowed


def test_pick_seeds(tmp_path):
    rows = [(9500, 0.5, "success", "auto_cut"), (9501, 0.7, "success", "drop"), (9502, 1.5, "success", "drop"),
            (9503, 2.2, "timeout", ""), (9504, 2.9, "success", "drop"), (9505, 3.3, "success", "auto_cut"),
            (9506, 3.6, "success", "drop"), (9507, 3.99, "success", "auto_cut"), (9508, 1.1, "hard_landing", "drop")]
    p = tmp_path / "f.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "pad_speed", "outcome", "reason"])
        w.writerows(rows)
    assert ES.pick_seeds(str(p), "drops") == [9501, 9502, 9504, 9506, 9507]
    assert ES.pick_seeds(str(p), "failures") == [9503, 9508]
