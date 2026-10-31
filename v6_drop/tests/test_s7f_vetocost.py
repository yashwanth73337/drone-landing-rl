"""v6_drop Step 7f: the PRESS COST (envs/landing_env.py, veto_cost=c).

Uses TinyRenderer and untrained networks, so it needs no results/.

Checks:
  1. a BLOCKED press costs exactly c: same flight with and without the cost, the reward
     differs by c at that step and nowhere else; reward_terms['veto'] = -c
  2. an ALLOWED drop costs nothing (same end reward with and without the cost)
  3. no press -> no cost; veto_cost 0 (default) = the old rewards exactly
  4. dice_exam: the return differs by c x (blocked presses); the flight itself is identical
  5. train_ppo --veto-cost: saved in config.json, the run finishes

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s7f_vetocost.py -v -s
"""
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

C = 0.05
STILL = {"move": np.zeros(4, np.float32), "drop": 0}
DROP = {"move": np.zeros(4, np.float32), "drop": 1}


@pytest.fixture(scope="module")
def envs():
    a = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal", drop_shield=0.8)
    b = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal", drop_shield=0.8, veto_cost=C)
    yield a, b
    a.close()
    b.close()


def place(e, feet):
    pos = [0.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + feet]
    e.reset(seed=1, options=dict(pad_speed=0.0, spawn=dict(pos=pos, yaw=0.0, v=[0.0, 0.0, 0.0])))


def run(e, feet, actions):
    place(e, feet)
    out = []
    for a in actions:
        _, r, term, _, info = e.step(a)
        out.append((r, info["reward_terms"]["veto"], info["shield_veto"], term))
        if term:
            break
    return out


# ---------------------------------------------------------------- 1-3. the cost
def test_blocked_press_costs_c(envs):
    a, b = envs
    acts = [STILL, DROP, STILL, DROP, STILL]
    ra, rb = run(a, 0.10, acts), run(b, 0.10, acts)
    print(f"\n[cost] 10 cm, presses at steps 1 and 3: rewards without {[round(x[0], 4) for x in ra]}, "
          f"with {[round(x[0], 4) for x in rb]}")
    for (r0, v0, k0, _), (r1, v1, k1, _), act in zip(ra, rb, acts):
        assert k0 == k1 == bool(act["drop"]) and v0 == 0.0
        assert v1 == (-C if act["drop"] else 0.0)
        assert r1 == pytest.approx(r0 + v1, abs=1e-9)


def test_allowed_drop_costs_nothing(envs):
    a, b = envs
    ra, rb = run(a, 0.02, [DROP]), run(b, 0.02, [DROP])
    print(f"\n[allowed] 2 cm drop: end reward {ra[0][0]:.3f} without, {rb[0][0]:.3f} with the cost")
    assert ra[0][3] and rb[0][3] and not rb[0][2]
    assert rb[0][0] == pytest.approx(ra[0][0]) and rb[0][1] == 0.0


def test_default_is_old_behaviour():
    assert E.LandingEnv.__init__.__defaults__[-1] == 0.0


# ---------------------------------------------------------------- 4. dice exam
def test_dice_return_shift(envs, monkeypatch):
    a, b = envs
    monkeypatch.setattr(W, "HORIZON", 30)
    torch.manual_seed(0)
    m = Student().eval()
    torch.nn.init.zeros_(m.drop_head[-1].weight)
    torch.nn.init.constant_(m.drop_head[-1].bias, math.log(0.3 / 0.7))
    r0 = DX.fly_dice(a, m, 9002, "drop", DX.sigma_norm())
    r1 = DX.fly_dice(b, m, 9002, "drop", DX.sigma_norm())
    print(f"\n[dice] vetoes {r1['vetoes']}: return {r0['ret']} -> {r1['ret']}")
    assert r1["vetoes"] == r0["vetoes"] > 0 and r1["steps"] == r0["steps"] and r1["outcome"] == r0["outcome"]
    assert r1["ret"] == pytest.approx(r0["ret"] - C * r0["vetoes"], abs=2e-3)


# ---------------------------------------------------------------- 5. train_ppo
def test_train_ppo_veto_cost(tmp_path):
    torch.manual_seed(0)
    init = tmp_path / "init.pt"
    torch.save(dict(model=Student().state_dict(), size=256, epoch=0), init)
    TP.main(["--init", str(init), "--name", "v", "--out-root", str(tmp_path), "--workers", "1",
             "--steps", "16", "--chunk", "8", "--mb-chunks", "2", "--rounds", "1", "--critic-min", "1",
             "--renderer", "tiny", "--device", "cpu", "--shield", "0.8", "--veto-cost", str(C)])
    cfg = json.load(open(tmp_path / "v" / "config.json"))
    assert cfg["veto_cost"] == C and cfg["shield"] == 0.8 and (tmp_path / "v" / "last.pt").exists()
