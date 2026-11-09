"""v6_drop T4 (9 Nov): the drop bonus only for drops FROM THE AIR (no touch before the cut).
envs/reward.py end_mark(air_drop_only), Reward(air_drop_only), LandingEnv(air_drop_bonus),
scripts/train_ppo.py --air-drop-bonus.

Checks:
  1. end_mark: with the flag a drop after a touch loses exactly DROP_BONUS; an air drop keeps
     it; perfect-mode touchdowns (no touched_before_cut key) are unchanged; flag off = old
  2. LandingEnv (sensor mode, real reward): air drop vs touch-then-drop, flag off / on
  3. defaults stay the old behaviour; train_ppo stores and passes the flag

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_t4_airdrop.py -v -s
"""
import inspect
import json

import numpy as np
import pytest
import torch

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs import reward as R
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.learn.student import Student
from v6_drop.scripts import train_ppo as TP


def td(reason, touched=None):
    d = dict(reason=reason, land_xy=[0.1, 0.0], impact=0.5)
    if touched is not None:
        d["touched_before_cut"] = touched
    return d


def test_end_mark_rule():
    base = R.end_mark("success", td("drop", False))
    assert R.end_mark("success", td("drop", False), True) == pytest.approx(base)          # air drop
    assert R.end_mark("success", td("drop", True), False) == pytest.approx(base)          # old rule
    assert R.end_mark("success", td("drop", True), True) == pytest.approx(base - R.DROP_BONUS)
    assert R.end_mark("success", td("drop"), True) == pytest.approx(base)                 # perfect mode
    assert R.end_mark("success", td("touch_detect", True), True) == pytest.approx(base - R.DROP_BONUS)
    assert R.end_mark("hard_landing", dict(td("drop", True), impact=2.0), True) == pytest.approx(-2.0)


def fly(env, feet, press_at, max_steps=40):
    """Spawn 'feet' m above the pad, matched to a 1 m/s pad, sink at 0.2 m/s; press drop at
    step press_at (or when the referee has seen a touch, if press_at == 'touch')."""
    pos = [0.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + feet]
    env.reset(seed=1, options=dict(pad_speed=1.0, spawn=dict(pos=pos, yaw=0.0, v=[1.0, 0.0, 0.0])))
    act = {"move": np.array([1.0 / E.ACT_SCALE[0], 0.0, -0.2, 0.0], np.float32), "drop": 0}
    for k in range(max_steps):
        touched = env.world._tr["first_k"] is not None
        press = (k == press_at) if press_at != "touch" else touched
        _, r, term, trunc, info = env.step(dict(act, drop=int(press)))
        if term or trunc:
            return info
    raise AssertionError("no end")


@pytest.mark.parametrize("flag", [False, True])
def test_env_air_vs_touch_first(flag):
    env = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal", touch="sensor", air_drop_bonus=flag)
    try:
        air = fly(env, 0.03, 0)
        first = fly(env, 0.05, "touch")
    finally:
        env.close()
    ea, ef = air["reward_terms"]["end"], first["reward_terms"]["end"]
    ta, tf = air["touchdown"], first["touchdown"]
    print(f"\n[flag {flag}] air drop: touched_before {ta['touched_before_cut']}, impact {ta['impact']:.3f}, end {ea:.3f} | "
          f"touch-first drop: touched_before {tf['touched_before_cut']}, impact {tf['impact']:.3f}, end {ef:.3f}")
    assert air["outcome"] == first["outcome"] == "success"
    assert ta["reason"] == tf["reason"] == "drop"
    assert ta["touched_before_cut"] is False and tf["touched_before_cut"] is True
    expect_f = R.end_mark("success", tf, flag)
    assert ef == pytest.approx(expect_f)
    assert ea == pytest.approx(R.end_mark("success", ta, False))       # air drop: bonus either way
    no_bonus = R.end_mark("success", tf, True)
    with_bonus = R.end_mark("success", tf, False)
    assert with_bonus - no_bonus == pytest.approx(R.DROP_BONUS)


def test_defaults_are_old():
    sig = inspect.signature(E.LandingEnv.__init__)
    assert sig.parameters["air_drop_bonus"].default is False
    assert E.LandingEnv.__init__.__defaults__[-1] == 0.0                # veto_cost still last
    assert inspect.signature(R.end_mark).parameters["air_drop_only"].default is False
    assert R.Reward().air_drop_only is False


def test_train_ppo_flag(tmp_path):
    torch.manual_seed(0)
    init = tmp_path / "init.pt"
    torch.save(dict(model=Student().state_dict(), size=256, epoch=0), init)
    common = ["--init", str(init), "--out-root", str(tmp_path), "--workers", "1", "--steps", "8",
              "--chunk", "8", "--mb-chunks", "1", "--rounds", "1", "--critic-min", "1",
              "--renderer", "tiny", "--device", "cpu", "--touch", "sensor"]
    TP.main(["--name", "a"] + common + ["--air-drop-bonus"])
    TP.main(["--name", "b"] + common)
    ca = json.load(open(tmp_path / "a" / "config.json"))
    cb = json.load(open(tmp_path / "b" / "config.json"))
    print(f"\n[t4] config air_drop_bonus: with flag {ca['air_drop_bonus']}, without {cb['air_drop_bonus']}")
    assert ca["air_drop_bonus"] is True and cb["air_drop_bonus"] is False
    src = inspect.getsource(TP.main)
    assert "a.air_drop_bonus" in src                                     # passed to the workers
