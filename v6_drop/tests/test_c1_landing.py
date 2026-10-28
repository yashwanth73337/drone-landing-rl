"""v6_drop C1: the drop, the auto-cut on touch, the flight checks and the five
landing checks (decided 28 Oct: D1 2 s window, D2 auto-cut, D3 flight rules,
D4 exact free-fall impact).

Every outcome is produced on purpose by a hand-built situation:
  success (drop and auto-cut), hard_landing (drop and auto-cut), slide_off,
  not_still, tip_over, crash_platform, crash_ground, flip, lost, timeout.
Plus: the exact impact formula matches sqrt(2 g h) to 0.005 m/s, and a finished
episode refuses further steps.

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_c1_landing.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import lmf2_params as P
from v6_drop.envs import world as W
from v6_drop.envs.platform_s1 import PAD_TOP_Z


@pytest.fixture(scope="module")
def world():
    w = W.World(renderer="tiny", seed=3, gains_mode="nominal")   # no rendering needed here
    yield w
    w.close()


def start(world, pad_speed, rel_xyz, dv=(0.0, 0.0, 0.0), w_body=(0.0, 0.0, 0.0), yaw=0.0):
    """Drone at pad centre + rel_xyz (world axes; z = height of the FEET above the pad
    top when rel_xyz[2] is given as a feet gap), moving with the pad + dv."""
    pos = np.array([rel_xyz[0], rel_xyz[1], PAD_TOP_Z + P.REST_HEIGHT + rel_xyz[2]])
    v = np.array([pad_speed, 0.0, 0.0]) + np.array(dv)
    world.reset(pad_speed=pad_speed, spawn=dict(pos=pos, yaw=yaw, v=v, w_body=np.array(w_body)))


def fly(world, cmd, max_steps=300, drop_at=None):
    for k in range(max_steps):
        term, trunc, info = world.step(cmd, drop=(drop_at is not None and k == drop_at))
        if term or trunc:
            return term, trunc, info
    raise AssertionError("episode did not end")


def show(tag, term, trunc, info):
    td = info.get("touchdown") or {}
    imp = td.get("impact")
    print(f"\n[{tag}] outcome={info['outcome']} term={term} trunc={trunc} t={info['t']} "
          f"reason={td.get('reason')} impact={None if imp is None else round(imp, 3)} "
          f"({td.get('impact_method')}) land_xy={np.round(td.get('land_xy') or [np.nan] * 2, 3)} "
          f"slide={td.get('slide')} v_rel_end={td.get('v_rel_end')}")


# ---------------------------------------------------------------- success by drop
@pytest.mark.parametrize("pad_speed", [0.0, 2.0, 4.0])
def test_drop_success(world, pad_speed):
    start(world, pad_speed, (0.0, 0.0, 0.03))
    term, trunc, info = world.step([0, 0, 0, 0], drop=True)
    show(f"drop success v={pad_speed}", term, trunc, info)
    td = info["touchdown"]
    assert term and not trunc and info["outcome"] == "success"
    assert td["reason"] == "drop" and td["impact_method"] == "free_fall_exact"
    assert td["impact"] == pytest.approx(np.sqrt(2 * P.G * 0.03), abs=0.005)
    assert np.linalg.norm(td["land_xy"]) < 0.01 and td["v_rel_end"] <= W.V_STILL
    assert td["strict_soft"] is False or td["impact"] <= W.V_IMPACT_STRICT


# ---------------------------------------------------------------- D4: exact impact
@pytest.mark.parametrize("h", [0.02, 0.05, 0.08])
def test_exact_free_fall_impact(world, h):
    start(world, 3.0, (0.0, 0.0, h))
    term, trunc, info = world.step([0, 0, 0, 0], drop=True)
    imp, theory = info["touchdown"]["impact"], np.sqrt(2 * P.G * h)
    print(f"\n[exact impact {h*100:.0f} cm] {imp:.3f} m/s, theory {theory:.3f} m/s")
    assert imp == pytest.approx(theory, abs=0.005)
    assert info["outcome"] == ("success" if theory <= W.V_IMPACT_MAX else "hard_landing")


def test_drop_hard_landing(world):
    start(world, 3.0, (0.0, 0.0, 0.15))                    # 15 cm -> 1.72 m/s
    term, trunc, info = world.step([0, 0, 0, 0], drop=True)
    show("drop 15 cm", term, trunc, info)
    assert info["outcome"] == "hard_landing"
    assert info["touchdown"]["impact"] == pytest.approx(np.sqrt(2 * P.G * 0.15), abs=0.005)


# ---------------------------------------------------------------- auto-cut (Option A)
def test_auto_cut_soft_touch_is_success(world):
    start(world, 2.0, (0.0, 0.0, 0.25))
    term, trunc, info = fly(world, [2.0, 0.0, -0.4, 0.0])    # match the pad, descend gently
    show("auto-cut soft", term, trunc, info)
    td = info["touchdown"]
    assert info["outcome"] == "success" and td["reason"] == "auto_cut"
    assert td["impact_method"] == "step_reading" and td["impact"] < 0.6


def test_auto_cut_fast_touch_is_hard_landing(world):
    start(world, 2.0, (0.0, 0.0, 1.0))
    term, trunc, info = fly(world, [2.0, 0.0, -2.5, 0.0])    # dive onto the pad
    show("auto-cut fast", term, trunc, info)
    assert info["outcome"] == "hard_landing" and info["touchdown"]["reason"] == "auto_cut"


# ---------------------------------------------------------------- the other landing checks
def test_slide_off(world):
    start(world, 3.0, (0.60, 0.0, 0.03), dv=(1.5, 0.0, 0.0))  # near the front edge, too fast
    term, trunc, info = world.step([0, 0, 0, 0], drop=True)
    show("slide off", term, trunc, info)
    assert info["outcome"] == "slide_off"


def test_not_still_on_slippery_pad(world):
    p.changeDynamics(world.plat.body, -1, lateralFriction=0.0, physicsClientId=world.cid)
    try:
        start(world, 2.0, (-0.65, 0.0, 0.02), dv=(0.5, 0.0, 0.0))  # frictionless: still sliding
        term, trunc, info = world.step([0, 0, 0, 0], drop=True)
        show("not still (icy pad)", term, trunc, info)
        assert info["outcome"] == "not_still"
    finally:
        p.changeDynamics(world.plat.body, -1, lateralFriction=1.0, physicsClientId=world.cid)


def test_tip_over_on_grippy_pad(world):
    """Tipping needs friction > foot half-spread / centre height (0.66). With a very
    grippy pad (1.0 x 3.0 x 0.5 -> 1.5) and a 2.5 m/s sideways mismatch it tips."""
    p.changeDynamics(world.plat.body, -1, lateralFriction=3.0, physicsClientId=world.cid)
    try:
        start(world, 2.0, (0.0, 0.0, 0.02), dv=(0.0, 2.5, 0.0))
        term, trunc, info = world.step([0, 0, 0, 0], drop=True)
        show("tip over (grippy pad)", term, trunc, info)
        assert info["outcome"] == "tip_over"
    finally:
        p.changeDynamics(world.plat.body, -1, lateralFriction=1.0, physicsClientId=world.cid)


# ---------------------------------------------------------------- crashes and flight checks
def test_drop_beside_pad_hits_ground(world):
    start(world, 0.0, (2.5, 0.0, 0.5))
    term, trunc, info = world.step([0, 0, 0, 0], drop=True)
    show("drop beside pad", term, trunc, info)
    assert info["outcome"] == "crash_ground"


def test_fly_into_platform_side(world):
    start(world, 0.0, (-1.3, 0.0, -0.45))                   # beside the box, below its top
    term, trunc, info = fly(world, [2.0, 0.0, 0.0, 0.0])
    show("into the side", term, trunc, info)
    assert info["outcome"] == "crash_platform" and term


def test_fly_into_ground(world):
    start(world, 0.0, (-4.0, 0.0, 0.5))
    term, trunc, info = fly(world, [0.0, 0.0, -1.5, 0.0])   # descend beside the pad
    show("into the ground", term, trunc, info)
    assert info["outcome"] == "crash_ground"


def test_flip(world):
    start(world, 0.0, (-3.0, 0.0, 3.0), w_body=(25.0, 0.0, 0.0))   # violent roll
    term, trunc, info = fly(world, [0.0, 0.0, 0.0, 0.0], max_steps=30)
    show("flip", term, trunc, info)
    assert info["outcome"] == "flip"


def test_lost(world):
    start(world, 0.0, (-14.0, 0.0, 3.0))
    term, trunc, info = fly(world, [-3.0, 0.0, 0.0, 0.0])    # fly away from the pad
    show("lost", term, trunc, info)
    assert info["outcome"] == "lost" and term and not trunc


def test_timeout_is_truncation(world, monkeypatch):
    monkeypatch.setattr(W, "HORIZON", 5)
    start(world, 0.0, (-3.0, 0.0, 3.0))
    term, trunc, info = fly(world, [0.0, 0.0, 0.0, 0.0])
    show("timeout", term, trunc, info)
    assert info["outcome"] == "timeout" and trunc and not term and info["t"] == 5


# ---------------------------------------------------------------- bookkeeping
def test_finished_episode_refuses_steps(world):
    start(world, 1.0, (0.0, 0.0, 0.03))
    world.step([0, 0, 0, 0], drop=True)
    with pytest.raises(AssertionError, match="reset"):
        world.step([0, 0, 0, 0])


def test_outcome_labels_known():
    assert set(W.OUTCOMES) == {"success", "hard_landing", "tip_over", "slide_off", "not_still",
                               "crash_platform", "crash_ground", "flip", "lost", "timeout"}
