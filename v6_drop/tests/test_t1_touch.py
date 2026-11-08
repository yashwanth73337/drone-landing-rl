"""v6_drop T1 (7 Nov): the drone's own TOUCH DETECTOR replaces the simulator's perfect
auto-cut (envs/touch_detector.py, World(touch="sensor"), LandingEnv(touch="sensor")).

Checks:
  1. the rule alone: fires after exactly hold_steps steps of SEE and FEEL; any break resets;
     no camera value or still sinking -> never fires
  2. 'perfect' (the default) is the old auto-cut, bit for bit: same outcome and numbers as a
     World built without the new arguments
  3. sensor mode, gentle touch: motors go off AFTER the touch (0 < delay < 0.5 s),
     reason touch_detect, the referee's impact is used, safe, no false alarm.
     (The delay can be a bit SHORTER than the hold time: the clues may already be true
     1-2 physics steps before the feet touch, e.g. feet 0.5 cm up and sinking slowly.)
  4. a longer hold gives a longer delay (hold 10 vs 30 steps: + about 0.2 s)
  5. sensor mode, hard touch (dive): hard_landing at the touch (reason 'touch'), as before
  6. MISSED touch: no camera value -> the detector cannot fire; the drone sits on the pad
     with motors on -> timeout, info touch.missed = True
  7. FALSE ALARM: a camera value that lies 'feet at the pad' while hovering 30 cm up ->
     motors off in the air, false_alarm = True, the fall is judged (hard landing)
  8. the drop path is unchanged in sensor mode (exact free-fall impact as in C1)
  9. LandingEnv with the REAL camera (TinyRenderer): a gentle descent onto a moving pad is
     detected from the camera + own speed; no false alarm; info has 'touch' only in sensor
     mode; headless gives the detector no camera value (after 0.5 s nothing is known)
 T2 (short-gap prediction, 7 Nov):
 10. the estimate = last camera value + integral of the own vertical speed; a frame with no
     confident value keeps the prediction; after MAX_GAP_STEPS (0.5 s) it is unknown
 11. world: the camera goes blind 5 cm above the pad -> the prediction still detects the
     touch (cam_age_s > 0.1); blind from 60 cm -> gap too long -> missed touch
 12. LandingEnv, REAL camera: landing 0.45 m BEHIND the pad centre, where the low camera
     sees no full marker below 8 cm feet height -> detected by the prediction
 13. labels: false_alarm only with no touch before; fired_in_hop is reported apart
World tests 3-8 feed the detector a STAND-IN camera value (the true feet height, set once
per decision like a real frame), to test the world logic alone; test 9 uses the camera.

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_t1_touch.py -v -s
"""
import numpy as np
import pytest

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs import world as W
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.envs.touch_detector import TouchDetector, HOLD_STEPS, MAX_GAP_STEPS


# ---------------------------------------------------------------- helpers
def make(touch=None, hold=None):
    kw = {} if touch is None else dict(touch=touch)
    if hold is not None:
        kw["touch_cfg"] = dict(hold_steps=hold)
    return W.World(renderer="tiny", seed=3, gains_mode="nominal", **kw)


def start(w, pad_speed, feet, dv=(0.0, 0.0, 0.0)):
    pos = np.array([0.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + feet])
    v = np.array([pad_speed, 0.0, 0.0]) + np.array(dv)
    w.reset(pad_speed=pad_speed, spawn=dict(pos=pos, yaw=0.0, v=v))


def true_feet(w):
    return w._lowest_z() - PAD_TOP_Z


def fly(w, cmd, camera="true", max_steps=W.HORIZON + 5, drop_at=None):
    """camera: 'true' = stand-in frame (true feet height) each decision; None = no frames;
    a number = a fixed (possibly lying) camera value; a function f(w) -> value or None."""
    for k in range(max_steps):
        if callable(camera):
            w.set_camera_feet(camera(w))
        elif camera == "true":
            w.set_camera_feet(true_feet(w))
        elif camera is not None:
            w.set_camera_feet(float(camera))
        else:
            w.set_camera_feet(None)
        term, trunc, info = w.step(cmd, drop=(drop_at is not None and k == drop_at))
        if term or trunc:
            return term, trunc, info
    raise AssertionError("episode did not end")


def show(tag, info):
    td = info.get("touchdown") or {}
    tc = info.get("touch") or {}
    print(f"\n[{tag}] outcome={info['outcome']} t={info['t']} reason={td.get('reason')} "
          f"impact={td.get('impact')} delay={td.get('detect_delay_s')} false_alarm={td.get('false_alarm')} "
          f"touches={tc.get('touches')} missed={tc.get('missed')} tilt_max={td.get('tilt_max_deg')} "
          f"slide={td.get('slide')}")


SOFT = [2.0, 0.0, -0.4, 0.0]          # match a 2 m/s pad, descend gently (as test_c1)


# ---------------------------------------------------------------- 1. the rule alone
def test_rule_fires_after_hold_steps():
    d = TouchDetector(hold_steps=5)
    d.set_camera(0.002)
    fired = [d.update(0.0) for _ in range(7)]
    assert fired == [False] * 4 + [True] * 3
    d.reset()
    d.set_camera(0.002)
    for _ in range(4):
        assert not d.update(0.0)
    assert not d.update(-0.2)                 # still sinking: count back to 0
    assert [d.update(0.0) for _ in range(5)] == [False] * 4 + [True]


def test_rule_needs_both_clues():
    d = TouchDetector()
    d.set_camera(None)                        # no confident frame
    assert not any(d.update(0.0) for _ in range(50))
    d.set_camera(0.05)                        # 5 cm up
    assert not any(d.update(0.0) for _ in range(50))
    d.set_camera(0.0)
    assert not any(d.update(-0.1) for _ in range(50))    # sinking 0.1 m/s
    assert d.config() == dict(feet_max=0.01, vz_max=0.05, hold_steps=HOLD_STEPS,
                              max_gap_steps=MAX_GAP_STEPS)


# ---------------------------------------------------------------- 2. old mode unchanged
def test_perfect_is_the_old_auto_cut():
    a, b = make(), make("perfect")
    try:
        out = []
        for w in (a, b):
            start(w, 2.0, 0.25)
            out.append(fly(w, SOFT, camera=None)[2])
        ia, ib = out
        show("old default", ia)
        assert ia["outcome"] == ib["outcome"] == "success"
        assert ia["touchdown"]["reason"] == "auto_cut"
        assert ia["t"] == ib["t"] and ia["touchdown"] == ib["touchdown"]
        assert "touch" not in ia and "touch" not in ib          # no new keys in the old mode
        assert "cut_s" not in ia["touchdown"]
    finally:
        a.close()
        b.close()


# ---------------------------------------------------------------- 3-8. sensor mode (world)
@pytest.fixture(scope="module")
def ws():
    w = make("sensor")
    yield w
    w.close()


def test_sensor_gentle_touch(ws):
    start(ws, 2.0, 0.25)
    term, trunc, info = fly(ws, SOFT)
    show("sensor gentle", info)
    td, tc = info["touchdown"], info["touch"]
    assert term and info["outcome"] == "success"
    assert td["reason"] == "touch_detect" and not td["false_alarm"] and td["touched_before_cut"]
    assert td["fired_in_hop"] is False and td["cam_age_s"] <= 0.1 + 1e-9    # fresh frames each 0.1 s
    assert 0.0 < td["detect_delay_s"] < 0.5
    assert td["impact_method"] == "step_reading" and td["impact"] == pytest.approx(tc["impact_max"])
    assert td["impact"] < 0.6 and tc["touches"] >= 1 and not tc["missed"]


def test_longer_hold_longer_delay():
    out = {}
    for hold in (10, 30):
        w = make("sensor", hold=hold)
        try:
            start(w, 2.0, 0.25)
            info = fly(w, SOFT)[2]
            show(f"hold {hold}", info)
            assert info["outcome"] == "success" and info["touchdown"]["reason"] == "touch_detect"
            out[hold] = info["touchdown"]["detect_delay_s"]
        finally:
            w.close()
    assert out[30] - out[10] == pytest.approx(0.2, abs=0.05)


def test_sensor_hard_touch(ws):
    start(ws, 2.0, 1.0)
    info = fly(ws, [2.0, 0.0, -2.5, 0.0])[2]                 # dive onto the pad
    show("sensor dive", info)
    assert info["outcome"] == "hard_landing" and info["touchdown"]["reason"] == "touch"
    assert info["touchdown"]["impact"] > W.V_IMPACT_MAX


def test_missed_touch_without_camera(ws):
    start(ws, 2.0, 0.25)
    term, trunc, info = fly(ws, SOFT, camera=None)
    show("missed", info)
    assert trunc and info["outcome"] == "timeout"
    assert info["touch"]["touches"] >= 1 and info["touch"]["missed"] is True


def test_false_alarm_in_the_air(ws):
    start(ws, 2.0, 0.30)                                      # hover 30 cm above the pad
    info = fly(ws, [2.0, 0.0, 0.0, 0.0], camera=0.0)[2]       # the 'camera' lies: feet at the pad
    show("false alarm", info)
    td = info["touchdown"]
    assert td["reason"] == "touch_detect" and td["false_alarm"] is True and td["fired_in_hop"] is False
    assert not td["touched_before_cut"] and td["detect_delay_s"] is None
    assert td["impact_method"] == "free_fall_exact"
    assert info["outcome"] == "hard_landing" and td["impact"] > 2.0     # ~sqrt(2 g 0.3) = 2.4


def test_drop_unchanged_in_sensor_mode(ws):
    start(ws, 3.0, 0.03)
    info = fly(ws, [3.0, 0.0, 0.0, 0.0], drop_at=0)[2]
    show("sensor drop", info)
    td = info["touchdown"]
    assert info["outcome"] == "success" and td["reason"] == "drop"
    assert td["impact_method"] == "free_fall_exact"
    assert td["impact"] == pytest.approx(np.sqrt(2 * P.G * 0.03), abs=0.005)
    assert not td["false_alarm"] and not td["touched_before_cut"]


# ---------------------------------------------------------------- 9. real camera (LandingEnv)
@pytest.fixture(scope="module")
def env_s():
    e = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal", touch="sensor")
    yield e
    e.close()


def test_env_camera_detects_touch(env_s):
    v = 2.0
    pos = [0.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + 0.05]
    env_s.reset(seed=1, options=dict(pad_speed=v, spawn=dict(pos=pos, yaw=0.0, v=[v, 0.0, 0.0])))
    act = {"move": np.array([v / E.ACT_SCALE[0], 0.0, -0.2, 0.0], np.float32), "drop": 0}
    for _ in range(60):
        _, _, term, trunc, info = env_s.step(act)
        if term or trunc:
            break
    show("env camera", info)
    td = info["touchdown"]
    assert info["outcome"] == "success" and td["reason"] == "touch_detect"
    assert not td["false_alarm"] and td["touched_before_cut"]
    assert 0.0 < td["detect_delay_s"] < 0.5


def test_env_hover_no_false_alarm(env_s):
    v = 3.0
    pos = [0.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + 0.04]            # hover 4 cm up for 3 s
    env_s.reset(seed=1, options=dict(pad_speed=v, spawn=dict(pos=pos, yaw=0.0, v=[v, 0.0, 0.0])))
    act = {"move": np.array([v / E.ACT_SCALE[0], 0.0, 0.0, 0.0], np.float32), "drop": 0}
    for _ in range(30):
        _, _, term, trunc, info = env_s.step(act)
        assert not (term or trunc), info
    feet = env_s.cam_feet(env_s.last_meas)
    print(f"\n[env hover] camera feet {feet:.4f} m after 3 s, detector count {env_s.world.detector.count}")
    assert feet is not None and feet > 0.02 and env_s.world.detector.count == 0


def test_env_modes_and_headless():
    e = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal")
    try:
        assert e.world.touch == "perfect" and e.world.detector is None
    finally:
        e.close()
    h = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal", touch="sensor", headless=True)
    try:
        pos = [0.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + 0.05]
        h.reset(seed=1, options=dict(pad_speed=0.0, spawn=dict(pos=pos, yaw=0.0)))
        for _ in range(6):                                          # 60 physics steps > 0.5 s
            h.step({"move": np.zeros(4, np.float32), "drop": 0})
        assert h.world.detector.feet_est() is None                 # no frames -> nothing known
    finally:
        h.close()


# ---------------------------------------------------------------- T2: short-gap prediction
def test_prediction_alone():
    d = TouchDetector()
    assert d.feet_est() is None
    d.set_camera(0.03)
    for _ in range(10):
        d.update(-0.2)                                    # sinks 0.2 m/s for 0.1 s = 2 cm
    assert d.feet_est() == pytest.approx(0.01, abs=1e-9)
    d.set_camera(None)                                    # blind frame: keep predicting
    assert d.feet_est() == pytest.approx(0.01, abs=1e-9) and d.age == 10
    for _ in range(MAX_GAP_STEPS - 10):
        d.update(0.0)
    assert d.feet_est() == pytest.approx(0.01, abs=1e-9)  # exactly 0.5 s: still known
    d.update(0.0)
    assert d.feet_est() is None                           # longer: unknown
    d.set_camera(0.002)                                   # a new frame restarts it
    assert d.feet_est() == pytest.approx(0.002) and d.age == 0


def test_world_blind_last_5cm_is_detected(ws):
    start(ws, 2.0, 0.25)
    cam = lambda w: true_feet(w) if true_feet(w) > 0.05 else None    # blind below 5 cm
    info = fly(ws, SOFT, camera=cam)[2]
    show("blind last 5 cm", info)
    td = info["touchdown"]
    assert info["outcome"] == "success" and td["reason"] == "touch_detect"
    assert td["cam_age_s"] > 0.1 and not td["false_alarm"] and 0.0 < td["detect_delay_s"] < 0.5


def test_world_blind_too_long_is_missed(ws):
    start(ws, 2.0, 0.60)
    cam = lambda w: true_feet(w) if true_feet(w) > 0.55 else None    # blind for > 0.5 s
    term, trunc, info = fly(ws, SOFT, camera=cam)
    show("blind from 55 cm", info)
    assert trunc and info["outcome"] == "timeout" and info["touch"]["missed"] is True


def test_env_landing_behind_centre(env_s):
    # body centre 0.45 m behind the centre: the camera sees a marker from feet >= 8 cm,
    # below that it is blind (sandbox, TinyRenderer); start 12 cm up, sink like the student
    v, x0 = 1.0, -0.45
    pos = [x0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + 0.12]
    env_s.reset(seed=1, options=dict(pad_speed=v, spawn=dict(pos=pos, yaw=0.0, v=[v, 0.0, 0.0])))
    act = {"move": np.array([v / E.ACT_SCALE[0], 0.0, -0.3, 0.0], np.float32), "drop": 0}
    blind = 0
    for _ in range(60):
        _, _, term, trunc, info = env_s.step(act)
        m = env_s.last_meas
        blind += m is not None and not (m.ok and m.confident)
        if term or trunc:
            break
    show("env behind centre", info)
    td = info["touchdown"]
    print(f"  blind frames before the end: {blind}")
    assert blind >= 1                                     # the camera really was blind
    assert info["outcome"] == "success" and td["reason"] == "touch_detect"
    assert td["cam_age_s"] > 0.1 and not td["false_alarm"]
