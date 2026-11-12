"""v6_drop V1 (12 Nov): videos that show what happens INSIDE the drone
(scripts/replay_flight.py panels, World.step_hook, LandingEnv.last_frame).

Uses TinyRenderer and scripted flights, so it needs no results/.

Checks:
  1. World.step_hook only looks: a scripted touch landing with a hook that renders and reads
     every physics step gives exactly the same result as without it
  2. LandingEnv.last_frame is the picture the drone used: obs image = shrunk last_frame
     (also with camera grain)
  3. Recorder on a scripted touch landing: frames are 3 panels wide plus an info strip;
     slow-motion frames show the hold filling up; the state ends "cut by the TOUCH DETECTOR"
  4. with noise the frame is the same size and the flight result is unchanged by filming

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_v1_video.py -v -s
"""
import numpy as np

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.scripts import replay_flight as RF
from v6_drop.scripts import show_teacher as ST

V = 2.0
ACT = {"move": np.array([V / E.ACT_SCALE[0], 0.0, -0.2, 0.0], np.float32), "drop": 0}


def land(noise=None, hook=None, recorder=False):
    env = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal", touch="sensor", sensor_noise=noise)
    try:
        pos = [0.1, 0.0, PAD_TOP_Z + P.REST_HEIGHT + 0.12]
        env.reset(seed=1, options=dict(pad_speed=V, spawn=dict(pos=pos, yaw=0.0, v=[V, 0.0, 0.0])))
        rec = RF.Recorder(env) if recorder else None
        if rec is not None:
            env.world.frame_hook = rec.motors_off
            env.world.step_hook = rec.physics_step
        elif hook is not None:
            env.world.step_hook = lambda k: hook(env, k)
        for _ in range(40):
            if rec is not None:
                rec.before(env, ACT, 0.01)
            _, _, term, trunc, info = env.step(ACT)
            if term or trunc:
                return info, rec
        raise AssertionError("no end")
    finally:
        env.close()


def strip(info):
    td = info["touchdown"]
    return (info["outcome"], info["t"], td["reason"], td["impact"], td["detect_delay_s"], td["tilt_max_deg"])


def test_step_hook_only_looks():
    calls = []

    def look(env, k):
        env.world.render()
        env.world._contact_kind()
        env.world.own_vz()
        calls.append(k)

    a, _ = land()
    b, _ = land(hook=look)
    print(f"\n[v1] without hook {strip(a)} | with hook {strip(b)} | hook calls {len(calls)}")
    assert strip(a) == strip(b) and a["touchdown"]["reason"] == "touch_detect" and len(calls) > 50


def test_last_frame_is_what_the_drone_used():
    for noise in (None, "high"):
        env = E.LandingEnv(renderer="tiny", seed=0, gains_mode="nominal", sensor_noise=noise)
        try:
            obs, _ = env.reset(seed=3)
            assert np.array_equal(obs["image"], E.shrink(env.last_frame))
            obs, *_ = env.step({"move": np.zeros(4, np.float32), "drop": 0})
            assert env.last_frame.shape == (ST.CHASE_H, ST.CHASE_W)
            assert np.array_equal(obs["image"], E.shrink(env.last_frame))
        finally:
            env.close()


def test_recorder_shows_the_detector():
    info, rec = land(recorder=True)
    shapes = {p.shape for p in rec.pics}
    print(f"\n[v1] {len(rec.pics)} frames, shapes {shapes}, state {rec.state}, delay {info['touchdown']['detect_delay_s']}")
    assert shapes == {(ST.CHASE_H + RF.STRIP_H, 2 * ST.CHASE_W + RF.PANEL_W, 3)}
    assert rec.state == dict(motors="OFF", why="cut by the TOUCH DETECTOR")
    assert rec.touched and len(rec.pics) > 13 + 8                # decision frames + slow motion


def test_noise_video_same_flight():
    a, _ = land(noise="high")
    b, rec = land(noise="high", recorder=True)
    assert strip(a) == strip(b)
    assert {p.shape for p in rec.pics} == {(ST.CHASE_H + RF.STRIP_H, 2 * ST.CHASE_W + RF.PANEL_W, 3)}
