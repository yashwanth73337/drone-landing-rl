"""v7 S4g tests (25 Nov 2026): the gimbal TILTS with height (--gimbal-schedule h3to6).

  1. the schedule: 60 deg at or below 3 m, 45 deg at or above 6 m, linear between
  2. set_pitch() turns the camera axes (forward vector) to the new angle
  3. paired starts: with and without the schedule every seed starts at EXACTLY the same place
     with the same pad (the start is chosen at the base pitch 60)
  4. below 3 m the scheduled picture is pixel-identical to the fixed 60 deg picture
  5. far view: drone 15 m behind a standing pad, skids 6 m above it: the pad centre is in the
     picture with the schedule (45 deg) and NOT at a fixed 60 deg; at 10 m behind the markers
     are read with the schedule (error < 0.6 m) and not at 60 deg
  6. marker reading at a tilted angle (4.5 m high, 52.5 deg) is as accurate as at 60 deg
  7. the default (no schedule) env never changes the pitch during a flight
  8. the switches pass through: eval_student_v7 --gimbal-schedule (one flight; output name
     ends _gh3to6) and the PPO env (make_env_ppo)
One EGL world open at a time (V6 B5 rule): each test opens and closes its own.
Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -s v7/tests/test_s4g_gimbal.py
"""
import json

import numpy as np
import pytest
import torch

from v6_drop.envs.world import euler_R
from v6_drop.learn.student import Student
from v7.config import make_env
from v7.envs import world_v7 as W7
from v7.envs.gimbal_camera import GimbalCamera

SCHED = W7.GIMBAL_SCHEDULES["h3to6"]


def env_of(schedule):
    return make_env(student_rules=True, renderer="egl", seed=7, gimbal_schedule=schedule)


# ---------------------------------------------------------------- 1, 2
def test_schedule_values():
    f = lambda h: W7.scheduled_pitch(SCHED, h)
    assert f(0.0) == 60.0 and f(3.0) == 60.0 and f(4.5) == pytest.approx(52.5)
    assert f(6.0) == 45.0 and f(9.0) == 45.0
    assert W7.GIMBAL_SCHEDULES["none"] is None


def test_set_pitch_axes():
    cam = GimbalCamera.__new__(GimbalCamera)        # axes only, no renderer needed
    cam.set_pitch(45.0)
    fwd, up = cam.world_axes(np.eye(3))
    assert np.degrees(np.arcsin(-fwd[2])) == pytest.approx(45.0)
    assert np.dot(fwd, up) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- 3, 4, 7
def starts(schedule, seeds):
    env = env_of(schedule)
    out = []
    try:
        for s in seeds:
            obs, info0 = env.reset(seed=s, options=dict(mode="random"))
            ts = env.world.true_state()
            pitches = [env.world.cam.pitch_deg]
            for _ in range(5):
                env.step(dict(move=np.array([0, 0, 1.0, 0], np.float32), drop=0))  # climb
                pitches.append(env.world.cam.pitch_deg)
            out.append(dict(pos=ts["pos"].copy(), pad=ts["pad_center"].copy(),
                            img=obs["image"].copy(), h=env.world.gimbal_height(ts["pos"]),
                            pitches=pitches))
    finally:
        env.close()
    return out


def test_paired_starts_and_low_pictures():
    seeds = list(range(30000, 30012))
    a, b = starts("none", seeds), starts("h3to6", seeds)
    n_low = 0
    for x, y in zip(a, b):
        np.testing.assert_allclose(x["pos"], y["pos"])
        np.testing.assert_allclose(x["pad"], y["pad"])
        assert set(x["pitches"]) == {60.0}                    # 7. default never tilts
        if y["h"] <= 3.0:
            n_low += 1
            assert np.array_equal(x["img"], y["img"])         # 4. identical below 3 m
            assert y["pitches"][0] == 60.0
        else:
            assert y["pitches"][0] < 60.0
    assert n_low >= 1


# ---------------------------------------------------------------- 5, 6
def test_far_view_and_accuracy():
    env = env_of("h3to6")
    w = env.world
    try:
        env.reset(seed=30000, options=dict(mode="straight", pad_speed=0.0))
        c = w.plat.pad_center()
        R = euler_R(0.0, 0.0, 0.0)

        def look(d, h, sched):
            pos = c + np.array([-d, 0.0, h + W7.REST_HEIGHT])
            w.quad.reset_pose(pos.tolist(), yaw=0.0)
            if sched:
                img = w.render()
            else:
                w.cam.set_pitch(60.0)
                img = w.cam.render(R, pos)
            m = w.measure(img)
            err = np.linalg.norm(m.p_body - R.T @ (c - pos)) if m.ok else None
            return w.pad_centre_in_view(R, pos), err, w.cam.pitch_deg

        inv_s, _, p_s = look(15.0, 6.0, True)
        inv_f, _, _ = look(15.0, 6.0, False)
        assert p_s == 45.0 and inv_s and not inv_f
        _, err_s, _ = look(10.0, 6.0, True)
        _, err_f, _ = look(10.0, 6.0, False)
        assert err_s is not None and err_s < 0.6 and err_f is None
        _, err_t, p_t = look(6.0, 4.5, True)
        _, err_60, _ = look(6.0, 4.5, False)
        print(f"\n4.5 m high, 6 m behind: error {err_t:.2f} m at {p_t:.1f} deg vs {err_60:.2f} m at 60 deg")
        assert p_t == pytest.approx(52.5) and err_t < 0.3 and err_60 < 0.3
    finally:
        env.close()


# ---------------------------------------------------------------- 8
def test_switches_pass_through(tmp_path):
    from v7.scripts import eval_student_v7 as E
    from v7.scripts import train_ppo_v7 as P7
    torch.manual_seed(0)
    ck = tmp_path / "s.pt"
    torch.save(dict(model=Student(256).state_dict(), size=256), ck)
    recs = E.main(["--ckpt", str(ck), "--n", "1", "--paths", "straight", "--workers", "1",
                   "--gimbal-schedule", "h3to6"])
    assert len(recs) == 1 and (tmp_path / "dev_s_1_gh3to6.jsonl").exists()
    assert "gimbal schedule h3to6" in (tmp_path / "dev_s_1_gh3to6.txt").read_text()
    env = P7.make_env_ppo("egl", gimbal_schedule="h3to6")
    try:
        assert env.world.gimbal_schedule_name == "h3to6"
    finally:
        env.close()
