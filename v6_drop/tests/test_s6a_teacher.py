"""v6_drop Step 6a: the privileged teacher.

Checks:
  1. 40 fixed test episodes (headless): all safe landings, all by DROP (not auto-cut),
     impact <= 0.95 m/s, landed <= 6 cm from the centre, within 20 s
  2. information gap: the teacher's pad-velocity estimate starts at zero and follows a
     first-order filter (time constant 0.5 s) towards the true value
  3. the drop gate must hold for HOLD_STEPS decisions
  4. go-around: started low and well behind the pad, it climbs ("go-around") and still
     lands safely
  5. with the camera (rendering): during glide and hover a marker is seen >= 95% of steps
  6. the same seed gives the same episode even after a different episode (B5 fix)
  7. show_teacher batch mode writes its table and figure

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s6a_teacher.py -v -s
"""
import os

import numpy as np
import pytest

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs import world as W
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.experts import teacher as TM


def _egl_ok():
    try:
        w = W.World(renderer="egl")
        w.close()
        return True
    except Exception:
        return False


RENDERER = "egl" if _egl_ok() else "tiny"


def fly(env, teacher, seed=None, options=None):
    env.reset(seed=seed, options=options)
    teacher.reset()
    phases, infos = [], []
    while True:
        act, ti = teacher.act(env)
        phases.append(ti["phase"])
        _, _, term, trunc, info = env.step(act)
        infos.append(info)
        if term or trunc:
            return info, phases, infos


@pytest.fixture(scope="module")
def headless():
    e = E.LandingEnv(renderer="tiny", seed=0, headless=True)
    yield e
    e.close()


# ---------------------------------------------------------------- 1. 40 test episodes
def test_teacher_lands_every_test_episode(headless):
    t = TM.Teacher()
    res = []
    for i in range(40):
        info, phases, _ = fly(headless, t, seed=9000 + i)
        td = info.get("touchdown") or {}
        res.append((info["outcome"], td.get("reason"), td.get("impact"),
                    None if not td.get("land_xy") else float(np.linalg.norm(td["land_xy"])),
                    info["t"] * P.POLICY_DT, headless.world.plat.v))
    outs = [r[0] for r in res]
    imp = np.array([r[2] for r in res])
    off = np.array([r[3] for r in res])
    tim = np.array([r[4] for r in res])
    print(f"\n[teacher 40] safe {outs.count('success')}/40, drops {sum(r[1] == 'drop' for r in res)}; "
          f"impact mean {imp.mean():.2f} max {imp.max():.2f}; offset mean {100 * off.mean():.1f} "
          f"max {100 * off.max():.1f} cm; time mean {tim.mean():.1f} max {tim.max():.1f} s")
    assert outs.count("success") == 40
    assert all(r[1] == "drop" for r in res)
    assert imp.max() <= 0.95 and off.max() <= 0.06 and tim.max() <= 20.0


# ---------------------------------------------------------------- 2. information gap
def test_pad_velocity_is_learned_gradually(headless):
    headless.reset(options=dict(pad_speed=4.0, spawn=dict(pos=[-3.0, 0.0, PAD_TOP_Z + 3.0], yaw=0.0)))
    t = TM.Teacher()
    est = []
    for _ in range(10):
        act, _ = t.act(headless)
        est.append(t.pad_v_est[0])
        headless.step(act)
    print(f"\n[pad speed estimate] first steps: {np.round(est[:6], 2).tolist()} ... (true 4.0)")
    a = P.POLICY_DT / (TM.PAD_V_TAU + P.POLICY_DT)
    expect = [4.0 * (1 - (1 - a) ** (k + 1)) for k in range(10)]   # first-order filter
    assert est[0] < 1.0                                            # not known at once
    assert np.allclose(est, expect, atol=1e-9)                     # 63% after ~0.5 s


# ---------------------------------------------------------------- 3. drop gate
def test_drop_needs_gate_held():
    assert TM.HOLD_STEPS >= 2
    assert np.sqrt(2 * P.G * (TM.H_DROP + TM.DH_DROP)) < 0.9   # worst drop height < 0.9 m/s


# ---------------------------------------------------------------- 4. go-around
def test_go_around_when_low_and_off_line(headless):
    spawn = dict(pos=[-1.6, 0.4, PAD_TOP_Z + P.REST_HEIGHT + 0.12], yaw=0.0, v=[1.0, 0.0, 0.0])
    info, phases, _ = fly(headless, TM.Teacher(), options=dict(pad_speed=1.0, spawn=spawn))
    print(f"\n[go-around] phases used: {sorted(set(phases))}; outcome {info['outcome']}")
    assert "go-around" in phases and info["outcome"] == "success"


# ---------------------------------------------------------------- 5. visibility (camera)
def test_marker_seen_during_glide_and_hover():
    env = E.LandingEnv(renderer=RENDERER, seed=0)
    try:
        seen = []
        for i in range(5):
            info, phases, infos = fly(env, TM.Teacher(), seed=9000 + i)
            for ph, inf in zip(phases, infos):
                rt = inf.get("reward_terms") or {}
                if ph in ("glide", "hover") and rt.get("seen") is not None:
                    seen.append(rt["seen"])
        frac = float(np.mean(seen))
        print(f"\n[visibility {RENDERER}] marker seen in {100 * frac:.1f}% of glide/hover steps "
              f"({len(seen)} steps)")
        assert frac >= 0.95
    finally:
        env.close()


# ---------------------------------------------------------------- 6. reproducible seeds
def test_seed_reproducible_after_other_episode(headless):
    t = TM.Teacher()
    a, _, _ = fly(headless, t, seed=9003)
    fly(headless, t, seed=9000)
    b, _, _ = fly(headless, t, seed=9003)
    assert a["t"] == b["t"] and a["touchdown"]["impact"] == b["touchdown"]["impact"]


# ---------------------------------------------------------------- 7. batch report
def test_batch_report_files(tmp_path, monkeypatch):
    from v6_drop.scripts import show_teacher as ST
    monkeypatch.setattr(ST, "OUT", str(tmp_path))
    original = E.LandingEnv                        # same module object as ST.E
    monkeypatch.setattr(ST.E, "LandingEnv",
                        lambda renderer, seed, headless: original("tiny", seed, headless=headless))
    ST.mode_batch(3)
    for f in ("batch.csv", "batch_summary.txt", "batch_paths.png"):
        assert os.path.exists(tmp_path / f)
