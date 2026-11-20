"""v7 S3a tests (20 Nov 2026): fixed settings, V7 troubles, demo recorder.

  1. v7.config: the fixed settings reach the env (60 deg since 20 Nov 22:00, 3 m/s^2, 9 m/s, 60 s); the
     student rules add the shield 0.8 and the touch detector
  2. troubles use the V7 speed scale: a 1.5 to 2 m/s push is 1.5 to 2 m/s with +-10 m/s
  3. troubles use the S500's TRUE lowest point: a drone leaning 15 deg with its centre-formula
     height 0.42 m (above the 0.4 m trigger) has a skid end below 0.4 m -> bad_approach fires
  4. one recorded attempt: labels are the teacher's, the drop is never disturbed, the flown
     action differs only on trouble steps, the same attempt twice gives identical data
  5. the V6 dataset builder reads the V7 .npz files unchanged (clips of 16 + 32 steps)
One EGL env for the module.
Run:  python -m pytest -q -s v7/tests/test_s3a_demos.py
"""
import numpy as np
import pybullet as p
import pytest

from v7 import config as C
from v7.envs.platform_v7 import PAD_TOP_Z
from v7.envs.s500 import REST_HEIGHT
from v7.experts.teacher_v7 import TeacherV7
from v7.experts.troubles_v7 import CODE, TroublesV7
from v7.scripts import record_demos_v7 as R


@pytest.fixture(scope="module")
def env():
    e = C.make_env(renderer="egl", seed=0)
    yield e
    e.close()


# ---------------------------------------------------------------- 1. fixed settings
def test_fixed_settings(env):
    assert (C.PITCH_DEG, C.RATE_XY, C.V_MAX, C.HORIZON) == (60.0, 0.3, 9.0, 600)   # 60 since 20 Nov 22:00
    assert env.world.cam.pitch_deg == C.PITCH_DEG and env.rate_lim[0] == 0.3 and env.v_max == 9.0
    assert env.drop_shield is None and env.world.touch == "perfect"        # demo recording
    e2 = C.make_env(student_rules=True, renderer="tiny", seed=0)
    try:
        assert e2.drop_shield == 0.8 and e2.world.touch == "sensor" and e2.world.cam.pitch_deg == C.PITCH_DEG
    finally:
        e2.close()


def _low_spawn(env, centre_feet, lean_deg=0.0):
    spawn = dict(pos=[0.0, 0.0, PAD_TOP_Z + REST_HEIGHT + centre_feet], yaw=0.0)
    env.reset(seed=500000, options=dict(mode="straight", pad_speed=0.0, spawn=spawn))
    if lean_deg:
        q = env.world.quad
        pos, _ = p.getBasePositionAndOrientation(q.body, physicsClientId=env.world.cid)
        p.resetBasePositionAndOrientation(q.body, pos, p.getQuaternionFromEuler(
            [0.0, np.radians(lean_deg), 0.0]), physicsClientId=env.world.cid)


# ---------------------------------------------------------------- 2. speed scale
def test_push_is_in_m_per_s(env):
    _low_spawn(env, 0.30)
    tr = TroublesV7()
    tr.reset("bad_approach", np.random.default_rng(3))
    name, speed, _ = tr.payload
    label = dict(move=np.zeros(4, np.float32), drop=0)
    flown, flag = tr.apply(env, label)
    dv = (flown["move"] - label["move"])[:2] * env.act_scale[:2]
    print(f"\n[push] {name} {speed:.2f} m/s -> flown change {np.round(dv, 3)} m/s")
    assert flag == "PUSH-LOW" and np.linalg.norm(dv) == pytest.approx(speed, abs=1e-5)
    assert flown["drop"] == 0
    env.world.done = True


# ---------------------------------------------------------------- 3. lowest point
@pytest.mark.parametrize("lean, fires", [(0.0, False), (15.0, True)])
def test_trigger_uses_lowest_point(env, lean, fires):
    _low_spawn(env, 0.42, lean)
    tr = TroublesV7()
    tr.reset("bad_approach", np.random.default_rng(3))
    _, flag = tr.apply(env, dict(move=np.zeros(4, np.float32), drop=0))
    assert (flag == "PUSH-LOW") == fires
    env.world.done = True


# ---------------------------------------------------------------- 4. one attempt
@pytest.fixture(scope="module")
def two_runs(env):
    t, tr = TeacherV7(), TroublesV7()
    e = 1                                            # wobble: every step disturbed
    a = R.fly_and_record(env, t, tr, e)
    b = R.fly_and_record(env, t, tr, e)
    return a, b


def test_attempt_record(two_runs):
    (rec, row), _ = two_runs
    print(f"\n[attempt] {row}")
    n = row["steps"]
    assert row["kind"] == "wobble" and row["seed"] == 500001 and row["kept"] == 1
    assert all(len(rec[k]) == n for k in R.KEYS)
    assert rec["image"][0].shape == (160, 256) and rec["state"][0].shape == (11,)
    move, flown = np.array(rec["move"]), np.array(rec["flown"])
    flag = np.array(rec["flag"])
    assert np.all(flag == CODE["wobble"])
    assert np.abs(flown - move).max() > 0.01                    # the flown action was disturbed
    assert sum(rec["drop"]) >= 1 and rec["drop"][-1] == 1        # the teacher's drop is the last step
    assert set(rec["phase"]) <= set(range(len(R.PHASES)))
    assert rec["t"] == list(range(n))


def test_attempt_reproducible(two_runs):
    (ra, wa), (rb, wb) = two_runs
    assert wa == wb
    for k in R.KEYS:
        assert np.array_equal(np.array(ra[k]), np.array(rb[k])), k


# ---------------------------------------------------------------- 5. V6 dataset reads it
def test_v6_dataset_reads_v7_demos(two_runs, tmp_path):
    from v6_drop.learn.dataset import BURN, CLIP, Demos, batch, build
    from v6_drop.scripts.record_demos import save_chunk
    (rec, row), _ = two_runs
    buf = {k: list(rec[k]) for k in R.KEYS}
    buf["ep"] = [row["ep"]] * row["steps"]
    src, dst = tmp_path / "demos", tmp_path / "data"
    src.mkdir()
    save_chunk(buf, str(src / "demos_w0_0_000.npz"))
    n, m = build(str(src), str(dst))
    d = Demos(str(dst))
    assert (n, m) == (row["steps"], 1)
    b = batch(d, [(0, 0)])                            # episode INDEX 0, clip start 0
    assert b["image"].shape == (BURN + CLIP, 1, 160, 256) and b["mask"].sum() == CLIP
