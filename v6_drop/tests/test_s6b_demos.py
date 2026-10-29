"""v6_drop Step 6b: troubles (DART) and the demonstration recorder.

Checks:
  1. the kind is chosen from the seed (reproducible) with the agreed mix 25/25/20/15/15
  2. clean: the flown action IS the label
  3. wobble: flown differs from the label by a smooth, bounded error that shrinks near the pad
  4. gust: a sideways force acts ONLY during the planned gust steps
  5. pad_lost: one push, only after 1 s, while high and seeing the pad; it lasts its duration
  6. bad_approach: one push, the first time the feet go below 0.4 m
  7. the teacher recovers: >= 90% safe for every kind (20 headless episodes each)
  8. regression: the teacher no longer turns wobble noise into drift (seed 100004 was lost)
  9. recorder: samples are saved BEFORE each decision, the label is the teacher's action,
     a saved chunk loads back unchanged, and the same attempt records identical data

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s6b_demos.py -v -s
"""
import collections

import numpy as np
import pytest

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs.platform_s1 import PAD_TOP_Z
from v6_drop.experts import troubles as TR
from v6_drop.experts.teacher import Teacher
from v6_drop.scripts import record_demos as RD


@pytest.fixture(scope="module")
def env():
    e = E.LandingEnv(renderer="tiny", seed=0, headless=True)
    yield e
    e.close()


def fly(env, kind, seed, max_steps=400, seen=True):
    """Returns per-step (label move, flown move, flag, ext_force, feet height, k) + info."""
    env.reset(seed=seed)
    t, tr = Teacher(), TR.Troubles()
    rng = np.random.default_rng(seed)
    TR.pick_kind(rng)
    tr.reset(kind, rng)
    rows, phases = [], set()
    for k in range(max_steps):
        act, ti = t.act(env)
        flown, flag = tr.apply(env, act, seen)
        phases.add(ti["phase"])
        rows.append((np.array(act["move"]), np.array(flown["move"]), flag,
                     env.world.quad.ext_force.copy(), ti["hf"], k))
        _, _, term, trunc, info = env.step(flown)
        if term or trunc:
            return rows, info, phases
    raise AssertionError("did not end")


# ---------------------------------------------------------------- 1. kind and mix
def test_kind_from_seed_and_mix():
    kinds = [TR.pick_kind(np.random.default_rng(100000 + e)) for e in range(4000)]
    again = [TR.pick_kind(np.random.default_rng(100000 + e)) for e in range(50)]
    assert kinds[:50] == again
    c = collections.Counter(kinds)
    share = {k: c[k] / len(kinds) for k in TR.KINDS}
    print(f"\n[mix] {', '.join(f'{k} {100 * v:.1f}%' for k, v in share.items())}")
    for k, m in zip(TR.KINDS, TR.MIX):
        assert abs(share[k] - m) < 0.025


# ---------------------------------------------------------------- 2-6. each trouble
def test_clean_flown_equals_label(env):
    rows, info, _ = fly(env, "clean", 100003)
    assert all(np.array_equal(a, b) and f == "" for a, b, f, *_ in rows)
    assert all(np.all(fc == 0) for *_, fc, _h, _k in rows)


def test_wobble_bounded_smooth_and_smaller_near_pad(env):
    rows, info, _ = fly(env, "wobble", 100009)
    err = np.array([(b - a) * E.ACT_SCALE for a, b, *_ in rows])[:, :3]
    hf = np.array([r[4] for r in rows])
    high, low = np.abs(err[hf > 0.6]), np.abs(err[hf < 0.1])
    step_change = np.abs(np.diff(err, axis=0)).mean(0)
    print(f"\n[wobble] |error| high: {np.round(high.mean(0), 2)} m/s, near pad: "
          f"{np.round(low.mean(0), 2)} m/s; mean change per step {np.round(step_change, 2)}")
    assert np.all(np.abs(err) < 4 * TR.WOBBLE_SIGMA[:3] + 1e-6)
    assert high[:, 0].mean() > 2 * low[:, 0].mean()
    assert np.all(step_change < high.mean(0))            # smooth: changes slower than its size


def test_gust_force_only_during_events(env):
    rows, info, _ = fly(env, "gust", 100005)
    on = [r for r in rows if r[2] == "GUST"]
    off = [r for r in rows if r[2] != "GUST"]
    mags = [np.linalg.norm(r[3]) for r in on]
    print(f"\n[gust] {len(on)} gust steps, force {min(mags):.2f}..{max(mags):.2f} N; outcome {info['outcome']}")
    assert on and all(TR.GUST_FORCE[0] - 1e-9 <= m <= TR.GUST_FORCE[1] + 1e-9 for m in mags)
    assert all(np.all(r[3] == 0) for r in off)
    assert all(np.array_equal(r[0], r[1]) for r in rows)  # gusts push the drone, not the command


def test_pad_lost_push_once_late_and_high(env):
    rows, info, _ = fly(env, "pad_lost", 100021)
    ks = [r[5] for r in rows if r[2] == "PUSH-AWAY"]
    print(f"\n[pad_lost] push at steps {ks[0]}..{ks[-1]} (feet {rows[ks[0]][4]:.2f} m); outcome {info['outcome']}")
    assert ks == list(range(ks[0], ks[0] + len(ks)))       # one continuous push
    assert ks[0] * P.POLICY_DT >= TR.AWAY_AFTER[0] and rows[ks[0]][4] > TR.AWAY_MIN_H
    assert TR.AWAY_DUR[0] - 0.1 <= len(ks) * P.POLICY_DT <= TR.AWAY_DUR[1] + 0.1


def test_bad_approach_push_when_low(env):
    rows, info, phases = fly(env, "bad_approach", 100001)
    ks = [r[5] for r in rows if r[2] == "PUSH-LOW"]
    print(f"\n[bad_approach] push at steps {ks[0]}..{ks[-1]} (feet {rows[ks[0]][4]:.2f} m); "
          f"go-around {'go-around' in phases}; outcome {info['outcome']}")
    assert ks == list(range(ks[0], ks[0] + len(ks)))
    assert rows[ks[0]][4] < TR.LOW_H and all(r[4] >= TR.LOW_H for r in rows[:ks[0]])


# ---------------------------------------------------------------- 7. recovery
@pytest.mark.parametrize("kind", TR.KINDS)
def test_teacher_recovers(env, kind):
    ok = 0
    for e in range(20):
        _, info, _ = fly(env, kind, 200000 + e)
        ok += info["outcome"] == "success"
    print(f"\n[recover] {kind}: {ok}/20 safe")
    assert ok >= 18


# ---------------------------------------------------------------- 8. no drift (regression)
def test_wobble_does_not_drift_away(env):
    _, info, _ = fly(env, "wobble", 100004)
    assert info["outcome"] != "lost"


# ---------------------------------------------------------------- 9. recorder
def test_recorder_data(tmp_path):
    env = E.LandingEnv(renderer="tiny", seed=0)
    try:
        t, tr = Teacher(), TR.Troubles()
        rec, row = RD.fly_and_record(env, t, tr, 1)               # attempt 1 = bad_approach
        rec2, row2 = RD.fly_and_record(env, t, tr, 1)
    finally:
        env.close()
    n = row["steps"]
    print(f"\n[recorder] attempt 1: kind {row['kind']}, {n} samples, outcome {row['outcome']}, "
          f"trouble steps {row['trouble_steps']}")
    assert len(rec["image"]) == n and rec["image"][0].shape == (160, 256)
    assert rec["t"] == list(range(n))                              # saved BEFORE each decision
    assert sum(rec["drop"]) == 1 and rec["drop"][-1] == 1          # teacher dropped once, at the end
    assert any(not np.array_equal(a, b) for a, b in zip(rec["move"], rec["flown"]))
    assert all(np.array_equal(a, b) for a, b in zip(rec["move"], rec2["move"]))   # reproducible
    assert all(np.array_equal(a, b) for a, b in zip(rec["image"], rec2["image"]))
    buf = dict(rec, ep=[1] * n)
    RD.save_chunk(buf, str(tmp_path / "c.npz"))
    d = np.load(tmp_path / "c.npz")
    assert d["image"].shape == (n, 160, 256) and d["image"].dtype == np.uint8
    assert np.array_equal(d["move"], np.stack(rec["move"])) and d["drop"].sum() == 1
    assert set(d.files) == {"image", "state", "priv", "move", "drop", "flown", "flag", "phase", "ep", "t"}
