"""v6_drop Step 5: the reward.

Checks:
  1. the potential: Phi = -max(d, h_feet) (= -(d + max(h_feet - d, 0)), 45 deg glide slope).
     Closer horizontally pays +1/m when d >= h_feet; lower pays +1/m when h_feet >= d.
  2. FAIR PROGRESS: any loop (out, up, back, down) earns exactly 0. For comparison, Shin et
     al.'s divided form (|dz| change / max(d, 1)) pays +0.75 per loop of the same shape.
  3. camera geometry: (d, h) from ArUco + own orientation match the truth
  4. no confident marker: progress pauses (last good Phi kept), -0.05; after re-detection the
     progress telescopes (the sum equals Phi_end - Phi_start)
  5. end marks for every outcome; hard landings are graded down to -10 at 6 m/s
  6. whole episodes in the env: a scripted approach + drop scores ~+12..+15 at pad speeds
     0, 2, 4 m/s; hovering until the time limit costs exactly 0.01 per step; a drop from
     1 m (hard landing) scores far below a soft landing
  7. source="true" (ablation) gives the same progress as the camera on the same episode

The scripted pilot here uses the simulator truth. It is a TEST helper, not the Step 6 expert.

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s5_reward.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.envs import pad_board as PB
from v6_drop.envs import reward as RW
from v6_drop.envs import world as W
from v6_drop.envs.lee_controller import yaw_rotation
from v6_drop.envs.platform_s1 import PAD_TOP_Z


def _egl_ok():
    try:
        w = W.World(renderer="egl")
        w.close()
        return True
    except Exception:
        return False


RENDERER = "egl" if _egl_ok() else "tiny"


@pytest.fixture(scope="module")
def env():
    e = E.LandingEnv(renderer=RENDERER, seed=3)
    yield e
    e.close()


def phi(d, h_feet):
    return RW.potential(d, h_feet + P.REST_HEIGHT)


# ---------------------------------------------------------------- 1. potential
def test_potential_values():
    assert phi(3.0, 1.0) == pytest.approx(-3.0)          # below the slope: only d counts
    assert phi(1.0, 3.0) == pytest.approx(-3.0)          # above the slope: only height counts
    assert phi(0.0, 0.0) == pytest.approx(0.0)           # feet on the pad centre
    assert phi(2.0, 1.0) - phi(3.0, 1.0) == pytest.approx(1.0)    # 1 m closer (low): +1
    assert phi(1.0, 2.0) - phi(1.0, 3.0) == pytest.approx(1.0)    # 1 m lower (high): +1
    assert phi(3.0, 1.0) - phi(3.0, 2.0) == pytest.approx(0.0)    # lower while far: 0
    assert phi(0.5, 3.0) - phi(1.0, 3.0) == pytest.approx(0.0)    # closer while high: 0


# ---------------------------------------------------------------- 2. loops earn nothing
def test_loop_earns_nothing_but_shin_form_does():
    loop = [(0.5, 2.0), (0.5, 1.0), (4.0, 1.0), (4.0, 2.0), (0.5, 2.0)]   # down, out, up, back
    ours = sum(phi(*b) - phi(*a) for a, b in zip(loop, loop[1:]))

    def shin_step(a, b):                                  # Table III, weights 1
        # Each leg is flown in many small steps, so Shin's per-step clip to [-1, 1]
        # never binds; summing the small steps gives these leg totals.
        lateral = a[0] - b[0]
        vertical = (abs(a[1]) - abs(b[1])) / max(b[0], 1.0)  # legs: d constant while z changes
        return lateral + vertical
    shin = sum(shin_step(a, b) for a, b in zip(loop, loop[1:]))
    print(f"\n[loop] ours {ours:+.3f} per loop; Shin-style divided form {shin:+.3f} per loop")
    assert ours == pytest.approx(0.0, abs=1e-12) and shin == pytest.approx(0.75)


# ---------------------------------------------------------------- 3. camera geometry
@pytest.mark.parametrize("rel,pitch,tol", [((-0.05, 0.0, 0.30), 0, 0.02),
                                           ((-1.5, 0.3, 1.5), 5, 0.10),
                                           ((-3.0, -0.5, 2.5), -5, 0.20)])
def test_camera_geometry(env, rel, pitch, tol):
    env.reset(options=dict(pad_speed=0.0, spawn=dict(pos=[rel[0], rel[1], PAD_TOP_Z + rel[2]], yaw=0.0)))
    w = env.world
    q = p.getQuaternionFromEuler([0.0, np.radians(pitch), 0.0])
    p.resetBasePositionAndOrientation(w.quad.body, [rel[0], rel[1], PAD_TOP_Z + rel[2]], q,
                                      physicsClientId=w.cid)
    s = w.true_state()
    m = PB.measure(w.render(), w.cam, w.layout, w.det)
    assert m.ok and m.confident
    d_cam, h_cam = RW.geometry_from_body(m.p_body, s["R"])
    d_true, h_true = RW.geometry_from_body(s["rel_pos_body"], s["R"])
    print(f"\n[geometry {rel}] camera d={d_cam:.3f} h={h_cam:.3f}; truth d={d_true:.3f} h={h_true:.3f}")
    assert abs(d_cam - d_true) < tol and abs(h_cam - h_true) < tol
    assert d_true == pytest.approx(np.hypot(rel[0], rel[1]), abs=1e-6)
    assert h_true == pytest.approx(rel[2], abs=1e-6)


# ---------------------------------------------------------------- 4. pause + telescoping
class _M:
    def __init__(self, d, h_feet):
        self.ok, self.confident = True, True
        self.p_body = np.array([d, 0.0, -(h_feet + P.REST_HEIGHT)])


def test_pause_when_not_seen_and_telescoping():
    truth = {"R": np.eye(3), "rel_pos_body": np.zeros(3)}
    r = RW.Reward("camera")
    seq = [_M(4.0, 2.0), _M(3.5, 2.0), None, None, _M(2.0, 1.5), _M(1.0, 1.0)]
    r.reset(seq[0], truth)
    total_prog, vis = 0.0, 0.0
    for m in seq[1:]:
        _, t = r.step(m, truth)
        total_prog += t["progress"]
        vis += t["visibility"]
        if m is None:
            assert t["progress"] == 0.0 and t["visibility"] == -RW.VISIBILITY_PENALTY
    print(f"\n[pause] progress sum {total_prog:.3f} (Phi_end - Phi_start = "
          f"{phi(1.0, 1.0) - phi(4.0, 2.0):.3f}); visibility {vis:.2f}")
    assert total_prog == pytest.approx(phi(1.0, 1.0) - phi(4.0, 2.0))
    assert vis == pytest.approx(-2 * RW.VISIBILITY_PENALTY)


def test_not_confident_counts_as_not_seen():
    m = _M(2.0, 1.0)
    m.confident = False
    assert not RW.Reward.seen(m)


# ---------------------------------------------------------------- 5. end marks
def test_end_marks():
    td = dict(land_xy=[0.0, 0.0], impact=0.0)
    assert RW.end_mark("success", td) == pytest.approx(14.0)            # centre + soft
    assert RW.end_mark("success", dict(land_xy=[0.75, 0.0], impact=1.0)) == pytest.approx(10.0)
    assert RW.end_mark("success", dict(land_xy=[0.3, 0.0], impact=0.5)) == pytest.approx(
        10 + 2 * (1 - 0.3 / 0.75) + 1.0)
    hard = [RW.end_mark("hard_landing", dict(impact=v)) for v in (1.0, 2.0, 4.0, 6.0, 9.0)]
    print(f"\n[end marks] hard landing at 1/2/4/6/9 m/s: {np.round(hard, 2).tolist()}")
    assert hard == pytest.approx([0.0, -2.0, -6.0, -10.0, -10.0])
    for o in ("slide_off", "tip_over", "not_still"):
        assert RW.end_mark(o, {}) == -3.0
    for o in ("crash_platform", "crash_ground", "flip", "lost"):
        assert RW.end_mark(o, {}) == -10.0
    assert RW.end_mark("timeout", {}) == 0.0
    assert set(W.OUTCOMES) == set(RW.END_MARKS) | {"success", "hard_landing"}


def test_drop_bonus_never_pays_less_than_auto_cut():
    """Step 7 P1 (B): for the same landing spot, any safe drop (impact 0-1 m/s) scores at
    least as much as any safe auto-cut, and a softer drop scores more than a harder one."""
    assert RW.DROP_BONUS == RW.SOFT_BONUS
    v = np.linspace(0.0, 1.0, 21)
    for xy in ([0.0, 0.0], [0.2, 0.0], [0.5, 0.3]):
        drop = np.array([RW.end_mark("success", dict(land_xy=xy, impact=x, reason="drop")) for x in v])
        cut = np.array([RW.end_mark("success", dict(land_xy=xy, impact=x, reason="auto_cut")) for x in v])
        assert drop.min() >= cut.max() - 1e-9
        assert np.all(np.diff(drop) < 0)                       # softer drop -> more
        assert np.allclose(drop - cut, RW.DROP_BONUS)
    d07 = RW.end_mark("success", dict(land_xy=[0.2, 0.0], impact=0.70, reason="drop"))
    c016 = RW.end_mark("success", dict(land_xy=[0.2, 0.0], impact=0.16, reason="auto_cut"))
    print(f"\n[drop bonus] 20 cm from centre: drop 0.70 m/s -> {d07:.2f}, auto-cut 0.16 m/s -> {c016:.2f}")
    assert d07 == pytest.approx(14.07, abs=0.01) and c016 == pytest.approx(13.15, abs=0.01)
    # the bonus is only for a SAFE drop: a hard drop is still punished, no bonus
    assert RW.end_mark("hard_landing", dict(impact=1.36, reason="drop")) == pytest.approx(-0.72)


# ---------------------------------------------------------------- 6. whole episodes
def scripted(env):
    """TEST helper using the truth: follow the pad, come down along the glide slope,
    drop when 4 cm above the centre at matched speed."""
    s = env.world.true_state()
    R = s["R"]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    rel = s["pad_center"] - s["pos"]
    d = np.hypot(rel[0], rel[1])
    hf = s["pos"][2] - PAD_TOP_Z - P.REST_HEIGHT
    vrel = np.linalg.norm((s["v"] - s["pad_v"])[:2])
    vxy = s["pad_v"][:2] + 1.0 * rel[:2]
    vz = (-0.4 if hf > 0.04 else 0.0) if d < 0.25 else float(np.clip(-(hf - max(d - 0.2, 0.0)), -0.8, 0.3))
    cmd = np.array([*(yaw_rotation(yaw).T @ np.array([*vxy, 0.0]))[:2], vz, 0.0])
    drop = int(d < 0.12 and hf < 0.045 and vrel < 0.25)
    return {"move": np.clip(cmd / E.ACT_SCALE, -1, 1).astype(np.float32), "drop": drop}


def run(env, policy, pad_speed, spawn, max_steps=400):
    env.reset(options=dict(pad_speed=pad_speed, spawn=spawn))
    total, sums = 0.0, dict(progress=0.0, visibility=0.0, time=0.0, end=0.0)
    for _ in range(max_steps):
        _, r, term, trunc, info = env.step(policy(env))
        total += r
        for k in sums:
            sums[k] += info["reward_terms"][k]
        if term or trunc:
            return total, sums, info
    raise AssertionError("episode did not end")


START = dict(pos=[-3.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + 2.5], yaw=0.0, v=[0.0, 0.0, 0.0])


@pytest.mark.parametrize("pad_speed", [0.0, 2.0, 4.0])
def test_scripted_landing_scores_high(env, pad_speed):
    total, sums, info = run(env, scripted, pad_speed, START)
    td = info["touchdown"]
    print(f"\n[episode pad {pad_speed}] outcome {info['outcome']} total {total:.2f} = progress "
          f"{sums['progress']:.2f} + visibility {sums['visibility']:.2f} + time {sums['time']:.2f} "
          f"+ end {sums['end']:.2f}  (impact {td['impact']:.2f} m/s)")
    assert info["outcome"] == "success"
    assert sums["progress"] == pytest.approx(3.0, abs=0.15)     # Phi_start = -max(3, 2.5)
    assert info["touchdown"]["reason"] == "drop"
    assert sums["end"] == pytest.approx(RW.end_mark("success", td))
    assert 13.0 < total < 18.0          # Step 7: +2 drop bonus (was 11-16)


def test_hover_until_timeout(env, monkeypatch):
    monkeypatch.setattr(W, "HORIZON", 50)
    hover = lambda e: {"move": np.zeros(4, np.float32), "drop": 0}
    total, sums, info = run(env, hover, 0.0, START)
    print(f"\n[hover] outcome {info['outcome']} total {total:.3f} over {info['t']} steps "
          f"(visibility {sums['visibility']:.2f})")
    assert info["outcome"] == "timeout"
    assert sums["time"] == pytest.approx(-0.01 * 50)
    assert abs(sums["progress"]) < 0.05 and total < 0


def test_hard_drop_scores_low(env):
    spawn = dict(pos=[0.0, 0.0, PAD_TOP_Z + P.REST_HEIGHT + 1.0], yaw=0.0, v=[2.0, 0.0, 0.0])
    drop_now = lambda e: {"move": np.array([2 / 6, 0, 0, 0], np.float32), "drop": 1}
    total, sums, info = run(env, drop_now, 2.0, spawn)
    print(f"\n[hard drop 1 m] outcome {info['outcome']} impact {info['touchdown']['impact']:.2f} "
          f"total {total:.2f}")
    assert info["outcome"] == "hard_landing" and total < -5.0


# ---------------------------------------------------------------- 7. true-state source
def test_true_source_matches_camera_progress(env):
    tot_cam, s_cam, _ = run(env, scripted, 2.0, START)
    e2 = E.LandingEnv(renderer="tiny", seed=3, reward_source="true")
    try:
        tot_true, s_true, info = run(e2, scripted, 2.0, START)
    finally:
        e2.close()
    print(f"\n[source] progress camera {s_cam['progress']:.3f} vs true {s_true['progress']:.3f}; "
          f"totals {tot_cam:.2f} vs {tot_true:.2f}")
    assert s_true["progress"] == pytest.approx(s_cam["progress"], abs=0.15)
