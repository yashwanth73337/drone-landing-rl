"""v6_drop B5: the world (ground + platform with pad + drone + camera + start rules).

Checks:
  1. 300 resets obey every Stage 1 start rule (height, offset, heading, initial
     velocity and turn rate, pad speed, pad centre in view, >= 1 marker detected)
  2. the same seed gives the same episode start (pad speed, spawn, first image)
  3. separate random streams: other controller gains do not change the starts
  4. one step = one decision = 0.1 s = 10 physics steps; the pad moves v * 0.1 m
  5. the drone hovers in place in the full world (pad moving underneath)
  6. the ground is solid: a drone dropped with motors off comes to rest on it
  7. after flying, the rendered image still matches the pad's true position
  8. timing: decisions per second (step + render + measure), printed only
  9. only one EGL World may be open per process (two open EGL clients render wrong
     images; found in B5). Extra Worlds in tests 2-3 therefore use TinyRenderer.

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_b5_world.py -v -s
"""
import time

import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import lmf2_params as P
from v6_drop.envs import pad_board as PB
from v6_drop.envs import world as W
from v6_drop.envs.platform_s1 import PAD_TOP_Z, V_MAX


def _egl_ok():
    try:
        w = W.World(renderer="egl")
        w.close()
        return True
    except Exception:
        return False


RENDERER = "egl" if _egl_ok() else "tiny"


@pytest.fixture(scope="module")
def world():
    w = W.World(renderer=RENDERER, seed=11)
    yield w
    w.close()


def yaw_of(R):
    return np.arctan2(R[1, 0], R[0, 0])


# ---------------------------------------------------------------- 1. start rules
def test_start_rules(world):
    n, tries, speeds = 300, [], []
    for _ in range(n):
        out = world.reset()
        sp, s = out["spawn"], world.true_state()
        tries.append(out["tries"])
        speeds.append(world.plat.v)
        rel = s["pos"] - s["pad_center"]
        assert W.H_RANGE[0] <= rel[2] <= W.H_RANGE[1]
        assert abs(rel[0]) <= W.DXY_MAX and abs(rel[1]) <= W.DXY_MAX
        assert abs(yaw_of(s["R"]) - world.plat.psi) <= W.YAW_NOISE + 1e-9
        assert np.all(np.abs(sp["v"]) <= W.V0_MAX) and np.allclose(s["v"], sp["v"], atol=1e-9)
        assert np.all(np.abs(sp["w_body"]) <= W.W0_MAX)
        assert 0.0 <= world.plat.v <= V_MAX
        assert world.pad_centre_in_view(s["R"], s["pos"])
        assert out["measurement"].ids, "start without a detected marker"
        assert out["image"].shape == (320, 512)
    tries = np.array(tries)
    print(f"\n[starts {RENDERER}] {n} resets OK; tries per start mean {tries.mean():.2f}, "
          f"max {tries.max()}; pad speed mean {np.mean(speeds):.2f} m/s")


# ---------------------------------------------------------------- 2. determinism
def test_same_seed_same_start():
    outs = []
    for _ in range(2):
        w = W.World(renderer="tiny", seed=123)        # tiny: the EGL fixture World is open
        try:
            o = w.reset()
            outs.append((w.plat.v, np.array(o["spawn"]["pos"]), o["image"].copy(),
                         np.array(o["spawn"]["v"])))
        finally:
            w.close()
    (v1, p1, i1, u1), (v2, p2, i2, u2) = outs
    assert v1 == v2 and np.array_equal(p1, p2) and np.array_equal(u1, u2)
    assert np.array_equal(i1, i2)


# ---------------------------------------------------------------- 3. stream independence
def test_gains_do_not_shift_starts():
    starts = []
    for mode in ("nominal", "random"):
        w = W.World(renderer="tiny", seed=5, gains_mode=mode)
        try:
            starts.append([(w.reset()["spawn"]["pos"].copy(), w.plat.v) for _ in range(5)])
        finally:
            w.close()
    for (pa, va), (pb, vb) in zip(*starts):
        assert np.array_equal(pa, pb) and va == vb


# ---------------------------------------------------------------- 4. one decision
def test_step_is_one_decision(world):
    world.reset(pad_speed=3.0)
    x0, t0 = world.plat.x, world.t
    for _ in range(5):
        world.step([0.0, 0.0, 0.0, 0.0])
    assert world.t == t0 + 5
    assert world.plat.x - x0 == pytest.approx(3.0 * 5 * P.POLICY_DT, abs=1e-9)
    assert P.SUBSTEPS == 10 and P.PHYSICS_DT * P.SUBSTEPS == pytest.approx(P.POLICY_DT)


# ---------------------------------------------------------------- 5. hover in the world
def test_hover_in_world(world):
    spawn = dict(pos=[-2.0, 0.0, PAD_TOP_Z + 3.0], yaw=0.0)
    world.reset(pad_speed=2.0, spawn=spawn)
    for _ in range(20):                                        # 2 s
        world.step([0.0, 0.0, 0.0, 0.0])
    drift = np.linalg.norm(world.true_state()["pos"] - np.array(spawn["pos"]))
    print(f"\n[hover in world] drift after 2 s = {drift*100:.2f} cm")
    assert drift < 0.05


# ---------------------------------------------------------------- 6. solid ground
def test_ground_is_solid(world):
    world.reset(pad_speed=0.0, spawn=dict(pos=[10.0, 0.0, 0.8], yaw=0.0))
    for _ in range(100):                                       # 1 s, motors OFF
        p.stepSimulation(physicsClientId=world.cid)
        world.plat.integrate()
    z = world.true_state()["pos"][2]
    print(f"\n[ground] drone centre height after falling 0.55 m with motors off: {z:.3f} m")
    assert z == pytest.approx(0.25, abs=0.01)                  # cube half-height on the ground


# ---------------------------------------------------------------- 7. image matches truth
def test_image_matches_truth_after_flying(world):
    world.reset(pad_speed=2.0, spawn=dict(pos=[-3.0, 0.3, PAD_TOP_Z + 3.0], yaw=0.0))
    for _ in range(15):                                        # 1.5 s, gentle forward flight
        world.step([1.0, 0.0, -0.3, 0.0])
    s = world.true_state()
    m = PB.measure(world.render(), world.cam, world.layout, world.det)
    assert m.ok
    err = np.linalg.norm(m.p_body - s["rel_pos_body"])
    print(f"\n[after flying] est {np.round(m.p_body, 3)} truth {np.round(s['rel_pos_body'], 3)} reproj {m.reproj_px:.2f}")
    print(f"[after flying] ids {m.ids}  |pad estimate - truth| = {err*100:.1f} cm "
          f"at distance {np.linalg.norm(s['rel_pos_body']):.2f} m")
    assert err < 0.15


# ---------------------------------------------------------------- 8. timing
def test_timing(world):
    world.reset()
    n, t0 = 100, time.time()
    for _ in range(n):
        world.step([0.5, 0.0, 0.0, 0.0])
        PB.measure(world.render(), world.cam, world.layout, world.det)
    rate = n / (time.time() - t0)
    print(f"\n[timing {RENDERER}] {rate:.0f} decisions/s (10 physics steps + 1 render + ArUco)")
    assert rate > 5


# ---------------------------------------------------------------- 9. one EGL World per process
@pytest.mark.skipif(RENDERER != "egl", reason="EGL plugin unavailable")
def test_second_egl_world_refused(world):
    with pytest.raises(RuntimeError, match="one EGL World per process"):
        W.World(renderer="egl", seed=99)
    t = W.World(renderer="tiny", seed=99)              # TinyRenderer is still allowed
    t.close()
