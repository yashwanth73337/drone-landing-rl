"""v7 S1a: the V7 platform with a random driver (19 Nov 2026).

Checks:
  1. limits: over many random 60 s flights, speed stays in [0, v_max], speeding up and
     braking stay <= 2 m/s^2, sideways acceleration (v x turn rate) stays <= 2 m/s^2
  2. random mode uses all four kinds of pieces (keep, speed change, wave, corner)
  3. same seed -> same path; other seed -> other path (repeatable)
  4. reset() takes exactly two numbers from the caller's rng
  5. named path "straight": one speed, never turns (Stage 1 at up to v_max)
  6. named path "wave": the heading swings and comes back; the road direction is kept
  7. named path "corners": every corner turns exactly 90 deg, and the pad brakes to the
     corner speed BEFORE turning (like a car)
  8. a fixed plan: braking 9 -> 3 m/s at 2 m/s^2 takes 3.0 s and 18 m (v^2 formula)
  9. PyBullet carries a box resting on the pad through a 90 deg corner (D19 + turn rate):
     it stays on the pad and turns with it; as a teaching check, WITHOUT the turn rate
     the box does not turn with the pad
 10. v_max can be changed per flight (8 m/s fallback)

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v7/tests/test_s1a_pad.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import lmf2_params as P
from v7.envs import platform_v7 as PV
from v7.envs.platform_v7 import A_LAT, A_MAX, PAD_SIZE, PAD_TOP_Z, V_MAX, PlatformV7

DT = P.PHYSICS_DT
EPS = 1e-6


@pytest.fixture(scope="module")
def plat():
    cid = p.connect(p.DIRECT)
    pl = PlatformV7(cid)
    yield pl
    p.disconnect(cid)


def fly(pl, seconds, record=True):
    out = []
    for _ in range(int(round(seconds / DT))):
        pl.integrate(DT)
        if record:
            out.append((pl.x, pl.y, pl.psi, pl.v, pl.a, pl.w, pl.phase))
    return out


def test_1_limits(plat):
    worst = dict(v_max=0.0, v_min=1e9, a=0.0, lat=0.0)
    for s in range(150):
        plat.reset(np.random.default_rng(s), mode="random")
        for x, y, psi, v, a, w, ph in fly(plat, 60.0):
            worst["v_max"] = max(worst["v_max"], v)
            worst["v_min"] = min(worst["v_min"], v)
            worst["a"] = max(worst["a"], abs(a))
            worst["lat"] = max(worst["lat"], abs(v * w))
    print("\n  150 random flights x 60 s:", {k: round(val, 4) for k, val in worst.items()})
    assert worst["v_min"] >= 0.0 and worst["v_max"] <= V_MAX + EPS
    assert worst["a"] <= A_MAX + EPS
    assert worst["lat"] <= A_LAT + 1e-3


def test_2_all_pieces_used(plat):
    seen = {}
    for s in range(100):
        plat.reset(np.random.default_rng(1000 + s), mode="random")
        for *_, ph in fly(plat, 60.0):
            seen[ph] = seen.get(ph, 0) + 1
    print("\n  steps per piece:", seen)
    for name in ("keep", "speed up", "slow down", "wave", "corner-in", "corner-turn", "corner-out"):
        assert seen.get(name, 0) > 0, name


def test_3_repeatable(plat):
    runs = []
    for s in (7, 7, 8):
        plat.reset(np.random.default_rng(s), mode="random")
        runs.append(np.array([r[:6] for r in fly(plat, 30.0)]))
    assert np.array_equal(runs[0], runs[1])
    assert not np.allclose(runs[0], runs[2])


def test_4_two_draws(plat):
    for mode in PV.MODES:
        rng = np.random.default_rng(3)
        plat.reset(rng, mode=mode)
        ref = np.random.default_rng(3)
        ref.uniform(); ref.integers(2 ** 62)
        assert rng.uniform() == ref.uniform(), mode


def test_5_straight(plat):
    plat.reset(np.random.default_rng(4), mode="straight")
    v0 = plat.v
    rec = np.array([r[:6] for r in fly(plat, 30.0)])
    assert np.allclose(rec[:, 3], v0) and np.allclose(rec[:, 2], 0.0)
    assert abs(rec[-1, 0] - v0 * 30.0) < 1e-6 and np.allclose(rec[:, 1], 0.0)


def test_6_wave(plat):
    plat.reset(np.random.default_rng(5), mode="wave")
    amp, per = plat.driver.wave
    rec = np.array([r[:6] for r in fly(plat, 3 * per + 0.5 * DT)])
    print(f"\n  wave: speed {plat.v:.2f} m/s, swing +-{np.degrees(amp):.1f} deg, period {per:.1f} s")
    assert np.degrees(np.abs(rec[:, 2]).max()) == pytest.approx(np.degrees(amp), abs=0.5)
    assert abs(np.degrees(rec[-1, 2])) < 1.0              # heading back after 3 periods
    assert np.abs(rec[:, 3] * rec[:, 5]).max() <= A_LAT + 1e-3
    assert abs(rec[-1, 1]) < 0.05 * rec[-1, 0]            # still going along the road (x)


def test_7_corners(plat):
    plat.reset(np.random.default_rng(6), mode="corners")
    cruise = plat.v
    rec = fly(plat, 60.0)
    psi = np.array([r[2] for r in rec])
    ph = [r[6] for r in rec]
    v = np.array([r[3] for r in rec])
    turns, i = [], 0
    while i < len(ph):
        if ph[i] == "corner-turn":
            j = i
            while j < len(ph) and ph[j] == "corner-turn":
                j += 1
            turns.append((i, j))
            i = j
        else:
            i += 1
    print(f"\n  corners: cruise {cruise:.2f} m/s, {len(turns)} corners in 60 s, "
          f"turn speeds {[round(float(v[a]), 2) for a, b in turns]}")
    assert len(turns) >= 2
    for a, b in turns:
        before = psi[a - 1]
        assert abs(abs(np.degrees(psi[b - 1] - before)) - 90.0) < 0.6
        assert np.allclose(v[a:b], v[a])                  # steady speed in the corner
        assert v[a] <= cruise + EPS                       # braked (or kept) before the turn
        R = PV.corner_radius(v[a])
        assert v[a] ** 2 / R <= A_LAT + 1e-6


def test_8_fixed_plan_braking(plat):
    plat.reset(np.random.default_rng(0), v=9.0, plan=[("speed", 3.0, 2.0)])
    rec = np.array([r[:6] for r in fly(plat, 4.0)])
    k = int(np.argmax(rec[:, 3] <= 3.0 + 1e-9))
    t = (k + 1) * DT
    dist = rec[k, 0]
    print(f"\n  braking 9 -> 3 m/s at 2 m/s^2: {t:.2f} s, {dist:.2f} m (formula 3.00 s, 18.00 m)")
    assert t == pytest.approx(3.0, abs=DT)
    assert dist == pytest.approx((9 ** 2 - 3 ** 2) / (2 * 2.0), abs=0.05)


def _box_on_corner(sync_turn_rate):
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -P.G, physicsClientId=cid)
    p.setTimeStep(DT, physicsClientId=cid)
    pl = PlatformV7(cid)
    half = [0.15, 0.15, 0.05]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
    box = p.createMultiBody(1.24, col, -1, [0.3, 0.0, PAD_TOP_Z + half[2] + 0.001], physicsClientId=cid)
    p.changeDynamics(box, -1, lateralFriction=PV.DRONE_FRICTION, physicsClientId=cid)
    # a 3 m/s pad, already turning: a 90 deg left corner on R = 6 m (2 s settle first)
    pl.reset(np.random.default_rng(0), v=3.0, plan=[("keep", 2.0), ("turn", np.pi / 2, 6.0, "corner-turn"),
                                                   ("keep", 2.0)])
    if not sync_turn_rate:
        orig = pl.sync

        def sync_no_w():
            q = p.getQuaternionFromEuler([0, 0, pl.psi])
            p.resetBasePositionAndOrientation(pl.body, [pl.x, pl.y, PAD_TOP_Z / 2], q, physicsClientId=cid)
            p.resetBaseVelocity(pl.body, pl.velocity().tolist(), [0, 0, 0], physicsClientId=cid)
        pl.sync = sync_no_w
    p.resetBaseVelocity(box, pl.velocity().tolist(), [0, 0, 0], physicsClientId=cid)
    yaw0 = None
    for k in range(int(2.0 / DT) + int((np.pi / 2 * 6.0 / 3.0) / DT) + int(2.0 / DT)):
        p.stepSimulation(physicsClientId=cid)
        pl.integrate(DT)
        if k == int(2.0 / DT):
            yaw0 = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(box, physicsClientId=cid)[1])[2]
    pos, q = p.getBasePositionAndOrientation(box, physicsClientId=cid)
    yaw = p.getEulerFromQuaternion(q)[2]
    on_pad = np.all(np.abs(pl.world_to_pad(np.array([pos]))[0]) < PAD_SIZE / 2)
    p.disconnect(cid)
    return on_pad, np.degrees(yaw - yaw0), np.degrees(pl.psi)


def test_9_carried_through_corner():
    on, turned, pad = _box_on_corner(True)
    on0, turned0, _ = _box_on_corner(False)
    print(f"\n  with turn rate: on pad {on}, box turned {turned:.1f} deg (pad {pad:.1f})"
          f"\n  without turn rate: on pad {on0}, box turned {turned0:.1f} deg")
    assert on and abs(turned - 90.0) < 5.0
    assert abs(turned0) < abs(turned) - 20.0             # teaching check: D19 needs the turn rate too


def test_10_v_max(plat):
    for s in range(30):
        plat.reset(np.random.default_rng(s), mode="random", v_max=8.0)
        vs = [r[3] for r in fly(plat, 40.0)]
        assert max(vs) <= 8.0 + EPS
    plat.reset(np.random.default_rng(0), mode="random")
    assert plat.v_max == V_MAX == 9.0
