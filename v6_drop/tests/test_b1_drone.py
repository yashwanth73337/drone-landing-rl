"""v6_drop B1: LMF2 drone + motor model + Lee velocity controller.

Ported from v5_shin/tests/test_block1_controller.py (same pass criteria), plus
one new test, test_tilt_when_accelerating, for the camera lesson of Step 3A.

Pass criteria for the velocity steps. An ideal velocity loop v' = K (v_sp - v)
reaches 50% at ln(2)/K. The lateral loop acts through attitude (a slow
attitude pole, ~0.23 s) plus 50-80 ms motor spin-up, so:
  0.85 ln2/K <= t50 <= 3 ln2/K,  overshoot < 15%,  steady-state error < 2%.
The lower bound is below ideal because motors spin DOWN in 5 ms but UP in
50-80 ms; in a descent step the drone briefly drops faster than commanded.

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_b1_drone.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v6_drop.envs import lmf2_params as P
from v6_drop.envs.quad import LMF2Quad, MotorModel, make_client
from v6_drop.envs.lee_controller import yaw_rotation

GAIN_MODES = ["nominal", "min", "max"]


def fly(cmd, T, gains_mode="nominal", yaw0=0.0, z0=20.0, seed=0):
    """Fly a constant heading-frame command; returns a per-physics-step log."""
    rng = np.random.default_rng(seed)
    cid = make_client()
    try:
        quad = LMF2Quad(cid, rng, P.sample_gains(rng, gains_mode))
        quad.reset_pose([0, 0, z0], yaw=yaw0)
        log = {"t": [], "v": [], "v_head": [], "pos": [], "rpy": [], "thrust": []}
        for k in range(int(round(T / P.PHYSICS_DT))):
            quad.apply_control(cmd)
            p.stepSimulation(physicsClientId=cid)
            s = quad.state()
            log["t"].append((k + 1) * P.PHYSICS_DT)
            log["v"].append(s["v"])
            yaw = p.getEulerFromQuaternion(s["quat"])[2]
            log["v_head"].append(yaw_rotation(yaw).T @ s["v"])
            log["pos"].append(s["pos"])
            log["rpy"].append(p.getEulerFromQuaternion(s["quat"]))
            log["thrust"].append(quad.motors.thrust.copy())
        return {k: np.array(v) for k, v in log.items()}, quad.ctrl
    finally:
        p.disconnect(cid)


def step_metrics(t, y, target):
    y = y / target
    t50 = t[np.argmax(y >= 0.5)]
    return t50, max(0.0, y.max() - 1.0), abs(y[-1] - 1.0)


# ---------------------------------------------------------------- static checks
def test_mass_and_inertia_loaded():
    cid = make_client()
    try:
        quad = LMF2Quad(cid, np.random.default_rng(0), P.sample_gains(None, "nominal"))
        info = p.getDynamicsInfo(quad.body, -1, physicsClientId=cid)
        assert info[0] == pytest.approx(P.MASS, abs=1e-6)
        assert np.allclose(info[2], P.INERTIA, atol=1e-9), info[2]
    finally:
        p.disconnect(cid)


def test_allocation_roundtrip():
    w = np.array([P.MASS * P.G, 0.05, -0.03, 0.02])
    assert np.allclose(P.ALLOCATION @ (P.ALLOCATION_INV @ w), w)


def test_motor_time_constants():
    """Spin-up in sqrt-thrust space reaches 63% in ~tau_inc; spin-down in ~tau_dec."""
    m = MotorModel(np.random.default_rng(0), dt=0.0005)   # fine dt to resolve tau
    m.tau_inc[:] = 0.06
    m.reset(1.0)
    s0, s1 = 1.0, np.sqrt(9.0)
    for k in range(1, 2000):
        m.update(np.full(4, 9.0))
        if (np.sqrt(m.thrust[0]) - s0) / (s1 - s0) >= 0.632:
            break
    assert k * 0.0005 == pytest.approx(0.06, rel=0.1)
    m.reset(9.0)
    for k in range(1, 2000):
        m.update(np.full(4, 1.0))
        if (s1 - np.sqrt(m.thrust[0])) / (s1 - s0) >= 0.632:
            break
    assert k * 0.0005 == pytest.approx(P.TAU_DEC, rel=0.15)


def test_gains_sampling_ranges():
    rng = np.random.default_rng(1)
    for _ in range(200):
        g = P.sample_gains(rng)
        for key, (lo, hi) in (("K_vel", P.K_VEL_RANGE), ("K_rot", P.K_ROT_RANGE),
                              ("K_angvel", P.K_ANGVEL_RANGE)):
            assert np.all(g[key] >= lo) and np.all(g[key] <= hi)


# ---------------------------------------------------------------- closed loop
@pytest.mark.parametrize("mode", GAIN_MODES)
def test_hover(mode):
    log, _ = fly([0, 0, 0, 0], T=5.0, gains_mode=mode)
    drift = np.linalg.norm(log["pos"][-1] - np.array([0, 0, 20.0]))
    tilt = np.degrees(np.abs(log["rpy"][-100:, :2]).max())
    print(f"\n[hover {mode}] drift={drift:.4f} m  max tilt(last 1 s)={tilt:.3f} deg")
    assert drift < 0.05 and tilt < 0.5


@pytest.mark.parametrize("mode", GAIN_MODES)
@pytest.mark.parametrize("axis,step", [(0, 2.0), (1, 2.0), (2, -1.0), (2, 1.0)])
def test_velocity_step(mode, axis, step):
    cmd = [0.0, 0.0, 0.0, 0.0]
    cmd[axis] = step
    # z loop is slower (K 1.3-1.7): 6 s window so steady state is reached.
    log, ctrl = fly(cmd, T=4.0 if axis < 2 else 6.0, gains_mode=mode)
    # Heading frame: the controller holds yaw RATE, not yaw angle, so a few
    # degrees of yaw drift during a manoeuvre is expected.
    vh = log["v_head"]
    t50, os_, sse = step_metrics(log["t"], vh[:, axis], step)
    theory = np.log(2) / ctrl.K_vel[axis]
    cross = np.abs(np.delete(vh[-100:], axis, axis=1)).max()
    print(f"\n[step {mode} ax{axis} {step:+}] t50={t50:.3f}s (ideal {theory:.3f}) "
          f"overshoot={os_*100:.1f}% sse={sse*100:.2f}% cross-axis={cross:.3f} m/s")
    assert 0.85 * theory <= t50 <= 3 * theory
    assert os_ < 0.15 and sse < 0.02 and cross < 0.05


def test_heading_frame():
    """Drone yawed +90 deg: heading-frame vx = 2 must become world +y."""
    log, _ = fly([2.0, 0, 0, 0], T=4.0, yaw0=np.pi / 2)
    v, yaw = log["v"][-1], log["rpy"][-1, 2]
    expect = yaw_rotation(yaw) @ np.array([2.0, 0, 0])
    print(f"\n[heading] yaw={np.degrees(yaw):.2f} deg  world v={np.round(v, 3)}  "
          f"expected={np.round(expect, 3)}")
    assert abs(yaw - np.pi / 2) < np.radians(10)
    assert np.allclose(v, expect, atol=0.04)


@pytest.mark.parametrize("mode", GAIN_MODES)
@pytest.mark.parametrize("rate,expect", [(0.5, 0.5), (-0.5, -0.5), (2.0, P.MAX_YAW_RATE)])
def test_yaw_rate(mode, rate, expect):
    log, _ = fly([0, 0, 0, rate], T=4.0, gains_mode=mode)
    yaw = np.unwrap(log["rpy"][:, 2])
    meas = (yaw[-1] - yaw[-101]) / 1.0            # mean rate over the last 1 s
    print(f"\n[yaw {mode} cmd={rate}] measured={meas:.4f} rad/s (expect {expect:.4f})")
    assert meas == pytest.approx(expect, rel=0.05)


def test_high_speed_chase():
    """Headroom check: Stage 1 pads go up to 4 m/s, later stages faster. The drone
    must reach 9.5 of a 10 m/s command in < 3 s, losing < 1 m of altitude."""
    log, _ = fly([10.0, 0, 0, 0], T=4.0)
    ok = log["v"][:, 0] >= 9.5
    t_reach = log["t"][np.argmax(ok)] if ok.any() else np.inf
    dz = log["pos"][:, 2].min() - 20.0
    sat = (log["thrust"] >= P.MOTOR_MAX_THRUST - 1e-6).any(axis=1).mean()
    print(f"\n[chase 10 m/s] t(9.5 m/s)={t_reach:.2f}s  min dz={dz:.3f} m  "
          f"steps with a saturated motor={sat*100:.1f}%")
    assert t_reach < 3.0 and dz > -1.0


@pytest.mark.parametrize("mode", GAIN_MODES)
@pytest.mark.parametrize("accel", [1.0, 2.0, 3.0, None])
def test_tilt_when_accelerating(mode, accel):
    """NEW in v6 (Step 3A camera lesson). To speed up forward the drone tilts
    nose-down, so the 60 deg camera looks more downward while chasing.
    In PyBullet, nose-down while accelerating along +x is a POSITIVE pitch.

    accel = 1, 2, 3 m/s^2: the command ramps 0 -> 4 m/s by accel * 0.1 s per
    decision. Theory: tilt = atan(a / g) = 5.8 / 11.5 / 17.0 deg. Measured in
    Claude's sandbox: 6.7 / 13.0 / 19.2 deg (nominal gains); the extra ~1 deg is
    the tilt needed against PyBullet's linear damping.
    accel = None: a sudden step command 0 -> 4 m/s. Measured: ~36 deg peak.
    At 36 deg nose-down the 60 deg camera points ~96 deg below forward, i.e.
    slightly BACKWARD: the forward view is lost. So the actions must not jump
    (to settle in Step 4: limit how fast the velocity command may change)."""
    T, dt_dec = 6.0, P.POLICY_DT   # 6 s: the 1 m/s^2 ramp ends at 4 s, then settles
    rng = np.random.default_rng(0)
    cid = make_client()
    try:
        quad = LMF2Quad(cid, rng, P.sample_gains(rng, mode))
        quad.reset_pose([0, 0, 20.0])
        pitch, vx = [], 0.0
        for k in range(int(round(T / P.PHYSICS_DT))):
            if k % P.SUBSTEPS == 0:                     # one decision every 0.1 s
                vx = 4.0 if accel is None else min(4.0, vx + accel * dt_dec)
            quad.apply_control([vx, 0.0, 0.0, 0.0])
            p.stepSimulation(physicsClientId=cid)
            pitch.append(p.getEulerFromQuaternion(quad.state()["quat"])[1])
    finally:
        p.disconnect(cid)
    pitch = np.degrees(np.array(pitch))
    peak, steady = pitch.max(), np.abs(pitch[-50:]).max()   # last 0.5 s at ~4 m/s
    label = "step" if accel is None else f"ramp {accel} m/s^2"
    print(f"\n[tilt {mode} {label}] peak nose-down={peak:.1f} deg  "
          f"most nose-up={pitch.min():.1f} deg  steady |pitch|={steady:.2f} deg")
    if accel is None:
        assert 30.0 < peak < 45.0
        assert pitch.min() > -5.0        # small nose-up: braking the ~11% speed overshoot
    else:
        theory = np.degrees(np.arctan(accel / P.G))
        assert theory - 1.0 < peak < theory + 3.0
        assert pitch.min() > -2.0
    assert steady < 2.0                  # nearly level once the speed is reached


def test_jitter_sinks_documented_property():
    """DOCUMENTED PROPERTY (found in V5 Block 9), not a goal. The motors spin UP
    in 50-80 ms but DOWN in 5 ms, so under jittery commands the delivered thrust
    averages below the commanded thrust, and the drone sinks. Matters for
    exploration noise in training. If this test starts failing, the motor model
    or the controller changed."""
    rng = np.random.default_rng(0)
    cid = make_client()
    try:
        quad = LMF2Quad(cid, rng, P.sample_gains(rng, "nominal"))
        quad.reset_pose([0, 0, 50.0])
        z0, v = quad.state()["pos"][2], np.zeros(2)
        for _ in range(150):
            v = 0.7 * v + 0.3 * rng.normal(0, 1.0, 2) * 3
            for _ in range(P.SUBSTEPS):
                quad.apply_control([v[0], v[1], 0.0, 0.0])
                p.stepSimulation(physicsClientId=cid)
        rate = (quad.state()["pos"][2] - z0) / 15.0
        print(f"\n[jitter] vz command 0, lateral jitter: mean vertical velocity {rate:+.3f} m/s")
        assert rate < -0.3
    finally:
        p.disconnect(cid)
