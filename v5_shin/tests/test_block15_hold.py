"""Block 15: post-touchdown hold (SPEC D20). DR off, hand-placed drone, no policy."""
import numpy as np
import pybullet as p
import pytest

from v5_shin.envs import dr as DR
from v5_shin.envs.landing_sim import (HOLD_STEPS, HOLD_STILL_STEPS, HORIZON, V_IMP, V_STILL,
                                      LandingSim, hold_verdict)
from v5_shin.envs.platform import PAD_SIZE, PAD_TOP_Z

SPAWN = dict(pos=np.array([0.0, 0.0, 3.0]), yaw=0.0, psi_plat=0.0, v=(0.0, 0.0, 0.0))


def make(hold_steps=HOLD_STEPS):
    sim = LandingSim(renderer="tiny", marker=False, dr=DR.DRConfig.off(), seed=0,
                     hold_steps=hold_steps)
    sim.reset(c=0.0, spawn=dict(SPAWN))
    return sim


def place(sim, gap, vz, dxy=(0.0, 0.0), dz_extra=0.0, roll=0.0):
    """Drone's lowest point `gap` m above the pad top, moving with the pad plus vz."""
    body, cid = sim.quad.body, sim.cid
    pc = sim.plat.pad_center()
    sim.quad.reset_pose(np.array([pc[0], pc[1], 5.0]), yaw=0.0, v=np.zeros(3), w_body=np.zeros(3))
    p.performCollisionDetection(physicsClientId=cid)
    zmin = min(p.getAABB(body, l, physicsClientId=cid)[0][2]
               for l in range(-1, p.getNumJoints(body, physicsClientId=cid)))
    off = 5.0 - zmin
    v = sim.plat.velocity().copy()
    v[2] = vz
    pos = np.array([pc[0] + dxy[0], pc[1] + dxy[1], PAD_TOP_Z + off + gap + dz_extra])
    sim.quad.reset_pose(pos, yaw=0.0, v=v, w_body=np.zeros(3), roll=roll)


def run(sim, cmd, max_steps=40):
    for _ in range(max_steps):
        te, tr, info = sim.step(np.asarray(cmd, float))
        if te or tr:
            return te, tr, info
    raise AssertionError("episode did not end")


def land(sim, cmd):
    """Step until the touchdown (phase becomes 'hold'); returns the info of that step."""
    for _ in range(20):
        te, tr, info = sim.step(np.asarray(cmd, float))
        assert not (te or tr), info["outcome"]
        if sim.phase == "hold":
            return info
    raise AssertionError("no touchdown")


def test_hold_verdict():
    n = HOLD_STILL_STEPS
    assert hold_verdict([1.0] * 5 + [V_STILL] * n) == "success"
    assert hold_verdict([0.0] * (n - 1) + [V_STILL + 0.01]) == "not_still"
    assert hold_verdict([0.0] * (n - 1)) == "not_still"         # too short


def test_hold_off_is_paper_task():
    sim = make(hold_steps=0)
    try:
        place(sim, 0.02, -0.3)
        te, tr, info = run(sim, [0, 0, -0.3, 0])
        assert te and not tr and info["outcome"] == "success"
        assert "phase" not in info and "touchdown" not in info      # info unchanged
    finally:
        sim.close()


def test_soft_landing_and_hold_success_static_pad():
    sim = make()
    try:
        place(sim, 0.02, -0.3)
        land(sim, [0, 0, -0.3, 0])
        te, tr, info = run(sim, [0, 0, -0.3, 0])
        assert te and not tr, info
        assert info["outcome"] == "success", info
        assert info["touchdown"] and info["hold_steps_done"] == HOLD_STEPS
        assert abs(info["rel_vel"][2]) <= V_IMP                    # impact, from the touchdown
        assert info["hold_vrel_max_last"] <= V_STILL
    finally:
        sim.close()


def test_hold_success_on_moving_pad():
    """D19 + D20: the drone holds on a pad moving at a constant 3 m/s."""
    sim = make()
    try:
        sim.plat.perturb = lambda rng: None    # constant speed (at c = 0 perturb clips v to 0)
        sim.plat.v = 3.0
        sim.plat._sync()
        place(sim, 0.02, -0.3)
        land(sim, [3.0, 0, -0.3, 0])
        te, tr, info = run(sim, [3.0, 0, -0.3, 0])
        assert info["outcome"] == "success", info
    finally:
        sim.close()


def test_hard_landing():
    sim = make()
    try:
        place(sim, 0.05, -3.0)
        te, tr, info = run(sim, [0, 0, -1.0, 0])
        assert te and info["outcome"] == "hard_landing", info
        assert info["touchdown"] and abs(info["rel_vel"][2]) > V_IMP
    finally:
        sim.close()


def test_bounce():
    sim = make()
    try:
        place(sim, 0.02, -0.3)
        land(sim, [0, 0, -0.3, 0])
        te, tr, info = run(sim, [0, 0, 1.0, 0])                   # climb away
        assert info["outcome"] == "bounce", info
    finally:
        sim.close()


def test_tip_over():
    sim = make()
    try:
        place(sim, 0.02, -0.3)
        land(sim, [0, 0, -0.3, 0])
        place(sim, 0.0, 0.0, dz_extra=0.3, roll=np.radians(40.0))  # tilted, above the pad
        te, tr, info = sim.step(np.array([0, 0, 0, 0], float))
        assert te and info["outcome"] == "tip_over", info
    finally:
        sim.close()


def test_slide_off():
    sim = make()
    try:
        place(sim, 0.02, -0.3)
        land(sim, [0, 0, -0.3, 0])
        place(sim, 0.0, 0.0, dxy=(PAD_SIZE / 2 + 0.2, 0.0), dz_extra=0.3)   # CoM beyond the edge
        te, tr, info = sim.step(np.array([0, 0, 0, 0], float))
        assert te and info["outcome"] == "slide_off", info
    finally:
        sim.close()


def test_hold_runs_past_flight_horizon():
    sim = make()
    try:
        sim.t = HORIZON - 1                    # last flight step
        place(sim, 0.01, -0.3)
        te, tr, info = sim.step(np.array([0, 0, -0.3, 0], float))
        assert sim.phase == "hold" and not (te or tr), info
        te, tr, info = run(sim, [0, 0, -0.3, 0])
        assert not tr and info["outcome"] == "success" and info["t"] > HORIZON, info
    finally:
        sim.close()


@pytest.mark.parametrize("hold_steps", [0, HOLD_STEPS])
def test_timeout_still_applies_in_flight(hold_steps):
    sim = make(hold_steps)
    try:
        sim.t = HORIZON - 1
        place(sim, 1.0, 0.0)                   # hovering 1 m above the pad
        te, tr, info = sim.step(np.array([0, 0, 0, 0], float))
        assert tr and not te and info["outcome"] == "timeout", info
    finally:
        sim.close()
