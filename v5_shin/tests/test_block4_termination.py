"""Block 4: world assembly, Table I spawn, termination.

Hand-built states check every outcome; spawn statistics are checked over
2000 resets (seed base 20000).

Run:  python -m pytest v5_shin/tests/test_block4_termination.py -v -s
"""
import numpy as np
import pytest

from v5_shin.envs import landing_sim as LS
from v5_shin.envs import lmf2_params as P
from v5_shin.envs.landing_sim import LandingSim
from v5_shin.envs.platform import PAD_SIZE, PAD_TOP_Z

REACH = PAD_SIZE / 2 + P.COLLISION_HALF    # contact reach at yaw 0 (1.0 m with the LMF2 box)


@pytest.fixture(scope="module")
def sim():
    s = LandingSim(seed=1, gains_mode="nominal")
    yield s
    s.close()


def run(sim, pos, cmd, yaw=0.0, c=0.0, steps=400, **kw):
    sim.reset(c=c, spawn=dict(pos=np.asarray(pos, float), yaw=yaw, psi_plat=0.0, **kw))
    for _ in range(steps):
        te, tr, info = sim.step(cmd)
        if te or tr:
            return te, tr, info
    raise AssertionError("episode did not end")


def ks_uniform(x, lo, hi):
    x = np.sort((np.asarray(x) - lo) / (hi - lo))
    n = len(x)
    cdf = np.arange(1, n + 1) / n
    return max(np.max(cdf - x), np.max(x - (cdf - 1 / n))), 1.63 / np.sqrt(n)


# ---------------------------------------------------------------- outcomes
def test_success_centre(sim):
    te, tr, info = run(sim, [0, 0, PAD_TOP_Z + 1.0], [0, 0, -1, 0])
    print(f"\n[centre] {info['outcome']} t={info['t']} sub={info['substeps']} "
          f"CoM above pad {info['rel_pos'][2]:.3f} m")
    assert te and not tr and info["outcome"] == "success"
    assert info["com_over_pad"]
    assert info["rel_pos"][2] == pytest.approx(P.COLLISION_HALF, abs=0.02)


@pytest.mark.parametrize("dx,expect", [(REACH - 0.03, "success"), (REACH + 0.03, "crash_ground")])
def test_success_region_edge(sim, dx, expect):
    """Documents the success region: contact reach = pad half-size + collision half-size."""
    te, _, info = run(sim, [dx, 0, PAD_TOP_Z + 1.0], [0, 0, -1, 0])
    print(f"\n[edge dx={dx:.2f}] {info['outcome']}  com_over_pad={info['com_over_pad']}")
    assert te and info["outcome"] == expect
    if expect == "success":
        assert not info["com_over_pad"]      # counted as success, but the CoM is off the pad


def test_crash_ground(sim):
    te, _, info = run(sim, [3.0, 0, PAD_TOP_Z + 1.0], [0, 0, -1, 0])
    assert te and info["outcome"] == "crash_ground"


def test_crash_platform_side(sim):
    te, _, info = run(sim, [-2.0, 0, 0.5], [2, 0, 0, 0])
    assert te and info["outcome"] == "crash_platform"
    assert info["rel_pos"][0] == pytest.approx(-REACH, abs=0.05)


def test_drift_lateral(sim):
    te, _, info = run(sim, [0, 0, PAD_TOP_Z + 4.0], [10, 0, 0, 0])
    assert te and info["outcome"] == "drift"
    assert np.hypot(*info["rel_pos"][:2]) == pytest.approx(LS.DRIFT_XY, abs=0.15)


def test_drift_vertical(sim):
    te, _, info = run(sim, [0, 0, PAD_TOP_Z + 10.0], [0, 0, 3, 0])
    assert te and info["outcome"] == "drift"
    assert info["rel_pos"][2] == pytest.approx(LS.DRIFT_Z, abs=0.05)


def test_tilt(sim):
    te, _, info = run(sim, [0, 0, PAD_TOP_Z + 5.0], [0, 0, 0, 0], roll=np.radians(85))
    assert te and info["outcome"] == "tilt" and info["t"] == 1 and info["substeps"] == 1


def test_timeout(sim):
    te, tr, info = run(sim, [0, 0, PAD_TOP_Z + 5.0], [0, 0, 0, 0])
    assert tr and not te and info["outcome"] == "timeout" and info["t"] == LS.HORIZON


def test_step_after_done_raises(sim):
    run(sim, [0, 0, PAD_TOP_Z + 1.0], [0, 0, -1, 0])
    with pytest.raises(AssertionError):
        sim.step([0, 0, 0, 0])


def test_moving_platform_success(sim):
    """Oracle-lite: match the platform's velocity while descending. Proves contact
    detection on a teleported kinematic body at 6 m/s."""
    sim.reset(c=1.0, spawn=dict(pos=np.array([0, 0, PAD_TOP_Z + 2.0]), yaw=0.0, psi_plat=0.0))
    sim.quad.reset_pose([0, 0, PAD_TOP_Z + 2.0], v=(6.0, 0, 0))
    for _ in range(100):
        # Hold v = 6 m/s, w = 0 each step (overrides the random walk; NOT c = 0,
        # which would also clip v to [0, 8c] = 0 and stop the platform)
        sim.plat.v, sim.plat.w = 6.0, 0.0
        rel_p, rel_v, s = sim.true_relative_state()
        cmd = [6.0 + 1.5 * rel_p[0], 1.5 * rel_p[1], -1.0, 0.0]
        te, tr, info = sim.step(cmd)
        if te or tr:
            break
    print(f"\n[moving 6 m/s] {info['outcome']} rel_pad={np.round(info['rel_pos_pad'], 3)} "
          f"rel_vel={np.round(info['rel_vel'], 2)}")
    assert info["outcome"] == "success" and info["com_over_pad"]


# ---------------------------------------------------------------- spawn
def test_spawn_statistics():
    sim = LandingSim(seed=20000)
    try:
        rows = [sim.sample_spawn(1.0) for _ in range(2000)]
    finally:
        sim.close()
    d = np.array([r["dxyz0"] for r in rows])
    tries = np.array([r["tries"] for r in rows])
    psi0 = np.array([r["psi0"] for r in rows])
    bearing = np.arctan2(-d[:, 1], -d[:, 0])
    yaw_off = np.angle(np.exp(1j * (np.array([r["yaw"] for r in rows]) - bearing)))
    uv = np.array([r["uv0"] for r in rows])
    ks = {k: ks_uniform(v, lo, hi) for k, (v, lo, hi) in dict(
        dx=(d[:, 0], -3, 3), dy=(d[:, 1], -3, 3), dz=(d[:, 2], 2, 8),
        psi0=(psi0, -LS.PSI0_MAX, LS.PSI0_MAX),
        yaw_off=(yaw_off, -LS.YAW_NOISE, LS.YAW_NOISE)).items()}
    acc = len(tries) / tries.sum()
    print("\n[spawn] acceptance rate {:.1%}; KS: ".format(acc)
          + ", ".join(f"{k} {v[0]:.3f}{'<' if v[0] < v[1] else '>='}{v[1]:.3f}" for k, v in ks.items())
          + f"\n        pad-centre pixel v: median {np.median(uv[:, 1]):.0f}, "
            f"range [{uv[:, 1].min():.0f}, {uv[:, 1].max():.0f}] of 320")
    assert (np.abs(d[:, :2]) <= 3).all() and (d[:, 2] >= 2).all() and (d[:, 2] <= 8).all()
    assert (np.abs(yaw_off) <= LS.YAW_NOISE + 1e-9).all()
    assert (uv >= LS.FOV_MARGIN_PX).all() and (uv[:, 0] <= 512 - LS.FOV_MARGIN_PX).all() \
        and (uv[:, 1] <= 320 - LS.FOV_MARGIN_PX).all()
    for k in ("psi0", "yaw_off"):         # never rejected on these alone -> must stay uniform
        assert ks[k][0] < ks[k][1], k
