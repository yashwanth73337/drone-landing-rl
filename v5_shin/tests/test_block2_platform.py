"""Block 2: platform random walk and kinematics.

Distribution checks use a one-sample Kolmogorov-Smirnov statistic against
the specified uniform. Critical value at alpha = 0.01: 1.63/sqrt(n).
Increments are only collected on steps where no clip occurred, since clipped
steps are by construction not U(-a, a).

Run:  python -m pytest v5_shin/tests/test_block2_platform.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v5_shin.envs import lmf2_params as P
from v5_shin.envs import platform as PL
from v5_shin.envs.platform import Platform
from v5_shin.envs.quad import make_client


def ks_uniform(x, lo, hi):
    x = np.sort((np.asarray(x) - lo) / (hi - lo))
    n = len(x)
    cdf = np.arange(1, n + 1) / n
    d = max(np.max(cdf - x), np.max(x - (cdf - 1 / n)))
    return d, 1.63 / np.sqrt(n)


@pytest.fixture
def plat():
    cid = make_client()
    yield Platform(cid)
    p.disconnect(cid)


def run_episodes(plat, c, n_eps, seed, steps=300):
    rng = np.random.default_rng(seed)
    v0, dv, dw, v_all, w_all = [], [], [], [], []
    clips = [0, 0, 0]
    for _ in range(n_eps):
        plat.reset(rng, c)
        v0.append(plat.v)
        for _ in range(steps):
            v_prev, w_prev = plat.v, plat.w
            cv, cw = plat.n_clip_v, plat.n_clip_w
            plat.perturb(rng)
            if plat.n_clip_v == cv:
                dv.append(plat.v - v_prev)
            if plat.n_clip_w == cw:
                dw.append(plat.w - w_prev)
            v_all.append(plat.v)
            w_all.append(plat.w)
        clips[0] += plat.n_clip_v
        clips[1] += plat.n_clip_w
        clips[2] += plat.n_perturb
    return map(np.array, (v0, dv, dw, v_all, w_all)), clips


@pytest.mark.parametrize("c", [0.125, 0.5, 1.0])
def test_random_walk_statistics(plat, c):
    (v0, dv, dw, v_all, w_all), clips = run_episodes(plat, c, n_eps=200, seed=20000)
    a_v, a_w = PL.DV_MAX * c, PL.DW_MAX * c
    d0, k0 = ks_uniform(v0, 0, PL.V0_MAX * c)
    d1, k1 = ks_uniform(dv, -a_v, a_v)
    d2, k2 = ks_uniform(dw, -a_w, a_w)
    print(f"\n[c={c}] v0 KS {d0:.4f}<{k0:.4f} | dv KS {d1:.4f}<{k1:.4f} "
          f"mean {dv.mean():+.4f} | dw KS {d2:.4f}<{k2:.4f} | "
          f"clip rate v {clips[0]/clips[2]*100:.1f}% w {clips[1]/clips[2]*100:.1f}% | "
          f"v mean {v_all.mean():.2f} m/s, |w| p95 {np.degrees(np.percentile(np.abs(w_all),95)):.1f} deg/s")
    assert d0 < k0 and d1 < k1 and d2 < k2
    assert v_all.min() >= 0 and v_all.max() <= PL.V0_MAX * c + 1e-12
    assert np.abs(w_all).max() <= PL.W_CLIP + 1e-12


def test_initial_yaw_rate_zero(plat):
    rng = np.random.default_rng(0)
    for _ in range(20):
        plat.reset(rng, 1.0)
        assert plat.w == 0.0


def test_stationary_at_c0(plat):
    rng = np.random.default_rng(0)
    plat.reset(rng, 0.0, xy=(1, 2), psi=0.3)
    for _ in range(300):
        for _ in range(P.SUBSTEPS):
            plat.integrate()
        plat.perturb(rng)
    assert plat.v == 0 and plat.w == 0
    assert np.allclose(plat.pad_center(), [1, 2, PL.PAD_TOP_Z])


def test_straight_line_kinematics(plat):
    rng = np.random.default_rng(0)
    plat.reset(rng, 1.0, psi=np.radians(30))
    plat.v, plat.w = 5.0, 0.0
    for _ in range(1000):          # 10 s
        plat.integrate()
    expect = 50.0 * np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])
    assert np.allclose([plat.x, plat.y], expect, atol=1e-9)


def test_circle_kinematics_and_body_sync(plat):
    rng = np.random.default_rng(0)
    plat.reset(rng, 1.0)
    plat.v, plat.w = 4.0, np.radians(20)
    R = plat.v / plat.w
    n = int(round(2 * np.pi / plat.w / P.PHYSICS_DT))
    pts = []
    for _ in range(n):
        plat.integrate()
        pts.append([plat.x, plat.y])
    pts = np.array(pts)
    r = np.linalg.norm(pts - np.array([0.0, R]), axis=1)   # centre at (0, R)
    pos, orn = p.getBasePositionAndOrientation(plat.body, physicsClientId=plat.cid)
    print(f"\n[circle] R={R:.3f} m, radius error max {np.abs(r-R).max():.2e} m")
    assert np.abs(r - R).max() < 1e-9
    assert np.allclose(pos[:2], [plat.x, plat.y]) and pos[2] == pytest.approx(PL.PAD_TOP_Z / 2)
    assert p.getEulerFromQuaternion(orn)[2] == pytest.approx(
        np.arctan2(np.sin(plat.psi), np.cos(plat.psi)), abs=1e-9)


def test_pad_geometry(plat):
    rng = np.random.default_rng(0)
    plat.reset(rng, 0.0, xy=(3, -2))
    lo, hi = p.getAABB(plat.body, physicsClientId=plat.cid)
    size = np.array(hi) - np.array(lo)
    # PyBullet pads AABBs by a small collision margin
    assert np.allclose(size[:2], PL.PAD_SIZE, atol=0.05)
    assert hi[2] == pytest.approx(PL.PAD_TOP_Z, abs=0.03)
    assert np.allclose(plat.pad_center(), [3, -2, PL.PAD_TOP_Z])


def test_velocity_matches_finite_difference(plat):
    rng = np.random.default_rng(3)
    plat.reset(rng, 1.0)
    for _ in range(50):
        plat.perturb(rng)
    x0 = plat.pad_center().copy()
    vel = plat.velocity()
    plat.w = 0.0                   # straight over one physics step
    plat.integrate()
    assert np.allclose((plat.pad_center() - x0) / P.PHYSICS_DT, vel, atol=1e-9)


def test_determinism(plat):
    def traj(seed):
        rng = np.random.default_rng(seed)
        plat.reset(rng, 0.7)
        out = []
        for _ in range(100):
            for _ in range(P.SUBSTEPS):
                plat.integrate()
            plat.perturb(rng)
            out.append([plat.x, plat.y, plat.v, plat.w])
        return np.array(out)
    assert np.array_equal(traj(5), traj(5))
    assert not np.array_equal(traj(5), traj(6))
