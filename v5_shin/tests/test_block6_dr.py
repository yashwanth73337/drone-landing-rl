"""Block 6: sensor noise + Table II domain randomisation.

Statistics are checked against the specified distributions; the physics of the
external force and torque is checked against closed-form steady states of the
Lee controller. Visual DR is checked by effect on rendered images. Marker
detectability under full DR is reported (diagnostic, loose bound).

Run:  python -m pytest v5_shin/tests/test_block6_dr.py -v -s
"""
import numpy as np
import pybullet as p
import pytest

from v5_shin.envs import dr as DR
from v5_shin.envs import ground as G
from v5_shin.envs import lmf2_params as P
from v5_shin.envs import pad_marker as PM
from v5_shin.envs.dr import DRConfig
from v5_shin.envs.landing_sim import LandingSim
from v5_shin.envs.platform import PAD_TOP_Z


def ks_uniform(x, lo, hi):
    x = np.sort((np.asarray(x) - lo) / (hi - lo))
    n = len(x)
    cdf = np.arange(1, n + 1) / n
    return max(np.max(cdf - x), np.max(x - (cdf - 1 / n))), 1.63 / np.sqrt(n)


def euler_R(r, pch, y):
    q = p.getQuaternionFromEuler([r, pch, y])
    return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3), np.array(q)


def quat_to_rotvec(q):
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    th = 2 * np.arctan2(s, q[3])
    return np.zeros(3) if s < 1e-12 else q[:3] / s * th


# ---------------------------------------------------------------- sensors
def test_velocity_noise():
    R, q = euler_R(0.2, -0.1, 1.0)
    st = dict(R=R, quat=q, v=np.array([3.0, -1.0, 0.5]))
    rng = np.random.default_rng(20000)
    clean = DR.measure(st, rng, noise=False)[:3]
    assert np.allclose(clean, R.T @ st["v"])
    e = np.array([DR.measure(st, rng)[:3] for _ in range(20000)]) - clean
    c = np.corrcoef(e.T)
    print(f"\n[vel noise] mean {np.round(e.mean(0), 4)} std {np.round(e.std(0), 4)} "
          f"max |corr| {np.abs(c - np.eye(3)).max():.3f}")
    assert np.abs(e.mean(0)).max() < 0.002
    assert np.allclose(e.std(0), DR.VEL_NOISE_STD, rtol=0.03)
    assert np.abs(c - np.eye(3)).max() < 0.03


def test_attitude_noise():
    R, q = euler_R(0.3, 0.2, -2.5)
    st = dict(R=R, quat=q, v=np.zeros(3))
    rng = np.random.default_rng(20001)
    clean = DR.measure(st, rng, noise=False)[3:]
    assert clean[3] >= 0 and np.isclose(abs(clean @ q), 1.0)     # same rotation, canonical sign
    qinv = np.array([-q[0], -q[1], -q[2], q[3]])
    errs = []
    for _ in range(20000):
        qn = DR.measure(st, rng)[3:]
        assert qn[3] >= 0 and np.isclose(np.linalg.norm(qn), 1.0)
        errs.append(quat_to_rotvec(DR.quat_mul(qinv, qn)))        # body-frame error
    e = np.degrees(np.array(errs))
    print(f"\n[att noise] std per axis {np.round(e.std(0), 4)} deg (spec 0.5)")
    assert np.abs(e.mean(0)).max() < 0.02
    assert np.allclose(e.std(0), 0.5, rtol=0.03)


# ---------------------------------------------------------------- physical DR
def test_force_steady_state():
    """Constant world force F along x with a hover command: Lee velocity loop settles at
    v_ss = F / (m K_vx). Checks that F is applied in newtons, in the world frame."""
    sim = LandingSim(seed=1, gains_mode="nominal", dr=DRConfig.off())
    try:
        sim.reset(c=0.0, spawn=dict(pos=np.array([0, 0, PAD_TOP_Z + 6]), yaw=0.7, psi_plat=0.0))
        sim.quad.ext_force[:] = [DR.F_EXT_MAX, 0, 0]
        for _ in range(60):
            sim.step([0, 0, 0, 0])
        v = sim.quad.state()["v"]
        expect = DR.F_EXT_MAX / (P.MASS * sim.gains["K_vel"][0])
        print(f"\n[force] v_ss {np.round(v, 4)}  expect vx {expect:.4f}")
        assert v[0] == pytest.approx(expect, rel=0.05) and abs(v[1]) < 0.01
    finally:
        sim.close()


def test_torque_steady_state():
    """Constant body torque M about z with yaw-rate command 0: the controller has no yaw-angle
    hold, so it settles at w_z = M / K_angvel_z."""
    sim = LandingSim(seed=1, gains_mode="nominal", dr=DRConfig.off())
    try:
        sim.reset(c=0.0, spawn=dict(pos=np.array([0, 0, PAD_TOP_Z + 6]), yaw=0.0, psi_plat=0.0))
        sim.quad.ext_torque[:] = [0, 0, DR.M_EXT_MAX]
        for _ in range(60):
            sim.step([0, 0, 0, 0])
        wz = sim.quad.state()["w_body"][2]
        expect = DR.M_EXT_MAX / sim.gains["K_angvel"][2]
        print(f"\n[torque] w_z {wz:.4f} rad/s  expect {expect:.4f}")
        assert wz == pytest.approx(expect, rel=0.05)
    finally:
        sim.close()


def test_disturbance_sampling_and_hold():
    sim = LandingSim(seed=3, dr=DRConfig(forces=True, initial_state=False, visual=False,
                                        sensor_noise=False))
    try:
        sim.reset(c=0.0, spawn=dict(pos=np.array([0, 0, PAD_TOP_Z + 6]), yaw=0.0, psi_plat=0.0))
        F, M = [], []
        orig = sim.quad.apply_control
        seen = []
        sim.quad.apply_control = lambda cmd: (seen.append(sim.quad.ext_force.copy()), orig(cmd))
        for _ in range(250):
            sim.step([0, 0, 0, 0])
            F.append(sim.quad.ext_force.copy())
            M.append(sim.quad.ext_torque.copy())
        F, M, seen = np.array(F), np.array(M), np.array(seen)
        held = all(np.allclose(seen[k * P.SUBSTEPS:(k + 1) * P.SUBSTEPS], F[k]) for k in range(len(F)))
        dF = ks_uniform(F.ravel(), -DR.F_EXT_MAX, DR.F_EXT_MAX)
        dM = ks_uniform(M.ravel(), -DR.M_EXT_MAX, DR.M_EXT_MAX)
        print(f"\n[disturbance] held over substeps {held}; KS F {dF[0]:.3f}<{dF[1]:.3f}, "
              f"M {dM[0]:.3f}<{dM[1]:.3f}")
        assert held and dF[0] < dF[1] and dM[0] < dM[1]
        assert len(np.unique(F[:, 0])) == len(F)          # fresh sample every policy step
    finally:
        sim.close()


def test_initial_state_dr():
    sim = LandingSim(seed=20002, dr=DRConfig(forces=False, initial_state=True, visual=False,
                                            sensor_noise=False))
    try:
        V, W, err = [], [], 0.0
        for _ in range(500):
            info = sim.reset(c=1.0)
            s = sim.quad.state()
            err = max(err, np.abs(s["v"] - info["v0"]).max(), np.abs(s["w_body"] - info["w0"]).max())
            V.append(info["v0"])
            W.append(info["w0"])
        V, W = np.array(V), np.array(W)
        kv = ks_uniform(V.ravel(), -DR.V0_MAX, DR.V0_MAX)
        kw = ks_uniform(W.ravel(), -DR.W0_MAX, DR.W0_MAX)
        print(f"\n[init] state==sample (max err {err:.2e}); KS v0 {kv[0]:.3f}<{kv[1]:.3f}, "
              f"w0 {kw[0]:.3f}<{kw[1]:.3f}")
        assert err < 1e-9 and kv[0] < kv[1] and kw[0] < kw[1]
    finally:
        sim.close()


def test_dr_flags_do_not_change_spawns():
    """Independent RNG streams: the same seed gives the same spawns with DR on or off."""
    a = LandingSim(seed=77, dr=DRConfig())
    b = LandingSim(seed=77, dr=DRConfig.off())
    try:
        for _ in range(20):
            sa, sb = a.reset(c=1.0)["spawn"], b.reset(c=1.0)["spawn"]
            assert np.allclose(sa["pos"], sb["pos"]) and sa["psi0"] == sb["psi0"]
            for _ in range(5):
                a.step([0, 0, 0, 0])
                b.step([0, 0, 0, 0])
            assert a.plat.v == b.plat.v and a.plat.w == b.plat.w
    finally:
        a.close()
        b.close()


# ---------------------------------------------------------------- visual DR
def test_textures_generated():
    import cv2
    paths = G.ensure_textures()
    assert len(paths) == DR.TEX_IDS
    hashes = set()
    for i, path in enumerate(paths):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert img.ndim == 3 and img.shape[2] == 3 and img.shape[1] % 4 == 0
        hashes.add(img.tobytes().__hash__())
    assert len(hashes) == DR.TEX_IDS
    assert np.array_equal(G.make_texture(7), G.make_texture(7))


def _egl_ok():
    from v5_shin.envs.quad import make_client
    try:
        p.disconnect(make_client(egl=True))
        return True
    except Exception:
        return False


RENDERERS = ["tiny", pytest.param("egl", marks=pytest.mark.skipif(
    not _egl_ok(), reason="EGL plugin unavailable"))]


@pytest.fixture(scope="module", params=RENDERERS)
def vsim(request):
    """Both renderers: the one-mesh-per-file EGL bug (Block 6) was invisible under tiny."""
    r = request.param
    s = LandingSim(seed=5, renderer=r, egl=(r == "egl"),
                   dr=DRConfig(forces=False, initial_state=False, visual=True, sensor_noise=False))
    s.reset(c=0.0, spawn=dict(pos=np.array([-40.0, 0, PAD_TOP_Z + 6]), yaw=0.0, psi_plat=0.0))
    yield s
    s.close()


def _ground_img(sim, **kw):
    s = sim.quad.state()          # platform is 40 m ahead: out of this view region
    return sim.cam.render(s["R"], s["pos"], light_direction=[0, 0, 1], **kw)[200:320].astype(float)


def test_only_one_ground_visual_active(vsim):
    zs = [p.getBasePositionAndOrientation(b, physicsClientId=vsim.cid)[0][2]
          for b in vsim.ground.visuals]
    assert sum(abs(z) < 1e-9 for z in zs) == 1 and len(zs) == len(G.SCALES)


def test_texture_scale_changes_pattern_size(vsim):
    """Nadir view from 5 m: the 512 px row spans 10 m of ground; checker cells are
    0.25*s m, so the row has ~10/(0.25 s) transitions. Every scale level is checked."""
    path = G.checker_texture(G.os.path.join(G.ASSETS, "ground_checker.png"))
    q = p.getQuaternionFromEuler([0, np.radians(30), 0])          # camera axis -> nadir
    R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
    worst = 0.0
    for sc in G.SCALES:
        vsim.ground.set_appearance(0, sc, 1.0)
        vsim.ground.set_texture(path)
        row = vsim.cam.render(R, np.array([-40.0, 0, 5.0]), light_direction=[0, 0, 1])[160]
        n = int(np.sum(np.abs(np.diff(row > row.mean())) > 0))
        worst = max(worst, abs(n / (10 / (0.25 * sc)) - 1))
    print(f"\n[scale {vsim.cam.renderer}] worst transition-count error over {len(G.SCALES)} "
          f"scales: {worst:.1%}")
    assert worst < 0.06


def test_brightness_and_rgb_scaling(vsim):
    vsim.ground.set_appearance(3, 1.0, 1.0)
    full = _ground_img(vsim).mean()
    vsim.ground.set_appearance(3, 1.0, 0.5)
    half = _ground_img(vsim).mean()
    vsim.ground.set_appearance(3, 1.0, 1.0)
    same = _ground_img(vsim, rgb_scale=np.ones(3))
    ref = _ground_img(vsim)
    halved = _ground_img(vsim, rgb_scale=np.full(3, 0.5)).mean()
    print(f"\n[brightness] mean {full:.1f} -> {half:.1f} (ratio {half/full:.3f}); "
          f"rgb 0.5 ratio {halved/ref.mean():.3f}; rgb 1.0 max diff {np.abs(same-ref).max():.0f}")
    assert half / full == pytest.approx(0.5, abs=0.05)
    assert halved / ref.mean() == pytest.approx(0.5, abs=0.02)
    assert np.abs(same - ref).max() <= 1


def test_light_direction_changes_image(vsim):
    s = vsim.quad.state()
    # 45 vs 90 deg (NOT 45 vs 135: those are mirror images and light a flat ground equally)
    imgs = [vsim.cam.render(s["R"], s["pos"], light_direction=[np.cos(np.radians(a)), 0,
                                                              np.sin(np.radians(a))])
            for a in (45, 90)]
    d = np.abs(imgs[0].astype(float) - imgs[1]).mean()
    print(f"\n[light {vsim.cam.renderer}] mean |diff| 45 vs 90 deg: {d:.2f}")
    assert d > 1.0


def test_marker_detectable_under_full_dr():
    """Diagnostic: the ArUco marker should usually stay detectable at spawn under full
    visual DR (the policy never uses detections; baseline H does). Loose bound."""
    sim = LandingSim(seed=20003)
    try:
        det, wrong = 0, 0
        n = 150
        for _ in range(n):
            sim.reset(c=1.0)
            _, ids, _ = PM.detector().detectMarkers(sim.render())
            if ids is not None:
                det += int(PM.MARKER_ID in ids.ravel())
                wrong += int(any(i != PM.MARKER_ID for i in ids.ravel()))
        print(f"\n[DR detect] marker detected at spawn {det}/{n} ({det/n:.1%}); false ids {wrong}")
        assert det / n >= 0.8 and wrong == 0
    finally:
        sim.close()
