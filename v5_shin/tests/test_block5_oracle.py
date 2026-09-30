"""Block 5: the task is solvable with the true state, and random actions move the drone.

Full evaluation (reported in SPEC §12): python -m v5_shin.scripts.eval_oracle
This test runs a smaller pinned subset (seed base 9000) to keep CI fast.

Run:  python -m pytest v5_shin/tests/test_block5_oracle.py -v -s
"""
import numpy as np
import pytest

from v5_shin.envs.dr import DRConfig
from v5_shin.envs.landing_sim import LandingSim
from v5_shin.envs.platform import PAD_TOP_Z
from v5_shin.scripts.eval_oracle import run_episodes, summarize


@pytest.fixture(scope="module")
def sim():
    """Full Table II DR (the paper's training condition)."""
    s = LandingSim(seed=9000)
    yield s
    s.close()


@pytest.fixture(scope="module")
def sim_bare():
    s = LandingSim(seed=9000, dr=DRConfig.off())
    yield s
    s.close()


@pytest.mark.parametrize("c,min_rate", [(0.0, 1.0), (1.0, 0.9)])
def test_oracle_lands(sim, c, min_rate):
    summ = summarize(run_episodes("oracle", c=c, episodes=50, seed_base=9000, sim=sim))
    print(f"\n[oracle c={c}] success {summ['success_rate']:.2f} strict {summ['strict_success_rate']:.2f} "
          f"timeouts {summ['timeout']} crashes {summ['crash_ground'] + summ['crash_platform']} "
          f"tilt {summ['tilt']} drift {summ['drift']}")
    assert summ["success_rate"] >= min_rate
    assert summ["crash_ground"] + summ["crash_platform"] + summ["tilt"] + summ["drift"] == 0


def test_random_actions_move_the_drone(sim):
    """V4 regression: its position-offset actions let white-noise exploration go nowhere
    (0.1 m in 300 steps). Velocity commands must integrate into real motion."""
    rows = run_episodes("random", c=0.0, episodes=20, seed_base=9000, hold=1, sim=sim)
    summ = summarize(rows)
    print(f"\n[random h1] median xy travel {summ['net_xy_travel_median']:.2f} m, "
          f"median min height {summ['min_height_median']:.2f} m, success {summ['success']}")
    assert summ["net_xy_travel_median"] > 1.0
    assert summ["min_height_median"] < 1.0      # random exploration reaches pad height


def test_impact_velocity_is_pre_contact(sim_bare):
    """touchdown rel_vel must be the impact velocity, not the post-contact one."""
    sim = sim_bare
    sim.reset(c=0.0, spawn=dict(pos=np.array([0, 0, PAD_TOP_Z + 2.0]), yaw=0.0, psi_plat=0.0))
    for _ in range(200):
        te, tr, info = sim.step([0, 0, -1.0, 0])
        if te or tr:
            break
    print(f"\n[impact] outcome {info['outcome']} rel_vz {info['rel_vel'][2]:.3f} m/s")
    assert info["outcome"] == "success"
    assert info["rel_vel"][2] == pytest.approx(-1.0, abs=0.05)
