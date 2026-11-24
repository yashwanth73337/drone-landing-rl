"""v7 S4q tests (24 Nov 2026): PPO A/B switch --reward-source true and --critic-rewarm.

  1. RewardV7("true"): progress from the TRUE pad position also when the camera sees nothing
     (no measurement); RewardV7("camera") pauses progress then. The blind mark (-0.05) is the
     same for both (it always follows the camera); end marks are the same
  2. the V7 world's true state has what the "true" reward reads (rel_pos_body, R)
  3. a tiny run: a PPO checkpoint is resumed into a NEW folder with --reward-source true
     --critic-rewarm 1: the first round after the resume is "rewarm" (actor frozen: the actor
     weights do not change in that round), then "ppo"; config records the switch; the
     source folder's last.pt is not changed
Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -s v7/tests/test_s4q_reward_source.py
"""
import csv
import hashlib
import json

import numpy as np
import pytest
import torch

from v6_drop.envs import reward as R6
from v6_drop.learn.student import Student
from v7.envs import reward_v7 as R7
from v7.scripts import train_ppo_v7 as P7


def ts(x, h):
    return dict(rel_pos_body=np.array([x, 0.0, -h]), R=np.eye(3))


# ---------------------------------------------------------------- 1. progress while blind
def test_true_progress_while_blind():
    cam, tru = R7.RewardV7("camera"), R7.RewardV7("true")
    cam.reset(None, ts(10.0, 5.0))
    tru.reset(None, ts(10.0, 5.0))
    r_c, t_c = cam.step(None, ts(8.0, 5.0))          # 2 m closer, camera sees nothing
    r_t, t_t = tru.step(None, ts(8.0, 5.0))
    assert t_c["progress"] == 0.0
    assert t_t["progress"] == pytest.approx(2.0)     # below the 45 deg slope: d counts fully
    assert t_c["visibility"] == t_t["visibility"] == -R6.VISIBILITY_PENALTY
    assert r_t - r_c == pytest.approx(2.0)
    _, e_c = cam.step(None, ts(0, 0), "tip_over", {})
    _, e_t = tru.step(None, ts(0, 0), "tip_over", {})
    assert e_c["end"] == e_t["end"] == -8.0


# ---------------------------------------------------------------- 2. the world provides it
def test_world_true_state_keys():
    env = P7.make_env_ppo("egl", reward_source="true")
    try:
        env.reset(seed=30000, options=dict(mode="straight"))
        s = env.world.true_state()
        assert s["rel_pos_body"].shape == (3,) and s["R"].shape == (3, 3)
        assert env.reward_fn.source == "true"
        _, r, _, _, info = env.step(dict(move=np.zeros(4, np.float32), drop=0))
        assert info["reward_terms"]["d"] is not None              # true geometry every step
    finally:
        env.close()


# ---------------------------------------------------------------- 3. resume into a new folder
def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def test_resume_true_with_rewarm(tmp_path):
    torch.manual_seed(0)
    init = tmp_path / "student.pt"
    torch.save(dict(model=Student(256).state_dict(), size=256), init)
    common = ["--workers", "1", "--steps", "16", "--critic-min", "1", "--critic-max", "1",
              "--exam-every", "1", "--exam-n", "1", "--guard", "0", "--device", "cpu",
              "--out-root", str(tmp_path / "runs"), "--mb-chunks", "1", "--chunk", "8"]
    P7.main(["--init", str(init), "--name", "a", "--rounds", "2"] + common)
    src = tmp_path / "runs" / "a" / "last.pt"
    before = md5(src)
    actor0 = torch.load(src, weights_only=False)["model"]
    P7.main(["--resume", str(src), "--name", "b", "--rounds", "3", "--reward-source", "true",
             "--critic-rewarm", "1"] + common)
    out = tmp_path / "runs" / "b"
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert [r["phase"] for r in rows] == ["rewarm"]
    ck = torch.load(out / "last.pt", weights_only=False)
    for k in actor0:                                         # actor frozen in the rewarm round
        assert torch.equal(ck["model"][k], actor0[k]), k
    cfg = json.load(open(out / "config.json"))
    assert cfg["reward_source"] == "true" and cfg["critic_rewarm"] == 1 and cfg["start_round"] == 2
    assert not (out / "exam_r0002.csv").exists()             # no exam in a rewarm round
    P7.main(["--resume", str(out / "last.pt"), "--name", "b", "--rounds", "4", "--reward-source", "true"]
            + common)
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert [r["phase"] for r in rows] == ["rewarm", "ppo"]
    assert (out / "exam_r0003.csv").exists()
    assert md5(src) == before                                # the source run is untouched
