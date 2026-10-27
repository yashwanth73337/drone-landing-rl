"""Block 24: teacher-student distillation with DAgger (SPEC D31). CPU only, no real training.

Run:  python -m pytest v5_shin/tests/test_block24_distill.py -v
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from v5_shin.policies.shin_policy import ShinPolicy
from v5_shin.scripts import distill as D

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def test_beta_schedule():
    assert D.beta_at(0, 1.0, 100) == 1.0
    assert D.beta_at(50, 1.0, 100) == pytest.approx(0.5)
    assert D.beta_at(100, 1.0, 100) == 0.0
    assert D.beta_at(500, 1.0, 100) == 0.0
    assert D.beta_at(10, 0.5, 0) == 0.0


def test_env_levels_spread_and_fixed():
    lv = D.env_levels(16)
    assert sorted(set(lv)) == [l / 80 for l in range(10, 81, 10)] and lv.count(1.0) == 2
    assert D.env_levels(3, c=0.5) == [0.5] * 3


def small(beta0=1.0, beta_steps=10 ** 6, n_envs=2, n_steps=8, seq_len=4):
    torch.manual_seed(0)
    venv = D.make_venv(n_envs, seed=3, renderer="tiny", hold_practice=0.25, hold_vz_down=5.0, sync=True)
    dz = D.Distiller(venv, ShinPolicy("vision"), ShinPolicy("privileged"), device="cpu", n_steps=n_steps,
                     seq_len=seq_len, epochs=1, minibatches=1, beta0=beta0, beta_steps=beta_steps)
    dz.reset(seeds=range(n_envs))
    return dz


def test_teacher_label_is_the_clipped_privileged_actor():
    dz = small()
    try:
        tob = dz._tensors(dz.obs)
        mu = dz.teacher_mu(tob)
        ref, _, _ = dz.teacher.actor_seq({"u": tob["u"][None], "s_rel": tob["s_rel"][None]}, None, None)
        assert torch.allclose(mu, ref[0].clamp(-1, 1))
        assert mu.abs().max() <= 1.0
    finally:
        dz.venv.close()


@pytest.mark.parametrize("beta0,expect", [(1.0, 1.0), (0.0, 0.0)])
def test_who_flies(beta0, expect):
    dz = small(beta0=beta0)
    try:
        buf, _ = dz.collect()
        assert float(buf["teacher_flew"].mean()) == expect
        assert buf["image"].shape[:2] == (8, 2) and buf["h0"].shape[0] == 2
    finally:
        dz.venv.close()


def test_bc_loss_alone_learns():
    """Copying the teacher on a fixed batch: the BC loss falls well below its start."""
    dz = small(n_steps=16)
    try:
        buf, _ = dz.collect()
        chunks = [(k, b) for k in range(4) for b in range(2)]
        obs, starts, state = dz._gather(buf, chunks)
        opt = torch.optim.Adam(dz.student.parameters(), lr=1e-3)
        losses = []
        for _ in range(60):
            mu, _, _ = dz.student.actor_seq(obs, state, starts)
            loss = ((mu - dz.teacher_mu(obs)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"\n[BC] {losses[0]:.4f} -> {losses[-1]:.4f}")
        assert losses[-1] < 0.2 * losses[0]
    finally:
        dz.venv.close()


def test_teacher_is_frozen():
    dz = small()
    try:
        before = {k: v.clone() for k, v in dz.teacher.state_dict().items()}
        buf, _ = dz.collect()
        dz.learn(buf)
        assert all(torch.equal(before[k], v) for k, v in dz.teacher.state_dict().items())
        assert dz.update == 1 and dz.steps == 16
    finally:
        dz.venv.close()


def test_distill_cli_end_to_end_and_evaluate_can_load():
    teacher, name = "_pytest_teacher", "_pytest_distill"
    tdir, run = os.path.join(RUNS, teacher), os.path.join(RUNS, name)
    shutil.rmtree(tdir, ignore_errors=True); shutil.rmtree(run, ignore_errors=True)
    os.makedirs(tdir)
    try:
        torch.save(dict(policy=ShinPolicy("privileged").state_dict(), update=0, global_step=0),
                   os.path.join(tdir, "latest.pt"))
        json.dump(dict(mode="privileged"), open(os.path.join(tdir, "config.json"), "w"))
        cmd = [sys.executable, "-m", "v5_shin.scripts.distill", "--name", name, "--teacher", teacher,
               "--student-init", "scratch", "--total-steps", "64", "--n-envs", "2", "--n-steps", "16",
               "--seq-len", "4", "--minibatches", "2", "--epochs", "1", "--device", "cpu", "--sync",
               "--ckpt-every", "1", "--beta-steps", "32"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=ROOT)
        assert res.returncode == 0, res.stderr[-3000:]
        import csv
        rows = list(csv.DictReader(open(os.path.join(run, "updates.csv"))))
        assert len(rows) == 2 and float(rows[0]["beta"]) == 1.0 and float(rows[1]["beta"]) == 0.0
        cfg = json.load(open(os.path.join(run, "config.json")))
        assert cfg["mode"] == "vision" and cfg["hold_vz_down"] == 5.0 and cfg["method"] == "distill_dagger"
        d = torch.load(os.path.join(run, "latest.pt"), map_location="cpu", weights_only=False)
        pol = ShinPolicy(cfg["mode"]); pol.load_state_dict(d["policy"])     # what evaluate.py does
        assert d["global_step"] == 64 and d["update"] == 2
    finally:
        shutil.rmtree(tdir, ignore_errors=True); shutil.rmtree(run, ignore_errors=True)
