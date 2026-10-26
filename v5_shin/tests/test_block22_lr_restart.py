"""Block 22: learning-rate restart on resume (SPEC D28). CPU unit tests, no training runs.

Run:  python -m pytest v5_shin/tests/test_block22_lr_restart.py -v
"""
import pytest
import torch

from v5_shin.tests.test_block12_ppo import small_trainer


def _lr_at(update, total, start, warmup=0):
    torch.manual_seed(0)
    tr = small_trainer(mode="privileged", n_steps=8, seq_len=4, lr=1e-3, lr_warmup=warmup)
    try:
        tr.update, tr.total_updates, tr.cfg.lr_schedule_start = update, total, start
        buf, _ = tr.collect()
        return tr.learn(buf)["lr"]
    finally:
        tr.venv.close()


def test_default_schedule_over_the_whole_history():
    assert _lr_at(5, 10, None) == pytest.approx(0.5e-3)


def test_restart_starts_at_full_lr_and_decays():
    assert _lr_at(5, 10, 5) == pytest.approx(1e-3)
    assert _lr_at(8, 10, 5) == pytest.approx(1e-3 * (1 - 3 / 5))


def test_train_script_lr_restart_end_to_end():
    import csv, json, os, shutil, subprocess, sys
    name = "_pytest_d28"
    run = os.path.join(os.path.dirname(__file__), "..", "runs", name)
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    shutil.rmtree(run, ignore_errors=True)
    base = [sys.executable, "-m", "v5_shin.scripts.train_ppo", "--mode", "privileged", "--name", name,
            "--n-envs", "2", "--n-steps", "16", "--seq-len", "4", "--minibatches", "2", "--renderer", "tiny",
            "--device", "cpu", "--ckpt-every", "1", "--lr", "1e-3"]
    try:
        r1 = subprocess.run(base + ["--total-steps", "64"], capture_output=True, text=True, timeout=600, cwd=root)
        assert r1.returncode == 0, r1.stderr[-3000:]
        r2 = subprocess.run(base + ["--total-steps", "128", "--resume", "--lr-restart"],
                            capture_output=True, text=True, timeout=600, cwd=root)
        assert r2.returncode == 0, r2.stderr[-3000:]
        rows = list(csv.DictReader(open(os.path.join(run, "updates.csv"))))
        assert len(rows) == 4
        assert float(rows[2]["lr"]) == pytest.approx(1e-3)          # restarted at the full LR
        assert float(rows[3]["lr"]) == pytest.approx(0.5e-3)
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_warmup_ramps_the_lr():
    """D29: with restart at update 5 and warm-up 4: k = 0..3 -> 1/4, 2/4, 3/4, 4/4 of the schedule."""
    total, s0 = 25, 5
    for k, w in [(0, 0.25), (1, 0.5), (3, 1.0), (6, 1.0)]:
        lin = 1 - k / (total - s0)
        assert _lr_at(s0 + k, total, s0, warmup=4) == pytest.approx(1e-3 * lin * w)


def test_warmup_off_by_default():
    assert _lr_at(5, 10, 5) == pytest.approx(1e-3)
