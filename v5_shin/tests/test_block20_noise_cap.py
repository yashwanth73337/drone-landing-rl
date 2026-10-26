"""Block 20: cap on the lateral exploration noise (SPEC D26). CPU unit tests, no training runs.

Run:  python -m pytest v5_shin/tests/test_block20_noise_cap.py -v
"""
import torch

from v5_shin.tests.test_block12_ppo import small_trainer


def test_cap_applies_at_start_and_after_updates():
    torch.manual_seed(0)
    tr = small_trainer(mode="privileged", n_steps=8, seq_len=4, log_std_max_xy=-2.0)
    try:
        ls = tr.policy.log_std.detach()
        assert torch.all(ls[:2] <= -2.0 + 1e-7)
        assert torch.allclose(ls[2:], torch.full((2,), -0.5))          # vz, yaw untouched
        with torch.no_grad():
            tr.policy.log_std[:2] = 0.0                                 # push above the cap
        buf, _ = tr.collect()
        tr.learn(buf)
        assert torch.all(tr.policy.log_std.detach()[:2] <= -2.0 + 1e-7)
    finally:
        tr.venv.close()


def test_no_cap_by_default():
    torch.manual_seed(0)
    tr = small_trainer(mode="privileged", n_steps=8, seq_len=4)
    try:
        assert tr.cfg.log_std_max_xy is None
        assert torch.allclose(tr.policy.log_std.detach(), torch.full((4,), -0.5))
    finally:
        tr.venv.close()


def test_train_script_cap_end_to_end():
    import json, os, shutil, subprocess, sys
    name = "_pytest_d26"
    run = os.path.join(os.path.dirname(__file__), "..", "runs", name)
    shutil.rmtree(run, ignore_errors=True)
    try:
        cmd = [sys.executable, "-m", "v5_shin.scripts.train_ppo", "--mode", "privileged", "--name", name,
               "--reward", "v4", "--log-std-max-xy", "-2.0", "--total-steps", "64", "--n-envs", "2",
               "--n-steps", "16", "--seq-len", "4", "--minibatches", "2", "--renderer", "tiny",
               "--device", "cpu", "--ckpt-every", "1"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                             cwd=os.path.join(os.path.dirname(__file__), "..", ".."))
        assert res.returncode == 0, res.stderr[-3000:]
        cfg = json.load(open(os.path.join(run, "config.json")))
        assert cfg["ppo"]["log_std_max_xy"] == -2.0 and cfg["log_std_max_xy"] == -2.0
        d = torch.load(os.path.join(run, "latest.pt"), map_location="cpu", weights_only=False)
        assert torch.all(d["policy"]["log_std"][:2] <= -2.0 + 1e-7)
    finally:
        shutil.rmtree(run, ignore_errors=True)
