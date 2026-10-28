"""Block 25: PPO fine-tuning resumed from a distill.py checkpoint (SPEC D32). CPU, tiny runs.

Run:  python -m pytest v5_shin/tests/test_block25_resume_from_distill.py -v
"""
import csv
import json
import os
import shutil
import subprocess
import sys

import torch

from v5_shin.policies.shin_policy import ShinPolicy

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def test_train_ppo_resumes_from_a_distill_checkpoint():
    name = "_pytest_d32"
    run = os.path.join(RUNS, name)
    shutil.rmtree(run, ignore_errors=True)
    os.makedirs(run)
    try:
        # a distill-style checkpoint: no curriculum, update counter of its own, large global_step
        pol = ShinPolicy("privileged")
        opt = torch.optim.Adam(pol.parameters(), lr=1e-4, eps=1e-5)
        torch.save(dict(policy=pol.state_dict(), opt=opt.state_dict(), update=7, global_step=3200,
                        distill_steps=224, extra={}), os.path.join(run, "latest.pt"))
        cmd = [sys.executable, "-m", "v5_shin.scripts.train_ppo", "--mode", "privileged", "--name", name,
               "--reward", "v5", "--start-level", "80", "--total-steps", "3264", "--n-envs", "2",
               "--n-steps", "16", "--seq-len", "4", "--minibatches", "2", "--renderer", "tiny",
               "--device", "cpu", "--resume", "--lr", "1e-3", "--lr-restart"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=ROOT)
        assert res.returncode == 0, res.stderr[-3000:]
        assert "[D32]" in res.stdout
        rows = list(csv.DictReader(open(os.path.join(run, "updates.csv"))))
        # 3200 / 32 = 100 -> updates 101 and 102 only (not 95 more from update 7)
        assert [int(r["update"]) for r in rows] == [101, 102]
        assert all(int(r["level"]) == 80 for r in rows)
        assert float(rows[0]["lr"]) == 1e-3                      # restart at the aligned update
        cfg = json.load(open(os.path.join(run, "config.json")))
        assert cfg["ppo"]["lr_schedule_start"] == 100
    finally:
        shutil.rmtree(run, ignore_errors=True)
