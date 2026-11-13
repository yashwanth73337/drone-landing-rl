"""v6_drop A2 (13 Nov): ABLATION 2, PPO with no demonstrations (scripts/train_ppo.py
--random-init): PPO starts from an UNTRAINED student instead of the DAgger student.

Uses UNTRAINED networks and TinyRenderer, so it needs no results/.

Checks:
  1. the start rule: exactly one of --init / --resume / --random-init (none, or two at once,
     stops with a clear message before anything starts)
  2. --random-init end to end (1 worker): the start line names ablation 2; config says
     random_init; after the coach-alone round the actor is EXACTLY the untrained Student
     made with torch seed --seed (so the run repeats), and a different --seed gives a
     different start; a following PPO round (--resume) changes the actor
  3. --init still starts from the given checkpoint (old behaviour; config random_init False)

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_a2_random.py -v -s
"""
import json

import pytest
import torch

from v6_drop.learn import ppo as PPO
from v6_drop.learn.student import Student
from v6_drop.scripts import train_ppo as TP

COMMON = ["--workers", "1", "--steps", "16", "--chunk", "8", "--mb-chunks", "2", "--critic-min", "1",
          "--critic-max", "1", "--exam-every", "1", "--exam-n", "1", "--renderer", "tiny", "--device", "cpu",
          "--actor-warmup", "1", "--lr-actor", "1e-3", "--guard", "0"]


def untrained_student(seed):
    """The student --random-init must start from: the same construction order as train_ppo."""
    torch.manual_seed(seed)
    return PPO.ActorCritic(sigma_n=(0.2 / 6, 0.2 / 6, 0.05, 0.05)).student


# ---------------------------------------------------------------- 1. start rule
def test_exactly_one_start(tmp_path):
    init = tmp_path / "init.pt"
    torch.save(dict(model=Student().state_dict(), size=256, epoch=0), init)
    base = ["--name", "x", "--out-root", str(tmp_path), "--rounds", "1"] + COMMON
    for extra in ([], ["--init", str(init), "--random-init"],
                  ["--resume", str(init), "--random-init"], ["--init", str(init), "--resume", str(init)]):
        with pytest.raises(AssertionError, match="exactly one of --init / --resume / --random-init"):
            TP.main(extra + base)
    assert not (tmp_path / "x").exists()                    # stopped before anything was written
    print("\n[a2] none / two starts at once: all stopped with the clear message")


# ---------------------------------------------------------------- 2. random start end to end
def test_random_init_end_to_end(tmp_path, capsys):
    TP.main(["--random-init", "--seed", "3", "--name", "r", "--out-root", str(tmp_path), "--rounds", "1"] + COMMON)
    out = capsys.readouterr().out
    run = tmp_path / "r"
    cfg = json.load(open(run / "config.json"))
    ck = torch.load(run / "last.pt", weights_only=False)
    assert "[ppo] actor RANDOM (untrained, no demos = ablation 2; torch seed 3); critic NEW" in out
    assert cfg["random_init"] is True and cfg["init"] is None and cfg["seed"] == 3
    assert ck["ppo_start"] is None and ck["round"] == 0       # one coach-alone round, actor frozen
    ref3, ref4 = untrained_student(3).state_dict(), untrained_student(4).state_dict()
    for k, v in ref3.items():
        assert torch.equal(ck["model"][k], v), k            # exactly the seed-3 untrained student
    differ = [k for k in ref3 if not torch.equal(ref3[k], ref4[k])]
    assert "memory.weight_hh_l0" in differ                  # another seed = another start
    p0 = torch.sigmoid(ck["model"]["drop_head.2.bias"]).item()
    print(f"\n[a2] random start = untrained student of torch seed 3 ({len(ref3)} tensors equal); "
          f"seed 4 differs in {len(differ)} tensors; starting drop chance {100 * p0:.2f} %")
    assert abs(p0 - 0.005) < 1e-6                           # the drop head's built-in 0.5 % start

    TP.main(["--resume", str(run / "last.pt"), "--seed", "3", "--name", "r", "--out-root", str(tmp_path),
             "--rounds", "2"] + COMMON)
    ck2 = torch.load(run / "last.pt", weights_only=False)
    changed = [k for k in ref3 if not torch.equal(ck2["model"][k], ref3[k])]
    print(f"[a2] after one PPO round: {len(changed)} actor tensors changed")
    assert ck2["ppo_start"] == 1 and "memory.weight_hh_l0" in changed and "move_head.0.weight" in changed


# ---------------------------------------------------------------- 3. --init unchanged
def test_init_still_loads_checkpoint(tmp_path, capsys):
    torch.manual_seed(123)
    s = Student()
    init = tmp_path / "init.pt"
    torch.save(dict(model=s.state_dict(), size=256, epoch=0), init)
    TP.main(["--init", str(init), "--name", "i", "--out-root", str(tmp_path), "--rounds", "1"] + COMMON)
    out = capsys.readouterr().out
    cfg = json.load(open(tmp_path / "i" / "config.json"))
    ck = torch.load(tmp_path / "i" / "last.pt", weights_only=False)
    assert "[ppo] actor from" in out and "RANDOM" not in out and cfg["random_init"] is False
    for k, v in s.state_dict().items():
        assert torch.equal(ck["model"][k], v), k
