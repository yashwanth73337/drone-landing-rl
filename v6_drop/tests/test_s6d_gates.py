"""v6_drop Step 6d: the shared drop rules (experts/drop_gates.py), decided 30 Oct.

Checks:
  1. the 'teacher' gate has exactly the teacher's original constants
  2. predicted impact and the hold rule on simple numbers; safe30m is safe30 with 0.8 m/s
  3. Teacher(gate='safe30m') flying (but never dropping): its per-step drop labels equal the
     OFFLINE relabelling from the saved true state; it says 'drop' earlier than the old gate
  4. Demos(..., drop_gate=...): relabelling recorded demos with the 'teacher' gate gives back
     EXACTLY the recorded labels; 'safe30m' changes only drop labels (moves untouched)
  5. train_bc --drop-gate safe30m runs and records the label counts; dagger_collect
     --drop-gate makes the teacher label with that gate (shown in the summary)

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s6d_gates.py -v -s
"""
import os
import sys

import numpy as np
import pytest
import torch

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.experts import drop_gates as G
from v6_drop.experts import teacher as T
from v6_drop.experts.troubles import Troubles
from v6_drop.learn import dataset as D
from v6_drop.learn.student import Student
from v6_drop.scripts import dagger_collect as DC
from v6_drop.scripts import record_demos as RD


def test_teacher_gate_constants():
    g = G.GATES["teacher"]
    assert (g["d"], g["hc"], g["hh"], g["vxy"], g["vz"], g["hold"]) == \
        (T.D_DROP, T.H_DROP, T.DH_DROP, T.V_REL_DROP, T.V_REL_DROP, T.HOLD_STEPS)
    assert g["tilt"] == T.TILT_DROP and g["impact"] is None
    assert T.Teacher().gate_cfg is g


def test_numbers_and_hold():
    assert G.predicted_impact(-0.18, 0.03) == pytest.approx(np.sqrt(0.18 ** 2 + 2 * P.G * 0.03))
    assert G.predicted_impact(0.1, -0.01) == 0.0
    a, b = G.GATES["safe30"], G.GATES["safe30m"]
    assert {k: v for k, v in a.items() if k != "impact"} == {k: v for k, v in b.items() if k != "impact"}
    assert (a["impact"], b["impact"]) == (0.9, 0.8)
    assert abs(b["hc"] - b["hh"] - 0.015) < 1e-12 and abs(b["hc"] + b["hh"] - 0.06) < 1e-12
    # safe30m at 3 cm: ok while sinking 0.2, not at 0.5 (impact 0.92 > 0.8)
    assert G.ok("safe30m", 0.1, 0.03, 0.1, -0.2, 0.02) and not G.ok("safe30m", 0.1, 0.03, 0.1, -0.5, 0.02)
    assert not G.ok("safe30m", 0.31, 0.03, 0.1, -0.2, 0.02)
    arr = G.ok("safe30m", np.array([0.1, 0.4]), np.array([0.03, 0.03]), np.zeros(2), np.zeros(2), np.zeros(2))
    assert arr.tolist() == [True, False]
    assert G.held(np.array([1, 1, 0, 1, 1, 1, 1], bool), 3).tolist() == [0, 0, 0, 0, 0, 1, 1]


@pytest.fixture(scope="module")
def env():
    e = E.LandingEnv(renderer="tiny", seed=0, headless=True)
    yield e
    e.close()


def test_teacher_safe30m_labels_match_offline(env):
    total = 0
    for seed in (100001, 100002):
        obs, _ = env.reset(seed=seed)
        new, old = T.Teacher(gate="safe30m"), T.Teacher()
        priv, state, lab_new, lab_old = [], [], [], []
        while True:
            a, _ = new.act(env)
            b, _ = old.act(env)
            priv.append(obs["priv"]); state.append(obs["state"])
            lab_new.append(a["drop"]); lab_old.append(b["drop"])
            obs, _, term, trunc, info = env.step({"move": a["move"], "drop": 0})
            if term or trunc:
                break
        n = len(lab_new)
        off = D.relabel_drop(np.stack(priv), np.stack(state), [0], [n], "safe30m")
        mism = int(np.sum(off != np.array(lab_new)))
        first_new = lab_new.index(1) if 1 in lab_new else None
        first_old = lab_old.index(1) if 1 in lab_old else None
        print(f"\n[gates] seed {seed}: {n} steps, safe30m labels {sum(lab_new)}, old-gate labels {sum(lab_old)}, "
              f"first safe30m {first_new}, first old {first_old}, offline mismatches {mism}, {info['outcome']}")
        assert mism <= 1                                   # float32 rounding at a border, at most
        assert first_new is not None
        if first_old is not None:
            assert first_new <= first_old
        total += sum(lab_new)
    assert total > 0


@pytest.fixture(scope="module")
def demo_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("g")
    src = root / "demos"
    os.makedirs(src)
    env = E.LandingEnv(renderer="tiny", seed=0)
    try:
        for e in (0, 1, 2, 3):
            rec, row = RD.fly_and_record(env, T.Teacher(), Troubles(), e)
            if row["kept"]:
                RD.save_chunk(dict(rec, ep=[e] * len(rec["t"])), str(src / f"demos_w0_0_{e:03d}.npz"))
    finally:
        env.close()
    D.build(str(src), str(root / "data"))
    return str(root / "data")


def test_relabel_demos(demo_data):
    rec = D.Demos(demo_data)
    same = D.Demos(demo_data, drop_gate="teacher")
    new = D.Demos(demo_data, drop_gate="safe30m")
    print(f"\n[gates] demos: recorded drop labels {int(rec.a['drop'].sum())}, teacher-gate relabel "
          f"{int(same.a['drop'].sum())}, safe30m relabel {int(new.a['drop'].sum())}")
    assert rec.a["drop"].sum() > 0
    assert np.array_equal(same.a["drop"], rec.a["drop"])
    assert new.a["drop"].sum() >= rec.a["drop"].sum() and new.drop_recorded == int(rec.a["drop"].sum())
    for k in ("move", "state", "priv", "flown"):
        assert np.array_equal(new.a[k], rec.a[k])


def test_train_bc_and_collect_with_gate(demo_data, tmp_path, monkeypatch):
    from v6_drop.scripts import train_bc as TB
    monkeypatch.setattr(TB, "RESULTS", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["train_bc", "--name", "g", "--data", demo_data, "--drop-gate", "safe30m",
                                      "--epochs", "1", "--batch", "2", "--max-clips", "2", "--device", "cpu"])
    TB.main()
    cfg = torch.load(tmp_path / "runs" / "g" / "last.pt", weights_only=False)["config"]
    assert cfg["drop_gate"] == "safe30m" and cfg["drop_labels"] >= cfg["drop_labels_recorded"] > 0
    torch.manual_seed(0)
    ck = tmp_path / "s.pt"
    torch.save(dict(model=Student().state_dict(), size=256, epoch=0), ck)
    monkeypatch.setattr(sys, "argv", ["dagger_collect", "--ckpt", str(ck), "--round", "4", "--flights", "1",
                                      "--workers", "1", "--renderer", "tiny", "--out", str(tmp_path / "r4"),
                                      "--drop-gate", "safe30m"])
    DC.main()
    assert "teacher drop gate 'safe30m'" in (tmp_path / "r4" / "summary.txt").read_text()
