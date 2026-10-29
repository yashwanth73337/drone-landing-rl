"""v6_drop Step 6d: the oracle-drop diagnostic (scripts/oracle_drop.py).

Checks:
  1. predicted impact = sqrt(sink^2 + 2 g h)
  2. the 'teacher' oracle gate, fed the exam's per-step record, cuts at exactly the step
     where the teacher itself labels the drop (same gate, same numbers)
  3. a pilot that flies like the teacher but NEVER drops: the 'safe20' oracle forces one
     drop, the landing is safe and ends by 'drop', and the predicted impact matches the
     simulator's exact free-fall impact
  4. the exam hook leaves a normal exam unchanged (drop_rule=None)

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s6d_oracle.py -v -s
"""
import numpy as np
import pytest

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.experts.teacher import Teacher
from v6_drop.scripts import eval_student as ES
from v6_drop.scripts import oracle_drop as OD


class TeacherNoDrop:
    """Exam 'model' that flies the teacher's move and never drops on its own."""

    def __init__(self, env, drop=False):
        self.env, self.t, self.drop, self.labels = env, Teacher(), drop, []

    def act(self, obs, hc, device):
        if hc is None:
            self.t.reset()
            self.labels = []
        a, _ = self.t.act(self.env)
        self.labels.append(a["drop"])
        return {"move": a["move"], "drop": a["drop"] if self.drop else 0}, 1, 0.0


@pytest.fixture(scope="module")
def env():
    e = E.LandingEnv(renderer="tiny", seed=0)
    yield e
    e.close()


def test_predicted_impact():
    assert OD.predicted_impact(dict(vz_rel=-0.3, hf=0.03)) == pytest.approx(np.sqrt(0.09 + 2 * P.G * 0.03))
    assert OD.predicted_impact(dict(vz_rel=0.2, hf=0.0)) == 0.0


def test_teacher_gate_matches_teacher(env):
    m = TeacherNoDrop(env, drop=True)
    for seed in (100001, 100002):
        rule = OD.OracleGate("teacher")
        log, info, _, _ = ES.fly(env, m, seed, "cpu", drop_rule=rule)
        rule.reset()
        fired = [rule(r) for r in log]
        print(f"\n[oracle-test] seed {seed}: teacher drop at {m.labels.index(1)}, oracle at "
              f"{fired.index(True) if any(fired) else None}, outcome {info['outcome']}")
        assert fired == [bool(x) for x in m.labels]


def test_safe20_forces_safe_drop(env):
    m = TeacherNoDrop(env, drop=False)
    rule = OD.OracleGate("safe20")
    log, info, _, _ = ES.fly(env, m, 100003, "cpu", drop_rule=rule)
    td = info["touchdown"]
    print(f"\n[oracle-test] forced at feet {100 * log[-1]['hf']:.1f} cm, predicted "
          f"{OD.predicted_impact(log[-1]):.3f}, sim {td['impact']:.3f} m/s, outcome {info['outcome']}")
    assert log[-1].get("forced_drop") and td["reason"] == "drop"
    assert info["outcome"] == "success"
    assert td["impact"] == pytest.approx(OD.predicted_impact(log[-1]), abs=0.08)
    rows = [dict(outcome=info["outcome"], forced_drop=1, impact=td["impact"], cut_d_cm=1.0, cut_hf_cm=3.0,
                 cut_vxy=0.05, cut_sink=0.1)]
    assert "forced drops: 1/1" in OD.summary(rows, "x.pt", "safe20", 1, 9000)


def test_hook_off_is_plain_exam(env):
    m = TeacherNoDrop(env, drop=True)
    log1, info1, _, _ = ES.fly(env, m, 100004, "cpu")
    log2, info2, _, _ = ES.fly(env, m, 100004, "cpu", drop_rule=None)
    assert info1["outcome"] == info2["outcome"] and len(log1) == len(log2)
    assert not any(r.get("forced_drop") for r in log1)
