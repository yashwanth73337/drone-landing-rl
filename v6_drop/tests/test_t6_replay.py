"""v6_drop T6 (10 Nov): exact replay + film of final_eval flights (scripts/replay_flight.py,
dice_exam.fly_dice before/after hooks).

Uses untrained networks and TinyRenderer, so it needs no results/.

Checks:
  1. the hooks only look: fly_dice with the replay Recorder gives the SAME row as without
  2. compare(): equal rows -> {}; a changed field is reported with csv and replay values
  3. end to end: final_eval (1 worker, 2 seeds flown one after the other in ONE env,
     touch sensor, shield 0.8) -> replay_flight of both seeds MATCHES every csv column,
     and writes replay_<student>_<seed>.mp4/.png
  4. a wrong --touch for the csv is refused

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_t6_replay.py -v -s
"""
import csv

import pytest
import torch

from v6_drop.envs import landing_env as E
from v6_drop.learn.student import Student
from v6_drop.scripts import dice_exam as DX
from v6_drop.scripts import final_eval as FE
from v6_drop.scripts import replay_flight as RF
from v6_drop.scripts.train_ppo import touch_kw


@pytest.fixture(scope="module")
def ck(tmp_path_factory):
    torch.manual_seed(1)
    p = tmp_path_factory.mktemp("ck") / "s.pt"
    torch.save(dict(model=Student().state_dict(), size=256, epoch=0), p)
    return p


def test_hooks_only_look(ck):
    torch.set_num_threads(1)
    m, _ = RF.ES.load(str(ck), "cpu")
    env = E.LandingEnv(renderer="tiny", seed=0, **touch_kw("sensor", 10))
    try:
        env.drop_shield = 0.8
        plain = DX.fly_dice(env, m, 9500, "clean", DX.sigma_norm())
        ref = {k: RF.as_csv(v) for k, v in dict(plain, shield=0.8).items()}
        row, diff, rec, info = RF.replay(env, m, 9500, ref, film=True)
    finally:
        env.close()
    print(f"\n[t6] plain {plain['outcome']} t {plain['time_s']} | with hooks {row['outcome']} t {row['time_s']}, "
          f"{len(rec.log)} records, {len(rec.pics)} frames")
    assert row == plain and diff == {}
    assert len(rec.log) == plain["steps"] and len(rec.pics) >= plain["steps"]


def test_compare():
    ref = dict(seed="1", outcome="success", impact="0.5", touches="", student="a", shield="0.8")
    assert RF.compare(dict(seed=1, outcome="success", impact=0.5, touches=None), ref) == {}
    d = RF.compare(dict(seed=1, outcome="timeout", impact=None, touches=None), ref)
    assert d == {"outcome": ("success", "timeout"), "impact": ("0.5", "")}


def test_end_to_end_match(tmp_path, ck):
    FE.main(["--tag", "t", "--n", "2", "--seed0", "9500", "--workers", "1", "--renderer", "tiny",
             "--out-root", str(tmp_path / "out"), "--touch", "sensor", "--students", f"a={ck}:0.8"])
    out = tmp_path / "out" / "t"
    rows = list(csv.DictReader(open(out / "flights_a.csv")))
    n = RF.main(["--csv", str(out / "flights_a.csv"), "--ckpt", str(ck), "--seeds", "9500", "9501",
                 "--touch", "sensor", "--renderer", "tiny"])
    print(f"\n[t6] csv outcomes {[r['outcome'] for r in rows]}; replay matches {n}/2")
    assert n == 2
    for s in (9500, 9501):
        assert (out / f"replay_a_{s}.mp4").exists() and (out / f"replay_a_{s}.png").exists()
    with pytest.raises(SystemExit):
        RF.main(["--csv", str(out / "flights_a.csv"), "--ckpt", str(ck), "--seeds", "9500",
                 "--renderer", "tiny", "--no-video"])                  # wrong touch mode
