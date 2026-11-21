"""v7 S3v tests (21 Nov 2026): the --teacher option (view-keeping teacher for demos and DAgger).

  1. v7.config.make_teacher: "v7" = TeacherV7 (default), "view" = TeacherV7View for the
     fixed camera angle (60 deg: k = tan 32 deg); anything else is an error
  2. one teacher per data folder (teacher.txt): same kind again is fine; another kind is
     refused; a folder with old .npz data and no teacher.txt counts as "v7"
  3. one demo attempt with the view teacher: the labels are the VIEW teacher's (they differ
     from TeacherV7's on the same seed), and the same attempt twice gives identical data
     (after one warm-up flight: a fresh simulator's first 2 pictures can differ slightly)
  4. the demo recorder and the DAgger collector with --teacher view (1 worker, 2 flights):
     teacher.txt says "view", the summary names the teacher, mixing is refused
  5. the flight diagnostic runs Part B with the view teacher
One EGL env per test that flies (the scripts open their own in worker processes).
Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -s v7/tests/test_s3v_teacher_option.py
"""
import numpy as np
import pytest
import torch

from v6_drop.learn.student import Student
from v7 import config as C
from v7.experts.teacher_v7 import TeacherV7
from v7.experts.teacher_view_v7 import TeacherV7View
from v7.experts.troubles_v7 import TroublesV7
from v7.scripts import dagger_collect_v7 as D
from v7.scripts import record_demos_v7 as R


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    torch.manual_seed(0)
    p = tmp_path_factory.mktemp("ck") / "untrained.pt"
    torch.save(dict(model=Student(256).state_dict(), size=256), p)
    return str(p)


# ---------------------------------------------------------------- 1. make_teacher
def test_make_teacher():
    assert type(C.make_teacher()) is TeacherV7
    assert type(C.make_teacher("v7")) is TeacherV7
    t = C.make_teacher("view")
    assert type(t) is TeacherV7View and t.pitch_deg == C.PITCH_DEG == 60.0
    assert t.k_view == pytest.approx(np.tan(np.radians(32.0)), abs=1e-3)
    with pytest.raises(ValueError):
        C.make_teacher("pid")


# ---------------------------------------------------------------- 2. one teacher per folder
def test_teacher_file(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    C.check_teacher_file(str(a), "view")
    assert (a / "teacher.txt").read_text().strip() == "view"
    C.check_teacher_file(str(a), "view")                     # same kind: fine
    with pytest.raises(SystemExit):
        C.check_teacher_file(str(a), "v7")
    old = tmp_path / "old"                                    # old data, no teacher.txt
    old.mkdir()
    (old / "demos_w0_0_000.npz").write_bytes(b"x")
    with pytest.raises(SystemExit):
        C.check_teacher_file(str(old), "view")
    C.check_teacher_file(str(old), "v7")
    assert (old / "teacher.txt").read_text().strip() == "v7"


# ---------------------------------------------------------------- 3. one demo attempt
def test_view_attempt_labels():
    env = C.make_env(renderer="egl", seed=0)
    try:
        # warm-up: in a FRESH simulator the first 2 pictures of the first flight can differ
        # slightly from a repeat (seen 21 Nov in the sandbox, both teachers; renderer start-up)
        R.fly_and_record(env, C.make_teacher("v7"), TroublesV7(), 1)
        e = 3
        r_view, row_v = R.fly_and_record(env, C.make_teacher("view"), TroublesV7(), e)
        r_view2, _ = R.fly_and_record(env, C.make_teacher("view"), TroublesV7(), e)
        r_v7, _ = R.fly_and_record(env, C.make_teacher("v7"), TroublesV7(), e)
    finally:
        env.close()
    mv, mv2 = np.array(r_view["move"]), np.array(r_view2["move"])
    np.testing.assert_array_equal(mv, mv2)                    # reproducible
    np.testing.assert_array_equal(np.array(r_view["image"]), np.array(r_view2["image"]))
    m7 = np.array(r_v7["move"])
    k = min(len(mv), len(m7))
    assert not np.array_equal(mv[:k], m7[:k])                 # the view teacher's labels
    assert all(0 <= ph < len(R.PHASES) for ph in r_view["phase"])
    print(f"\nattempt {e}: view teacher {row_v['outcome']} in {row_v['flight_s']} s")


# ---------------------------------------------------------------- 4. the scripts
def test_record_demos_script(tmp_path):
    out = tmp_path / "demos"
    R.main(["--teacher", "view", "--attempts", "2", "--workers", "1", "--out", str(out)])
    assert (out / "teacher.txt").read_text().strip() == "view"
    assert "(teacher: view)" in (out / "summary.txt").read_text()
    with pytest.raises(SystemExit):                           # no v7 top-up into this folder
        R.main(["--teacher", "v7", "--attempts", "1", "--start", "2", "--workers", "1",
                "--out", str(out)])


def test_dagger_script(ckpt, tmp_path):
    out = tmp_path / "r1"
    D.main(["--teacher", "view", "--ckpt", ckpt, "--round", "1", "--flights", "2",
            "--workers", "1", "--beta", "0.5", "--out", str(out)])
    assert (out / "teacher.txt").read_text().strip() == "view"
    assert "teacher: view" in (out / "summary.txt").read_text()
    assert len(list(out.glob("dagger_*.npz"))) == 1


# ---------------------------------------------------------------- 5. diagnostic
def test_diag_part_b_view(ckpt):
    from v6_drop.scripts.eval_student import load
    from v7.scripts.diag_student_v7 import part_b
    model, _ = load(ckpt, "cpu")
    lines = part_b(model, "cpu", [30000], every=50, teacher_kind="view")
    assert any("seed 30000" in x for x in lines)
