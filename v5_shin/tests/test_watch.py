"""Viewer script (scripts/watch.py): headless video mode for a P run, a vision run and the oracle.

Run:  python -m pytest v5_shin/tests/test_watch.py -v
"""
import os
import shutil
import subprocess
import sys

import cv2
import pytest

from v5_shin.tests.test_block14_evaluate import make_run

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _frames(path):
    cap = cv2.VideoCapture(path)
    n, shape = 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n, shape = n + 1, f.shape
    cap.release()
    return n, shape


@pytest.mark.parametrize("who", ["privileged", "vision", "oracle"])
def test_video_mode(who, tmp_path):
    name = f"_pytest_watch_{who}"
    run = None if who == "oracle" else make_run(name, who)
    try:
        out = str(tmp_path / "w.mp4")
        args = ["--oracle"] if who == "oracle" else ["--run", name, "--ckpt", "c.pt"]
        cmd = [sys.executable, "-m", "v5_shin.scripts.watch", *args, "--c", "0.5", "--episodes", "2",
               "--max-steps", "4", "--video", out]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=ROOT)
        assert res.returncode == 0, res.stderr[-3000:]
        assert res.stdout.count("episode seed 900") == 2
        path = out if os.path.exists(out) else out[:-4] + ".avi"
        n, shape = _frames(path)
        assert n >= 2 * 4 and shape == (400, 1280, 3)
    finally:
        if run:
            shutil.rmtree(run, ignore_errors=True)
