"""v7 S1d: speed of the V7 environment on this computer (decisions per second).

Runs the true-state test pilot (v7/tests/test_s1d_env.py) for whole flights, first WITH
camera images (as in training), then headless (no images, as in fast teacher tests).
With images: as the Stage 1 final runs (drop shield 0.8 m/s, the drone's own touch
detector). Headless: the simulator's touch (no frames, so no detector).

Laptop: plug in the charger and run `powerprofilesctl set performance` first.
Run:    python -m v7.scripts.time_env_v7            (about 2 to 4 minutes)
        python -m v7.scripts.time_env_v7 --flights 3
"""
import argparse
import collections
import time

from v7.envs.landing_env_v7 import LandingEnvV7
from v7.tests.test_s1d_env import fly, pilot_move

MODES = ("straight", "wave", "corners", "random")


def run(headless, flights, seed0=30200):
    # headless has no frames, so the touch detector cannot fire: use the simulator's touch
    kw = dict(touch="perfect") if headless else dict(touch="sensor", drop_shield=0.8)
    env = LandingEnvV7(renderer="egl", seed=7, headless=headless, **kw)
    out = collections.Counter()
    n_all, t_all = 0, 0.0
    try:
        for mode in MODES:
            for i in range(flights):
                env.reset(seed=seed0 + i, options=dict(mode=mode))
                t0 = time.time()
                info, n, _ = fly(env, pilot_move)
                t_all += time.time() - t0
                n_all += n
                out[info["outcome"]] += 1
    finally:
        env.close()
    return n_all, t_all, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=5, help="flights per path")
    a = ap.parse_args()
    for headless in (False, True):
        n, t, out = run(headless, a.flights)
        name = "headless   " if headless else "with images"
        print(f"{name}: {n} decisions in {t:.1f} s = {n / t:.0f} decisions/s "
              f"({n / t / 10:.1f} x real time); outcomes {dict(out)}", flush=True)


if __name__ == "__main__":
    main()
