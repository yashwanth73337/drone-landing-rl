"""v6_drop Step 6d: the drop-gap diagnostic rebuilds the teacher's gate numbers correctly.

On a real teacher flight (TinyRenderer, headless), the numbers rebuilt OFFLINE from the saved
'priv' and 'state' (distance, feet height) equal the teacher's own, and the offline gate
gives exactly the teacher's drop labels.

Run (from the repo root):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest v6_drop/tests/test_s6d_dropgap.py -v -s
"""
import numpy as np

from v6_drop.envs import landing_env as E
from v6_drop.experts.teacher import Teacher
from v6_drop.scripts import drop_gap as G


def test_offline_gate_matches_teacher():
    env = E.LandingEnv(renderer="tiny", seed=0, headless=True)
    try:
        n_drop_flights = 0
        for seed in (100001, 100002, 100003):
            obs, _ = env.reset(seed=seed)
            t = Teacher()
            priv, state, d, hf, drop = [], [], [], [], []
            while True:
                act, ti = t.act(env)
                priv.append(obs["priv"]); state.append(obs["state"])
                d.append(ti["d"]); hf.append(ti["hf"]); drop.append(act["drop"])
                obs, _, term, trunc, info = env.step(act)
                if term or trunc:
                    break
            g = G.gate_numbers(np.stack(priv), np.stack(state))
            assert np.allclose(g["d"], d, atol=2e-4) and np.allclose(g["hf"], hf, atol=2e-4)
            _, dr = G.gate(g)
            assert np.array_equal(dr.astype(int), np.array(drop)), (seed, np.flatnonzero(dr), np.flatnonzero(drop))
            n_drop_flights += int(any(drop))
            print(f"\n[gap-test] seed {seed}: {len(d)} steps, outcome {info['outcome']}, drop labels {sum(drop)}")
        assert n_drop_flights >= 2
    finally:
        env.close()
