import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from envs.LandingAviary import LandingAviary

env = LandingAviary(gui=True)
obs, info = env.reset(seed=0)

# Normalised action in [-1, 1]. 0 = hover, negative = descend.
action = np.array([[-0.05, -0.05, -0.05, -0.05]])

for i in range(300):
    obs, r, term, trunc, info = env.step(action)
    if i % 10 == 0:
        print(f"step {i:3d}  above_pad {info['height_above_pad']:+.3f}  "
              f"horiz {info['horiz_dist']:.3f}  reward {r:+.2f}")
    time.sleep(1/30)
    if term or trunc:
        print(f"\nENDED step {i}: terminated={term} truncated={trunc}")
        print(info)
        print()
        print("  approach controlled? :", env._approachWasControlled())
        print("  settle ok?           :", env.settle_ok)
        print("  offset < pad size?   :", env.touchdown_offset < env.PLAT_SIZE,
              f"({env.touchdown_offset:.3f} vs {env.PLAT_SIZE})")
        break
env.close()