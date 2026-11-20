"""v7: the FIXED V7 settings (decided by Yashwanth on 20 Nov 2026 from the S2 teacher study).

  setting                     value   from
  camera (gimbal) pitch       45 deg  S2 images check: 100/100 safe at 45 and 60; pad in view
                                      74.6 % vs 64.2 %; S1c: markers read better near the drop
  speed-change limit          0.3 m/s per decision = 3 m/s^2   (5 m/s^2: 2 to 2.5 x tip-overs)
  pad top speed               9 m/s   (teacher 98.85 % safe on 2000 fresh flights)
  flight-time limit           60 s    (600 decisions; longest safe teacher flight 54.8 s)
  drop shield, touch          0.8 m/s shield and the drone's own touch detector when the
                                      STUDENT flies (Stage 1 rules); demos are recorded with
                                      the simulator's touch and no shield, as in V6
Every V7 script after S2 builds its env with make_env(), so the settings live in one place.
"""
from .envs import world_v7 as W7

PITCH_DEG = 45.0
RATE_XY = 0.3
V_MAX = 9.0
HORIZON = W7.HORIZON            # 600 decisions = 60 s
DROP_SHIELD = 0.8
assert HORIZON == 600


def make_env(student_rules=False, **kw):
    """LandingEnvV7 with the fixed settings. student_rules=True adds the drop shield and the
    drone's own touch detector (how the student is tested). Other keywords pass through."""
    from .envs.landing_env_v7 import LandingEnvV7
    base = dict(pitch_deg=PITCH_DEG, rate_xy=RATE_XY, v_max=V_MAX)
    if student_rules:
        base.update(touch="sensor", drop_shield=DROP_SHIELD)
    base.update(kw)
    return LandingEnvV7(**base)
