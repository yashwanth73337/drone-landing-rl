"""v7: the FIXED V7 settings (decided by Yashwanth on 20 Nov 2026 from the S2 teacher study).

  setting                     value   from
  camera (gimbal) pitch       60 deg  CHANGED 20 Nov 22:00 from 45 (his decision). S2: teacher
                                      100/100 with images at both; pad in view 64 % (60) vs 75 %
                                      (45). But the 45 deg student (BC + 3 DAgger rounds, blind
                                      mask) stayed at 0 to 1/400 on dev: near the pad a ~1 m slip
                                      puts the pad under/behind a 45 deg camera (its lower edge is
                                      13 deg short of straight down) and it never finds it again.
                                      60 deg sees straight down. The 45 deg runs stay in
                                      v7/results/{demos, bc_data, dagger*, runs/bc_v7, dagger_r*};
                                      60 deg runs go to v7/results/p60/ and runs/*_p60.
  speed-change limit          0.3 m/s per decision = 3 m/s^2   (5 m/s^2: 2 to 2.5 x tip-overs)
  pad top speed               9 m/s   (teacher 98.85 % safe on 2000 fresh flights)
  flight-time limit           60 s    (600 decisions; longest safe teacher flight 54.8 s)
  drop shield, touch          0.8 m/s shield and the drone's own touch detector when the
                                      STUDENT flies (Stage 1 rules); demos are recorded with
                                      the simulator's touch and no shield, as in V6
Every V7 script after S2 builds its env with make_env(), so the settings live in one place.
"""
from .envs import world_v7 as W7

PITCH_DEG = 60.0
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
