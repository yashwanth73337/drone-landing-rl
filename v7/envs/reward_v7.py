"""v7 S1d: the V7 reward (20 Nov 2026).

The SAME formulas and numbers as V6 (v6_drop/envs/reward.py). Only one value changes:
the standing height used for the feet height, h_feet = h - REST_HEIGHT, is the S500's
0.22 m (V6: 0.152 m, the old legs).

STEP R (24 Nov 2026, decided by Yashwanth, "yes R"): two END MARKS change, all else as V6.
  unsafe landing (slide_off, tip_over, not_still)   -3  ->  -8
  hard landing (impact > 1.0 m/s)   -2 (impact - 1), floor -10  ->  -8 - 2 (impact - 1), floor -10
Why: over 60 s flights the time cost alone adds up to -6, so with the V6 values a tip-over
at 20 s (-3 - 2 = -5) or a hard landing at 1.2 m/s (-0.4 - 2) scored BETTER than a safe hover
until the timeout (about -6): PPO would learn that a risky or slammed landing is worth it.
New order: success (+10 to +16) > timeout (0, with the time cost ~ -6) > unsafe landing (-8)
> hard landing (-8 to -10, slower always better) > crash / lost (-10). PPO still tries to
land when its chance of a safe landing is above ~10 %.
Kept (his decision): success marks and bonuses (air-drop-only option), crash/lost -10,
timeout 0, time -0.01 per step, blind -0.05 per step, the V6 progress potential, gamma 0.99
(in the PPO script). Slide after landing on a braking pad: the referee's 2 s check already
gives slide_off (now -8).
"""
import numpy as np

from v6_drop.envs import reward as R6
from v6_drop.envs.reward import TIME_PENALTY, VISIBILITY_PENALTY, geometry_from_body

from .s500 import REST_HEIGHT

__all__ = ["potential", "end_mark", "RewardV7", "geometry_from_body", "REST_HEIGHT"]

UNSAFE_ON_PAD = -8.0          # step R (V6: -3)
HARD_BASE = -8.0              # step R: hard landing starts at the unsafe value ...
HARD_SLOPE = R6.HARD_SLOPE    # ... and loses 2 per m/s above 1.0 m/s (as V6)
HARD_FLOOR = R6.HARD_FLOOR    # -10 (= a crash) from 2.0 m/s
END_MARKS = dict(R6.END_MARKS, slide_off=UNSAFE_ON_PAD, tip_over=UNSAFE_ON_PAD,
                 not_still=UNSAFE_ON_PAD)


def end_mark(outcome, touchdown, air_drop_only=False):
    """V7 end mark (step R). success: exactly V6 (10 + centre + soft + drop bonus)."""
    if outcome == "success":
        return R6.end_mark(outcome, touchdown, air_drop_only)
    if outcome == "hard_landing":
        excess = float(touchdown["impact"]) - R6.V_IMPACT_MAX
        return max(HARD_FLOOR, HARD_BASE - HARD_SLOPE * excess)
    return END_MARKS[outcome]


def potential(d, h):
    """V6 potential with the S500 standing height: -(d + max(h_feet - SLOPE * d, 0))."""
    h_feet = max(h - REST_HEIGHT, 0.0)
    return -(d + max(h_feet - R6.SLOPE * d, 0.0))


class RewardV7(R6.Reward):
    """V6 Reward; reset() and step() are the V6 code with the V7 potential."""

    def reset(self, meas, true_state):
        g, ok = self._geometry(meas, true_state)
        self.phi_last = potential(*g) if ok else None

    def step(self, meas, true_state, outcome=None, touchdown=None):
        terms = dict(progress=0.0, visibility=0.0, time=-TIME_PENALTY, end=0.0,
                     d=None, h=None, seen=None)
        if outcome is None:
            g, ok = self._geometry(meas, true_state)
            terms["seen"] = self.seen(meas)
            if ok:
                terms["d"], terms["h"] = g
                phi = potential(*g)
                if self.phi_last is not None:
                    terms["progress"] = phi - self.phi_last
                self.phi_last = phi
            if not terms["seen"]:
                terms["visibility"] = -VISIBILITY_PENALTY
        else:
            terms["end"] = end_mark(outcome, touchdown, self.air_drop_only)
        total = terms["progress"] + terms["visibility"] + terms["time"] + terms["end"]
        return float(total), terms
