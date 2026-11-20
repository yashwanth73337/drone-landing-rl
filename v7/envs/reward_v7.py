"""v7 S1d: the V7 reward (20 Nov 2026).

The SAME formulas and numbers as V6 (v6_drop/envs/reward.py). Only one value changes:
the standing height used for the feet height, h_feet = h - REST_HEIGHT, is the S500's
0.22 m (V6: 0.152 m, the old legs).

The reward itself is NOT final for V7. It is discussed with Yashwanth before any PPO run
(step R, 25 to 26 Nov): time cost over 60 s, blind penalty during a long chase, progress
while the pad pulls away, gamma 0.99 vs 0.995, slide after landing on a braking pad.
"""
from v6_drop.envs import reward as R6
from v6_drop.envs.reward import TIME_PENALTY, VISIBILITY_PENALTY, end_mark, geometry_from_body

from .s500 import REST_HEIGHT

__all__ = ["potential", "RewardV7", "geometry_from_body", "REST_HEIGHT"]


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
