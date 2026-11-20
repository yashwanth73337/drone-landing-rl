"""v7 S3a: TROUBLES for robust demonstrations (DART), V7 version (20 Nov 2026).

The V6 troubles (v6_drop/experts/troubles.py) with the SAME kinds, mix and numbers
(decided 29 Oct; kept for V7 on 20 Nov, D3 "V6 settings"):
  clean 25 %, wobble 25 %, gust 20 %, pad_lost 15 %, bad_approach 15 %.
The flown action is disturbed; the saved LABEL is always the teacher's own action; the drop
decision is never disturbed.

Only two things change, both in apply():
  1. speeds: m/s <-> [-1, 1] with the V7 env's scale (env.act_scale, +-10 m/s), not V6's +-6.
     So a 2 to 3 m/s push stays a 2 to 3 m/s push.
  2. height: the S500's TRUE lowest point above the pad top (as the V7 teacher), not the V6
     legs formula (0.152 m). It decides when wobble shrinks (below 0.5 m), when pad_lost
     may fire (above 0.8 m) and when bad_approach fires (first time below 0.4 m).
"""
import numpy as np

from v6_drop.envs import lmf2_params as P
from v6_drop.experts import troubles as T6
from v6_drop.experts.troubles import (AWAY_MIN_H, CODE, KINDS, LOW_H, MIX, WOBBLE_MIN,  # noqa: F401
                                      WOBBLE_NEAR, WOBBLE_SIGMA, WOBBLE_TAU, pick_kind)

from ..envs.platform_v7 import PAD_TOP_Z
from ..envs.s500 import lowest_z


class TroublesV7(T6.Troubles):
    """Same use as V6: reset(kind, rng), then before every env.step:
        flown_action, flag = troubles.apply(env, teacher_action, seen)"""

    def apply(self, env, teacher_action, seen=True):
        scale = env.act_scale
        move = np.asarray(teacher_action["move"], np.float64) * scale     # m/s, rad/s
        flag = ""
        quad = env.world.quad
        quad.ext_force[:] = 0.0
        s = env.world.true_state()
        hf = float(lowest_z(s["R"], s["pos"]) - PAD_TOP_Z)
        if self.kind == "wobble":
            rho = np.exp(-P.POLICY_DT / WOBBLE_TAU)
            self.noise = rho * self.noise + np.sqrt(1 - rho * rho) * WOBBLE_SIGMA * self.rng.normal(size=4)
            move = move + np.clip(hf / WOBBLE_NEAR, WOBBLE_MIN, 1.0) * self.noise
            flag = "wobble"
        elif self.kind == "gust":
            for start, n, force in self.events:
                if start <= self.k < start + n:
                    quad.ext_force[:] = force
                    flag = "GUST"
        elif self.kind in ("pad_lost", "bad_approach"):
            if self.active is None and not self.done_once:
                trigger = ((hf > AWAY_MIN_H and seen and self.k >= self.after)
                           if self.kind == "pad_lost" else (hf < LOW_H))
                if trigger:
                    self.active = [self.payload[2], self.payload]
                    self.done_once = True
            if self.active is not None:
                name, speed, _ = self.active[1]
                move[:2] = move[:2] + speed * self._dir(name)       # heading frame
                self.active[0] -= 1
                if self.active[0] <= 0:
                    self.active = None
                flag = "PUSH-AWAY" if self.kind == "pad_lost" else "PUSH-LOW"
        self.k += 1
        flown = {"move": np.clip(move / scale, -1.0, 1.0).astype(np.float32),
                 "drop": int(teacher_action["drop"])}
        return flown, flag
