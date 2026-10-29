"""v6_drop Step 6b: TROUBLES for robust demonstrations (DART-style, Laskey et al. 2017).

While recording, the action that is FLOWN is disturbed on purpose, but the LABEL saved is
always the teacher's own (correct) action. The data then holds many "slightly wrong
situation -> right correction" pairs, which a learner of perfect paths would never see.

Kinds and mix (decided 29 Oct, R1/R2):
  clean          25%  no trouble
  wobble         25%  smooth random errors on every flown velocity command:
                      about +-0.4 m/s horizontal, +-0.15 m/s vertical, correlated over ~0.5 s,
                      shrinking near the pad (x 0.2 at the pad) so a drop stays possible
  gust           20%  1-3 sideways pushes of 2-4 N (1.24 kg drone: ~1.6-3.2 m/s^2),
                      0.2-0.5 s each, at random times between 1 s and 12 s
  pad_lost       15%  once, after a random 1-6 s, when the feet are > 0.8 m up and a
                      marker is in view: fly 2-3 m/s away from the pad (back, left or
                      right) for 0.6-1.0 s
  bad_approach   15%  once, when the feet first go below 0.4 m: a 1.5-2 m/s push
                      (forward, back, left or right) for 0.3-0.5 s -> usually a go-around
Each kind draws its random numbers from the episode seed, so an episode is reproducible.
The drop decision is never disturbed (the teacher decides it).
"""
import numpy as np

from ..envs import lmf2_params as P
from ..envs.landing_env import ACT_SCALE
from ..envs.platform_s1 import PAD_TOP_Z

KINDS = ("clean", "wobble", "gust", "pad_lost", "bad_approach")
MIX = (0.25, 0.25, 0.20, 0.15, 0.15)
CODE = {"": 0, "wobble": 1, "GUST": 2, "PUSH-AWAY": 3, "PUSH-LOW": 4}   # per-step flag codes

WOBBLE_SIGMA = np.array([0.4, 0.4, 0.15, 0.0])     # m/s, m/s, m/s, rad/s
WOBBLE_TAU = 0.5                                  # s
WOBBLE_NEAR = 0.5                                 # m: below this feet height the noise shrinks
WOBBLE_MIN = 0.2                                  # ... down to this fraction
GUST_N = (1, 3)
GUST_FORCE = (2.0, 4.0)                           # N
GUST_DUR = (0.2, 0.5)                             # s
GUST_WINDOW = (1.0, 12.0)                         # s
AWAY_SPEED = (2.0, 3.0)                           # m/s
AWAY_DUR = (0.6, 1.0)                             # s
AWAY_MIN_H = 0.8                                  # m, feet height
AWAY_AFTER = (1.0, 6.0)                           # s, earliest trigger time (else it fired at t=0)
LOW_H = 0.4                                       # m, feet height that triggers bad_approach
LOW_SPEED = (1.5, 2.0)                            # m/s
LOW_DUR = (0.3, 0.5)                              # s


def pick_kind(rng):
    return KINDS[int(rng.choice(len(KINDS), p=MIX))]


def _steps(sec):
    return max(1, int(round(sec / P.POLICY_DT)))


class Troubles:
    """Call reset(kind, rng) at the episode start, then before every env.step:
        flown_action, flag = troubles.apply(env, teacher_action, seen)
    'seen' = was a marker seen in the last camera frame (pad_lost waits for it)."""

    def reset(self, kind, rng):
        assert kind in KINDS, kind
        self.kind, self.rng = kind, rng
        self.k = 0
        self.noise = np.zeros(4)
        self.events = []                          # (start step, n steps, payload)
        self.done_once = False
        self.active = None                        # (steps left, payload)
        if kind == "gust":
            n = int(rng.integers(GUST_N[0], GUST_N[1] + 1))
            for _ in range(n):
                start = _steps(rng.uniform(*GUST_WINDOW))
                ang = rng.uniform(-np.pi, np.pi)
                f = rng.uniform(*GUST_FORCE)
                self.events.append((start, _steps(rng.uniform(*GUST_DUR)),
                                    np.array([f * np.cos(ang), f * np.sin(ang), 0.0])))
        elif kind == "pad_lost":
            self.payload = (rng.choice(["back", "left", "right"]), rng.uniform(*AWAY_SPEED),
                            _steps(rng.uniform(*AWAY_DUR)))
            self.after = _steps(rng.uniform(*AWAY_AFTER))
        elif kind == "bad_approach":
            self.payload = (rng.choice(["forward", "back", "left", "right"]),
                            rng.uniform(*LOW_SPEED), _steps(rng.uniform(*LOW_DUR)))

    @staticmethod
    def _dir(name):
        return {"forward": np.array([1.0, 0.0]), "back": np.array([-1.0, 0.0]),
                "left": np.array([0.0, 1.0]), "right": np.array([0.0, -1.0])}[name]

    def apply(self, env, teacher_action, seen=True):
        """Returns (flown action dict, flag string). Sets env.world.quad.ext_force (gusts)."""
        move = np.asarray(teacher_action["move"], np.float64) * ACT_SCALE    # m/s, rad/s
        flag = ""
        quad = env.world.quad
        quad.ext_force[:] = 0.0
        s = env.world.true_state()
        hf = float(s["pos"][2] - PAD_TOP_Z - P.REST_HEIGHT)
        if self.kind == "wobble":
            rho = np.exp(-P.POLICY_DT / WOBBLE_TAU)
            self.noise = rho * self.noise + np.sqrt(1 - rho * rho) * WOBBLE_SIGMA * self.rng.normal(size=4)
            scale = np.clip(hf / WOBBLE_NEAR, WOBBLE_MIN, 1.0)
            move = move + scale * self.noise
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
        flown = {"move": np.clip(move / ACT_SCALE, -1.0, 1.0).astype(np.float32),
                 "drop": int(teacher_action["drop"])}
        return flown, flag
