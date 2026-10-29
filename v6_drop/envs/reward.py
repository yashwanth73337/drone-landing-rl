"""v6_drop Step 5: the reward (decided 29 Oct, R1-R5 "all OK").

R1  Step marks come from the CAMERA: ArUco + solvePnP give the pad centre in the body
    frame; the drone's own orientation (IMU) turns that into a horizontal distance d and
    a height h. End marks come from the simulator acting as a referee (C1 outcomes).
R2  "Fair progress" (potential-based shaping, Ng et al. 1999): the step mark is the
    change of a potential Phi, so going away and coming back earns exactly nothing.
        Phi = -( d + max(h_feet - SLOPE * d, 0) )
    d       horizontal distance from the drone centre to the pad centre (m)
    h_feet  height of the feet above the pad top (m) = h - REST_HEIGHT, >= 0
    Getting 1 m closer horizontally: +1. Getting 1 m lower: +1, but only while the drone
    is ABOVE a 45 deg glide slope (h_feet > d). Below the slope, going lower earns nothing.
    So height counts fully only when close, as Shin et al. intend, but in a form that
    cannot be gamed (see the Step 5 notes: Shin's divided form pays a climb-far /
    descend-near loop). The 45 deg slope also keeps the pad inside the 60 deg camera's
    view (the pad is visible from ~28 to ~92 deg below the horizon).
R3  If the camera has no CONFIDENT measurement (no marker >= MIN_SIDE_PX wide), progress
    pauses (the last good Phi is kept) and the step costs VISIBILITY_PENALTY.
R4  Every step costs TIME_PENALTY.
R5  End marks (added on the last step):
      success           +10 + up to +2 for landing near the centre + up to +2 for softness
      hard_landing      -2 * (impact - 1.0), floored at -10 (= 6 m/s): every bit slower helps
      slide_off, tip_over, not_still     -3
      crash_platform, crash_ground, flip, lost   -10
      timeout           0 (the time cost has already added up)
No progress mark is computed on the last step (after a touchdown or crash the camera
view says nothing useful); the end mark covers it.

source="true" computes d and h (the progress) from the simulator truth instead of the
camera, for the ablation "camera reward vs true-state reward" (not the main method).
The visibility mark still follows the camera, so only one thing changes.
"""
import numpy as np

from . import lmf2_params as P

SLOPE = 1.0                     # glide slope h_feet / d (45 deg)
VISIBILITY_PENALTY = 0.05
TIME_PENALTY = 0.01
SUCCESS = 10.0
CENTRE_BONUS = 2.0              # at the exact centre, 0 at CENTRE_RADIUS
CENTRE_RADIUS = 0.75            # m (the pad half-size)
SOFT_BONUS = 2.0                # at 0 m/s, 0 at V_IMPACT_MAX
V_IMPACT_MAX = 1.0
HARD_SLOPE = 2.0                # per m/s above V_IMPACT_MAX
HARD_FLOOR = -10.0
UNSAFE_ON_PAD = -3.0
CRASH = -10.0
END_MARKS = {"slide_off": UNSAFE_ON_PAD, "tip_over": UNSAFE_ON_PAD, "not_still": UNSAFE_ON_PAD,
             "crash_platform": CRASH, "crash_ground": CRASH, "flip": CRASH, "lost": CRASH,
             "timeout": 0.0}


def geometry_from_body(p_body, R):
    """Pad centre in the body frame + body->world rotation -> (d, h): horizontal distance
    and height of the drone CENTRE above the pad top. Uses only the camera result and the
    drone's own orientation."""
    p_lvl = R @ np.asarray(p_body, float)
    return float(np.hypot(p_lvl[0], p_lvl[1])), float(-p_lvl[2])


def potential(d, h):
    h_feet = max(h - P.REST_HEIGHT, 0.0)
    return -(d + max(h_feet - SLOPE * d, 0.0))


def end_mark(outcome, touchdown):
    if outcome == "success":
        r = float(np.linalg.norm(touchdown["land_xy"]))
        centre = CENTRE_BONUS * max(0.0, 1.0 - r / CENTRE_RADIUS)
        soft = SOFT_BONUS * max(0.0, 1.0 - touchdown["impact"] / V_IMPACT_MAX)
        return SUCCESS + centre + soft
    if outcome == "hard_landing":
        return max(HARD_FLOOR, -HARD_SLOPE * (touchdown["impact"] - V_IMPACT_MAX))
    return END_MARKS[outcome]


class Reward:
    """Stateful per episode: remembers the last good potential."""

    def __init__(self, source="camera"):
        assert source in ("camera", "true")
        self.source = source
        self.phi_last = None

    @staticmethod
    def seen(meas):
        return bool(meas is not None and meas.ok and meas.confident)

    def _geometry(self, meas, true_state):
        """(d, h) for the potential and whether it is available. The visibility mark
        always follows the CAMERA, also for source='true' (one variable per ablation)."""
        if self.source == "true":
            return geometry_from_body(true_state["rel_pos_body"], true_state["R"]), True
        if self.seen(meas):
            return geometry_from_body(meas.p_body, true_state["R"]), True
        return None, False

    def reset(self, meas, true_state):
        g, ok = self._geometry(meas, true_state)
        self.phi_last = potential(*g) if ok else None

    def step(self, meas, true_state, outcome=None, touchdown=None):
        """Returns (reward, terms dict)."""
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
            terms["end"] = end_mark(outcome, touchdown)
        total = terms["progress"] + terms["visibility"] + terms["time"] + terms["end"]
        return float(total), terms
