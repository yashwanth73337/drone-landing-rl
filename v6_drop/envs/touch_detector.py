"""v6_drop T1 (7 Nov): the drone's own TOUCH DETECTOR (replaces the perfect auto-cut).

Why: until T1, World.step cut the motors in the same 0.01 s physics step in which PyBullet's
contact list said "a foot touches the pad top". A real drone has no contact list. It must
work out "I have landed" from its own sensors, and that takes some time.

The rule (decided 7 Nov, from a 60-touch sandbox study with the teacher, sinking 0.1-0.3 m/s):
  SEE   the estimated feet height above the pad top is lower than FEET_MAX, and
  FEEL  the drone's own vertical speed (IMU/EKF; no noise in Stage 1) is slower than VZ_MAX,
  both true for HOLD_STEPS physics steps in a row (0.01 s each)  ->  motors off.
Study numbers: on the pad the camera read the feet at -0.4..+0.1 cm (60/60); sinking stopped
within ~0.03 s of the touch; in flight the rule never held.
Not used: thrust (our Lee controller has no integral part, so on the pad thrust drops only
2-8 %, inside its normal flight range) and the accelerometer bump (clear in the simulator,
but a real IMU shakes with the propellers and our simulator has no vibration).

T2 (7 Nov, decided by Yashwanth after the student videos 9016 / 9094 / 9039): SHORT-GAP
PREDICTION. The student often lands 0.15-0.35 m BEHIND the pad centre; from there, low and
nose-up, the 60 deg camera sees no full marker (sandbox: level drone, feet 0.5 cm, blind at
>= 0.4 m behind the centre). So the feet height is ESTIMATED like a real state estimator does:
  feet_est = last confident camera value + integral of the own vertical speed since that frame
(the pad does not move up or down in Stage 1). The estimate is used for at most MAX_GAP_STEPS
physics steps after the last confident frame (0.5 s); after that it is unknown (SEE false).
With a fresh frame every 0.1 s this also removes the frame's age inside a decision.

Only ONBOARD values come in. The simulator's contact list never does.
"""

from . import lmf2_params as P

FEET_MAX = 0.01        # m, estimated feet height above the pad top
VZ_MAX = 0.05          # m/s, |own vertical speed|
HOLD_STEPS = 10        # physics steps (0.01 s each) = 0.1 s
MAX_GAP_STEPS = 50     # T2: physics steps a camera value may be carried forward = 0.5 s
DT = P.PHYSICS_DT      # s, one physics step (0.01)


class TouchDetector:
    def __init__(self, feet_max=FEET_MAX, vz_max=VZ_MAX, hold_steps=HOLD_STEPS,
                 max_gap_steps=MAX_GAP_STEPS):
        self.feet_max = float(feet_max)
        self.vz_max = float(vz_max)
        self.hold_steps = int(hold_steps)
        self.max_gap_steps = int(max_gap_steps)
        assert self.hold_steps >= 1 and self.max_gap_steps >= 0
        self.reset()

    def reset(self):
        self.cam_feet = None           # last confident camera feet height (m), or None
        self.age = 0                   # physics steps since that frame
        self.sink = 0.0                # m, integral of own vz since that frame
        self.count = 0                 # physics steps in a row with SEE and FEEL both true

    def set_camera(self, feet):
        """New camera frame: feet height in m, or None (no confident measurement in this
        frame: keep predicting from the last confident one)."""
        if feet is not None:
            self.cam_feet = float(feet)
            self.age = 0
            self.sink = 0.0

    def feet_est(self):
        """Estimated feet height (m) now, or None (never seen, or the gap is too long)."""
        if self.cam_feet is None or self.age > self.max_gap_steps:
            return None
        return self.cam_feet + self.sink

    def update(self, vz):
        """One physics step. vz = own vertical speed (m/s) after the step.
        Returns True = cut the motors."""
        vz = float(vz)
        if self.cam_feet is not None:
            self.age += 1
            self.sink += vz * DT
        h = self.feet_est()
        see = h is not None and h < self.feet_max
        feel = abs(vz) < self.vz_max
        self.count = self.count + 1 if (see and feel) else 0
        return self.count >= self.hold_steps

    def config(self):
        return dict(feet_max=self.feet_max, vz_max=self.vz_max, hold_steps=self.hold_steps,
                    max_gap_steps=self.max_gap_steps)
