"""v6_drop Step 6a: the privileged TEACHER (expert pilot).

Decided 29 Oct (E1, E2): the teacher may read the simulator's TRUE state, because only
the student (camera + own IMU) is ever deployed. It lands the supervisor's way:
  follow     match the pad's speed and close the gap
  glide      come down along an approach line BEHIND the pad
  hover      feet H_DROP (3 cm) above the pad centre, at the pad's speed
  drop       cut the motors (3 cm -> ~0.77 m/s impact)
  go-around  low but off the line -> climb to 0.3 m and line up again
It outputs the same action as the policy: move in [-1, 1]^4 (scaled by ACT_SCALE, then
rate-limited by the env) and drop in {0, 1}. Call act(env) with a LandingEnv.

Keeping the student able to follow (the INFORMATION GAP, Step 6 notes):
  * pad velocity: the teacher uses a LOW-PASS FILTERED pad velocity that starts from zero
    each episode (PAD_V_TAU), i.e. it "learns" the pad's speed about as fast as a camera
    pilot could, instead of acting on knowledge the student cannot have yet;
  * pad in view: it approaches along a line K_BACK x height behind the pad, which holds the
    pad ~55 deg below the horizon, the middle of the 60 deg camera's view;
  * acceleration: only the env's 3 m/s^2 rate limit. (Tried gentler, 1.5 m/s^2, and a
    "keep the pad in the picture" acceleration limit with climbing: both were WORSE,
    because the gap to a fast pad grows and the pad drops towards the image top edge.)
The reward is a potential, so the path does not change the teacher's total progress mark.

Tuning history (29 Oct, all in Claude's sandbox, headless):
  v1 straight at the pad: 60/60 safe but all by AUTO-CUT (the ~0.7 s vertical lag made it
     touch before it could stop at 3 cm)  -> descent limited to (height - 3 cm) / T_BRAKE
  v2 steady 8.8 cm lag behind the centre at 3.9 m/s (Lee loop has no integral, drag) ->
     drop gate never met -> integral term near the pad
  v3 500/500 safe, all by drop, but a marker was VISIBLE only 83% of steps (lost while
     chasing, and when passing over the pad) -> approach line K_BACK x height behind the
     pad: glide and hover then 100% visible; overall ~94% (60 episodes). The rest is
     mostly the first second of a chase (nose-down while speeding up) and far follows:
     inherent to a 60 deg forward camera; the student's LSTM must bridge it.
"""
import numpy as np

from ..envs import lmf2_params as P
from ..envs.landing_env import ACT_SCALE
from ..envs.lee_controller import yaw_rotation
from ..envs.platform_s1 import PAD_TOP_Z

H_DROP = 0.03           # m, feet above the pad top when dropping
K_BACK = 0.5            # approach line: stay K_BACK x feet height behind the pad centre
E_DESC0, E_DESC_K = 0.3, 0.3   # descend only within E_DESC0 + E_DESC_K x height of the line
GO_AROUND_H = 0.3       # m, height to climb to when low and off the line
K_XY = 0.8              # 1/s, horizontal position gain
V_REL_MAX = 3.0         # m/s, max speed relative to the pad
K_I = 0.5               # 1/s^2, integral on the horizontal error when low and close
I_ZONE = 0.5            # m, the integral runs only when height and error are below this
I_MAX = 0.4             # m/s, cap on the integral's speed contribution
K_Z = 1.5               # 1/s, height gain
VZ_DOWN, VZ_UP = 0.9, 0.8
VZ_SLOW0 = 0.04         # m/s, smallest descent-speed limit
H_MIN = 0.02            # m: below this (and not dropping) the teacher pushes back up, so it
                        # drops on purpose instead of sinking onto the pad (found 29 Oct:
                        # 9 of 500 landings were slow auto-cut touches)
VZ_LIFT = 0.25          # m/s
T_BRAKE = 2.2           # s, descent limit = (height - H_DROP) / T_BRAKE
K_YAW = 1.5             # 1/s, turn to face the pad's travel direction
PAD_V_TAU = 0.5         # s, how fast the teacher "learns" the pad velocity
# drop gate
D_DROP = 0.06           # m, horizontal distance to the pad centre
DH_DROP = 0.010         # m, |feet height - H_DROP| (drop between 2 and 4 cm)
V_REL_DROP = 0.10       # m/s, horizontal and vertical relative speed
TILT_DROP = np.radians(4.0)
HOLD_STEPS = 3          # decisions the gate must hold before dropping


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Teacher:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pad_v_est = np.zeros(3)
        self.i_xy = np.zeros(2)
        self.gate = 0
        self.phase = "follow"

    def act(self, env):
        """Returns (action dict, info dict). Reads the TRUE state of env.world and the
        command env is currently applying (env.applied)."""
        world = env.world
        s = world.true_state()
        pos, v, R = s["pos"], s["v"], s["R"]
        a = P.POLICY_DT / (PAD_V_TAU + P.POLICY_DT)
        self.pad_v_est = self.pad_v_est + a * (s["pad_v"] - self.pad_v_est)
        rel = s["pad_center"] - pos
        d = float(np.hypot(rel[0], rel[1]))
        hf = float(pos[2] - PAD_TOP_Z - P.REST_HEIGHT)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        tilt = float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))

        # horizontal: pad velocity (as learned) + pull to the approach line (+ integral)
        u = np.array([np.cos(world.plat.psi), np.sin(world.plat.psi)])
        target = s["pad_center"][:2] - u * K_BACK * max(hf, 0.0)
        err = target - pos[:2]
        e = float(np.linalg.norm(err))
        pull = K_XY * err
        n = np.linalg.norm(pull)
        if n > V_REL_MAX:
            pull *= V_REL_MAX / n
        if hf < I_ZONE and e < I_ZONE:
            self.i_xy = self.i_xy + err * P.POLICY_DT
            n_i = np.linalg.norm(K_I * self.i_xy)
            if n_i > I_MAX:
                self.i_xy *= I_MAX / n_i
        v_xy = self.pad_v_est[:2] + pull + K_I * self.i_xy

        # vertical: descend only near the line; climb if low and off the line
        if e < E_DESC0 + E_DESC_K * hf:
            vz = K_Z * (H_DROP - hf)
            phase = "hover" if hf < 0.15 else ("glide" if d < 2.0 else "follow")
        elif hf < GO_AROUND_H:
            vz, phase = K_Z * (GO_AROUND_H - hf) + 0.2, "go-around"
        else:
            vz, phase = 0.0, "follow"
        vz_down = min(VZ_DOWN, max(VZ_SLOW0, (hf - H_DROP) / T_BRAKE))
        vz = float(np.clip(vz, -vz_down, VZ_UP))
        if hf < H_MIN:
            vz = max(vz, VZ_LIFT)

        yaw_rate = float(np.clip(K_YAW * wrap(world.plat.psi - yaw), -ACT_SCALE[3], ACT_SCALE[3]))
        cmd = yaw_rotation(yaw).T @ np.array([v_xy[0], v_xy[1], vz])
        # No teacher-side acceleration limit: the env's rate limit (3 m/s^2) does it.
        # (A limit relative to env.applied turned noise into DRIFT in Step 6b: with flown =
        # label + noise, the applied command ratcheted 0.3 m/s per step away and the drone
        # was "lost" in 26/60 wobble episodes.)
        move = np.clip(np.array([*cmd, yaw_rate]) / ACT_SCALE, -1.0, 1.0)

        # drop gate (must hold HOLD_STEPS decisions)
        vrel = v - s["pad_v"]
        ok = (d < D_DROP and abs(hf - H_DROP) < DH_DROP
              and np.hypot(vrel[0], vrel[1]) < V_REL_DROP and abs(vrel[2]) < V_REL_DROP
              and tilt < TILT_DROP)
        self.gate = self.gate + 1 if ok else 0
        drop = int(self.gate >= HOLD_STEPS)
        if drop:
            phase = "drop"
        self.phase = phase
        return ({"move": move.astype(np.float32), "drop": drop},
                dict(phase=phase, d=d, hf=hf, e=e, gate=self.gate))
