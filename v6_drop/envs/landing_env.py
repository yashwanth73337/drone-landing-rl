"""v6_drop Step 4: the Gymnasium environment (what the policy sees and what it may do).

Decisions (Yashwanth, 29 Oct):
  O1 image: the 512 x 320 camera frame shrunk to 256 x 160 (cv2 INTER_AREA), uint8
  O2 state: own body velocity (3) + orientation quaternion (4) + the command the drone is
     ACTUALLY executing after the rate limit (4)  -> 11 numbers
  O3 no sensor noise in Stage 1 (added later as one separate change)
  A1 ranges: vx, vy +-6 m/s; vz +-1 m/s; yaw rate +-60 deg/s (heading frame)
  A2 rate limit per decision: vx, vy, vz change <= 0.3 m/s; yaw rate <= 30 deg/s
  A3 drop: a SEPARATE yes/no output (Discrete(2)). The policy's drop head starts with a
     low probability (~0.5% per decision); that is set in the policy (Step 7), not here.
  P1 "priv": true relative state for the CRITIC ONLY (asymmetric critic). ACTOR_KEYS
     lists what the actor may read; "priv" is never in it.

Observation (Dict):
  image  uint8   (160, 256)  grey camera frame, shrunk
  state  float32 (11,)       [v_body / VEL_SCALE (3), q = (x, y, z, w) with w >= 0 (4),
                              applied command / ACT_SCALE (4)]
  priv   float32 (8,)        [pad centre in the body frame (3) / 5 m,
                              pad velocity relative to the drone, body frame (3) / 5 m/s,
                              pad speed / 4 m/s, drone centre height above the pad top / 5 m]
Action (Dict):
  move   float32 (4,) in [-1, 1]  -> target = move * ACT_SCALE, then rate-limited
  drop   Discrete(2)               1 = cut the motors now (ends the episode, C1 rules)

Reward (Step 5, envs/reward.py): camera-based fair progress + visibility and time costs
+ end marks. info["reward_terms"] has the breakdown. reward_source="true" = ablation.
The applied command at reset = the drone's current heading-frame velocity (clipped to the
ranges), yaw rate 0, so the controller does not brake hard at the first decision.

Drop SHIELD (Step 7e, decided 30 Oct, option C; Alshiekh et al. 2018 "Safe RL via
shielding"): with drop_shield = L (m/s), a drop the policy asks for is carried out only if a
physics check on ONBOARD data predicts a safe impact:
  h      feet height above the pad top from the last CONFIDENT ArUco + solvePnP measurement
         (the camera frame the policy just saw) and the drone's own orientation (IMU)
  sink   the drone's own downward speed (IMU/EKF); Stage 1 pads move only horizontally,
         so this is the relative sink speed
  pred = sqrt(sink^2 + 2 g h) <= L     (same formula as the safe30m gate; L = 0.8 there)
No confident measurement -> no drop. A blocked drop = the drone keeps flying this step
(drop = 0); info: drop_requested, shield_veto, shield_pred (m/s), shield_feet (m).

TOUCH mode (T1, 7 Nov): touch="perfect" (default) = the old simulator auto-cut on touch;
touch="sensor" = the drone's own touch detector (envs/touch_detector.py). After every camera
frame this env gives the world the frame's feet height (cam_feet: last confident ArUco +
solvePnP measurement + own orientation, the same geometry as the shield; None if not
confident). headless=True has no frames, so the detector cannot fire there.
sensor_noise (N1, 10 Nov): None (default, exact sensing as before) or a level name "low" /
"medium" / "high" (envs/sensor_noise.py). The drone's OWN estimates get noise: the student's
picture (camera grain) and state (speed, orientation), the drop shield (camera height + own
orientation + own sink), the touch detector (camera height + own vertical speed). The reward
and the referee keep the truth. The student is not retrained (evaluation-only robustness).
air_drop_bonus=True (T4, 9 Nov): the reward's drop bonus only for drops with no touch
before (envs/reward.py, air_drop_only). Default False = the old reward exactly.
drop_shield=None (default) = no shield, exactly the old behaviour.

Press cost (Step 7f, decided 31 Oct, F1; the 'punish the learner when the shield steps
in' variant of shielding): with veto_cost = c, every drop press the shield BLOCKS costs c
(reward - c, info reward_terms['veto'] = -c). Reason: in ppo2 a blocked press was free, so
the policy drifted to pressing at every low step ("press and wait") and some clean flights
hovered until the timeout. A carried-out drop and a step without a press cost nothing.
veto_cost=0.0 (default) = exactly the old behaviour.
"""
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import pad_board as PB
from .lee_controller import yaw_rotation
from . import lmf2_params as P
from .reward import Reward, geometry_from_body
from .platform_s1 import V_MAX
from .world import World
from .sensor_noise import SensorNoise, R_to_quat

OBS_W, OBS_H = 256, 160                                    # O1
ACT_SCALE = np.array([6.0, 6.0, 1.0, np.radians(60.0)])     # A1: m/s, m/s, m/s, rad/s
RATE_LIMIT = np.array([0.3, 0.3, 0.3, np.radians(30.0)])    # A2: change per decision
VEL_SCALE = np.array([6.0, 6.0, 1.0])                       # body-velocity normalisation
PRIV_SCALE = np.array([5, 5, 5, 5, 5, 5, V_MAX, 5], dtype=np.float64)
ACTOR_KEYS = ("image", "state")                             # P1: never "priv"
CRITIC_KEYS = ("image", "state", "priv")


def shrink(gray):
    """512 x 320 -> 256 x 160 area average (an exact 2 x 2 mean for this factor)."""
    return cv2.resize(gray, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA)


def rate_limit(applied, target, limit=RATE_LIMIT):
    """Move 'applied' towards 'target' by at most 'limit' per component."""
    return applied + np.clip(target - applied, -limit, limit)


class LandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, renderer="egl", seed=0, gains_mode="random", layout=PB.DEFAULT_LAYOUT,
                 reward_source="camera", headless=False, drop_shield=None,
                 touch="perfect", touch_cfg=None, air_drop_bonus=False, sensor_noise=None, veto_cost=0.0):
        """headless=True (Step 6): no camera image per step (blank image), reward 0 and
        info['reward_terms'] None. ONLY for fast teacher evaluation; never for training.
        Episode starts still use the camera (start rule: >= 1 marker detected)."""
        super().__init__()
        self.headless = bool(headless)
        self.drop_shield = None if drop_shield is None else float(drop_shield)
        assert not (self.headless and self.drop_shield is not None), "the shield needs camera images"
        self.veto_cost = float(veto_cost)
        assert self.veto_cost >= 0.0
        self.last_meas = None                      # measurement of the frame the policy sees
        self.reward_fn = Reward(reward_source, air_drop_only=air_drop_bonus)
        self.noise = None if sensor_noise in (None, "none") else SensorNoise(sensor_noise, seed=seed)   # N1
        self.world = World(renderer=renderer, layout=layout, seed=seed, gains_mode=gains_mode,
                           touch=touch, touch_cfg=touch_cfg, noise=self.noise)
        f32 = np.float32
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (OBS_H, OBS_W), np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (11,), f32),
            "priv": spaces.Box(-np.inf, np.inf, (8,), f32),
        })
        self.action_space = spaces.Dict({
            "move": spaces.Box(-1.0, 1.0, (4,), f32),
            "drop": spaces.Discrete(2),
        })
        self.applied = np.zeros(4)                 # command the drone is actually executing
        self.asked = np.zeros(4)                   # what the policy last asked for (scaled)

    def close(self):
        self.world.close()

    # ---- observation ------------------------------------------------------------------
    def own(self):
        """N1: the drone's own orientation and velocity estimate (exact if no noise)."""
        s = self.world.true_state()
        if self.noise is None:
            return s["R"], s["v"]
        return self.noise.rotation(s["R"]), self.noise.velocity(s["v"])

    def _obs(self, img_full):
        s = self.world.true_state()
        R, v = self.own()
        q = (np.array(s["quat"], dtype=np.float64) if self.noise is None
             else R_to_quat(R))                            # PyBullet order (x, y, z, w)
        if q[3] < 0:
            q = -q                                         # q and -q are the same rotation
        state = np.concatenate([(R.T @ v) / VEL_SCALE, q, self.applied / ACT_SCALE])
        priv = np.concatenate([s["rel_pos_body"], s["rel_vel_body"],
                               [self.world.plat.v],
                               [s["pos"][2] - s["pad_center"][2]]]) / PRIV_SCALE
        return {"image": shrink(img_full),
                "state": state.astype(np.float32),
                "priv": priv.astype(np.float32)}

    # ---- gym API ------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.world.reseed(int(seed))
        options = options or {}
        out = self.world.reset(pad_speed=options.get("pad_speed"), spawn=options.get("spawn"))
        s = self.world.true_state()
        yaw = np.arctan2(s["R"][1, 0], s["R"][0, 0])
        v_head = yaw_rotation(yaw).T @ s["v"]
        self.applied = np.clip(np.array([*v_head, 0.0]), -ACT_SCALE, ACT_SCALE)
        self.asked = self.applied.copy()
        if self.noise is not None and not self.headless:   # N1: the drone's first frame is grainy
            out["image"] = self.noise.image(out["image"])
            out["measurement"] = PB.measure(out["image"], self.world.cam, self.world.layout, self.world.det)
        self.last_frame = None if self.headless else out["image"]          # V1: shown in videos
        self.reward_fn.reset(out["measurement"], s)
        self.last_meas = out["measurement"]
        self.world.set_camera_feet(self.cam_feet(out["measurement"]))       # T1
        info = dict(spawn=out["spawn"], tries=out["tries"], pad_speed=self.world.plat.v,
                    start_marker_ids=list(out["measurement"].ids))
        return self._obs(out["image"]), info

    def cam_feet(self, m):
        """T1: feet height above the pad top (m) from a camera measurement and the drone's
        own orientation; None if the measurement is not confident."""
        if m is None or not (m.ok and m.confident):
            return None
        _, h = geometry_from_body(m.p_body, self.own()[0])
        return float(h - P.REST_HEIGHT)

    def shield_check(self):
        """(allowed, predicted impact m/s or None, feet height m or None) from onboard data
        only: the last confident camera measurement + the drone's own orientation and
        vertical velocity."""
        m = self.last_meas
        if m is None or not (m.ok and m.confident):
            return False, None, None
        R, v = self.own()                                  # own R and v (IMU / EKF; N1 noisy)
        _, h = geometry_from_body(m.p_body, R)
        h_feet = max(h - P.REST_HEIGHT, 0.0)
        sink = max(-float(v[2]), 0.0)
        pred = float(np.sqrt(sink * sink + 2.0 * P.G * h_feet))
        return pred <= self.drop_shield, pred, h_feet

    def step(self, action):
        move = np.clip(np.asarray(action["move"], dtype=np.float64), -1.0, 1.0)
        drop_req = bool(int(action["drop"]))
        veto, pred, h_feet = False, None, None
        if drop_req and self.drop_shield is not None:
            ok, pred, h_feet = self.shield_check()
            veto = not ok
        drop = drop_req and not veto
        sh = dict(drop_requested=drop_req, shield_veto=veto, shield_pred=pred, shield_feet=h_feet)
        self.asked = move * ACT_SCALE
        self.applied = rate_limit(self.applied, self.asked)
        term, trunc, info = self.world.step(self.applied.copy(), drop=drop)
        if self.headless:
            obs = self._obs(np.zeros((self.world.cam.h, self.world.cam.w), np.uint8))
            self.world.set_camera_feet(None)                                # T1: no frames
            info.update(applied=self.applied.copy(), asked=self.asked.copy(), drop=drop,
                        reward_terms=None, **sh)
            return obs, 0.0, term, trunc, info
        img = self.world.render()
        if self.noise is not None:                         # N1: camera grain
            img = self.noise.image(img)
        self.last_frame = img                              # V1: the picture the drone used (videos)
        obs = self._obs(img)
        meas = None
        if info["outcome"] is None:                        # no camera reward after the end
            meas = PB.measure(img, self.world.cam, self.world.layout, self.world.det)
        self.last_meas = meas
        self.world.set_camera_feet(self.cam_feet(meas))                     # T1
        reward, terms = self.reward_fn.step(meas, self.world.true_state(), info["outcome"],
                                            info.get("touchdown"))
        terms["veto"] = -self.veto_cost if veto else 0.0           # Step 7f press cost
        reward += terms["veto"]
        info.update(applied=self.applied.copy(), asked=self.asked.copy(), drop=drop,
                    reward_terms=terms, **sh)
        return obs, reward, term, trunc, info
