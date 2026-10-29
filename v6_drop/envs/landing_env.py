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
"""
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import pad_board as PB
from .lee_controller import yaw_rotation
from .reward import Reward
from .platform_s1 import V_MAX
from .world import World

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
                 reward_source="camera", headless=False):
        """headless=True (Step 6): no camera image per step (blank image), reward 0 and
        info['reward_terms'] None. ONLY for fast teacher evaluation; never for training.
        Episode starts still use the camera (start rule: >= 1 marker detected)."""
        super().__init__()
        self.headless = bool(headless)
        self.reward_fn = Reward(reward_source)
        self.world = World(renderer=renderer, layout=layout, seed=seed, gains_mode=gains_mode)
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
    def _obs(self, img_full):
        s = self.world.true_state()
        R, v = s["R"], s["v"]
        q = np.array(s["quat"], dtype=np.float64)          # PyBullet order (x, y, z, w)
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
        self.reward_fn.reset(out["measurement"], s)
        info = dict(spawn=out["spawn"], tries=out["tries"], pad_speed=self.world.plat.v,
                    start_marker_ids=list(out["measurement"].ids))
        return self._obs(out["image"]), info

    def step(self, action):
        move = np.clip(np.asarray(action["move"], dtype=np.float64), -1.0, 1.0)
        drop = bool(int(action["drop"]))
        self.asked = move * ACT_SCALE
        self.applied = rate_limit(self.applied, self.asked)
        term, trunc, info = self.world.step(self.applied.copy(), drop=drop)
        if self.headless:
            obs = self._obs(np.zeros((self.world.cam.h, self.world.cam.w), np.uint8))
            info.update(applied=self.applied.copy(), asked=self.asked.copy(), drop=drop,
                        reward_terms=None)
            return obs, 0.0, term, trunc, info
        img = self.world.render()
        obs = self._obs(img)
        meas = None
        if info["outcome"] is None:                        # no camera reward after the end
            meas = PB.measure(img, self.world.cam, self.world.layout, self.world.det)
        reward, terms = self.reward_fn.step(meas, self.world.true_state(), info["outcome"],
                                            info.get("touchdown"))
        info.update(applied=self.applied.copy(), asked=self.asked.copy(), drop=drop,
                    reward_terms=terms)
        return obs, reward, term, trunc, info
