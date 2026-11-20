"""v7 S1d: the V7 Gymnasium environment (20 Nov 2026).

Same observation, action, drop shield, touch modes, sensor noise and press cost as V6
(v6_drop/envs/landing_env.py). Changed parts only:

  part                      V6                         V7 (decided 19 Nov)
  world                     World                      WorldV7 (V7 pad, S500, gimbal camera)
  speed command vx, vy      +-6 m/s  (ACT_SCALE)       +-10 m/s (the lab drone's top speed)
  own-speed scale vx, vy    6        (VEL_SCALE)       10
  speed-change limit        0.3 m/s per step (3 m/s^2) 0.3 default; 0.5 (5 m/s^2) = S2 option
  critic pad-speed scale    V_MAX 4 m/s                v_max (9 default, 8 fallback)
  standing height           0.152 m (camera feet       0.22 m (S500 skids)
                            height, drop shield)
  reward                    Reward                     RewardV7 (same formulas, 0.22 m)
  info                      pad_speed (one number)     pad_speed (START speed) + info["pad"]
                                                       at the end: mode, start / mean / max /
                                                       end speed, driver phase
  reset options             pad_speed, spawn           + mode, plan (pad path for this flight)
  marker reading            PB.measure                 measure_gimbal (gimbal mount)

vz +-1 m/s and yaw rate +-60 deg/s stay as V6. The vz and yaw-rate change limits stay.
The scales are per-object (self.act_scale, ...), so an option never changes another env.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from v6_drop.envs import landing_env as E6
from v6_drop.envs import lmf2_params as P
from v6_drop.envs import pad_board as PB
from v6_drop.envs.landing_env import ACTOR_KEYS, CRITIC_KEYS, OBS_H, OBS_W, rate_limit, shrink
from v6_drop.envs.lee_controller import yaw_rotation
from v6_drop.envs.sensor_noise import SensorNoise, R_to_quat

from .gimbal_camera import PITCH_DEG
from .platform_v7 import V_MAX
from .reward_v7 import RewardV7, geometry_from_body
from .s500 import REST_HEIGHT
from .world_v7 import WorldV7

__all__ = ["LandingEnvV7", "ACTOR_KEYS", "CRITIC_KEYS", "VXY_MAX", "RATE_XY_OPTIONS"]

VXY_MAX = 10.0                     # m/s, vx and vy command range (lab drone top speed)
VZ_MAX = 1.0                       # m/s (V6)
YAW_RATE_MAX = np.radians(60.0)    # rad/s (V6)
RATE_XY_OPTIONS = (0.3, 0.5)       # m/s per decision = 3 or 5 m/s^2 (S2 study)


class LandingEnvV7(E6.LandingEnv):
    metadata = {"render_modes": []}

    def __init__(self, renderer="egl", seed=0, gains_mode="random", layout=PB.DEFAULT_LAYOUT,
                 reward_source="camera", headless=False, drop_shield=None,
                 touch="perfect", touch_cfg=None, air_drop_bonus=False, sensor_noise=None,
                 veto_cost=0.0, pitch_deg=PITCH_DEG, mode="random", v_max=V_MAX, rate_xy=0.3):
        gym.Env.__init__(self)
        assert rate_xy in RATE_XY_OPTIONS, rate_xy
        self.headless = bool(headless)
        self.drop_shield = None if drop_shield is None else float(drop_shield)
        assert not (self.headless and self.drop_shield is not None), "the shield needs camera images"
        self.veto_cost = float(veto_cost)
        assert self.veto_cost >= 0.0
        self.act_scale = np.array([VXY_MAX, VXY_MAX, VZ_MAX, YAW_RATE_MAX])
        self.rate_lim = np.array([rate_xy, rate_xy, E6.RATE_LIMIT[2], E6.RATE_LIMIT[3]])
        self.vel_scale = np.array([VXY_MAX, VXY_MAX, VZ_MAX])
        self.priv_scale = np.array([5, 5, 5, 5, 5, 5, v_max, 5], dtype=np.float64)
        self.rate_xy, self.v_max, self.pitch_deg = float(rate_xy), float(v_max), float(pitch_deg)
        self.last_meas = None
        self.reward_fn = RewardV7(reward_source, air_drop_only=air_drop_bonus)
        self.noise = None if sensor_noise in (None, "none") else SensorNoise(sensor_noise, seed=seed)
        self.world = WorldV7(renderer=renderer, layout=layout, seed=seed, gains_mode=gains_mode,
                             touch=touch, touch_cfg=touch_cfg, noise=self.noise,
                             pitch_deg=pitch_deg, mode=mode, v_max=v_max)
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
        self.applied = np.zeros(4)
        self.asked = np.zeros(4)

    # ---- observation ------------------------------------------------------------------
    def _obs(self, img_full):
        s = self.world.true_state()
        R, v = self.own()
        q = (np.array(s["quat"], dtype=np.float64) if self.noise is None else R_to_quat(R))
        if q[3] < 0:
            q = -q
        state = np.concatenate([(R.T @ v) / self.vel_scale, q, self.applied / self.act_scale])
        priv = np.concatenate([s["rel_pos_body"], s["rel_vel_body"],
                               [self.world.plat.v],
                               [s["pos"][2] - s["pad_center"][2]]]) / self.priv_scale
        return {"image": shrink(img_full),
                "state": state.astype(np.float32),
                "priv": priv.astype(np.float32)}

    # ---- gym API ------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.world.reseed(int(seed))
        options = options or {}
        out = self.world.reset(pad_speed=options.get("pad_speed"), spawn=options.get("spawn"),
                               mode=options.get("mode"), plan=options.get("plan"))
        s = self.world.true_state()
        yaw = np.arctan2(s["R"][1, 0], s["R"][0, 0])
        v_head = yaw_rotation(yaw).T @ s["v"]
        self.applied = np.clip(np.array([*v_head, 0.0]), -self.act_scale, self.act_scale)
        self.asked = self.applied.copy()
        if self.noise is not None and not self.headless:
            out["image"] = self.noise.image(out["image"])
            out["measurement"] = self.world.measure(out["image"])
        self.last_frame = None if self.headless else out["image"]
        self.reward_fn.reset(out["measurement"], s)
        self.last_meas = out["measurement"]
        self.world.set_camera_feet(self.cam_feet(out["measurement"]))
        info = dict(spawn=out["spawn"], tries=out["tries"], pad_speed=self.world.plat.v,
                    pad_mode=self.world.plat.mode, start_marker_ids=list(out["measurement"].ids))
        return self._obs(out["image"]), info

    def cam_feet(self, m):
        """Feet (skid) height above the pad top from a confident camera measurement and the
        drone's own orientation; None if not confident."""
        if m is None or not (m.ok and m.confident):
            return None
        _, h = geometry_from_body(m.p_body, self.own()[0])
        return float(h - REST_HEIGHT)

    def shield_check(self):
        """V6 drop shield with the S500 standing height."""
        m = self.last_meas
        if m is None or not (m.ok and m.confident):
            return False, None, None
        R, v = self.own()
        _, h = geometry_from_body(m.p_body, R)
        h_feet = max(h - REST_HEIGHT, 0.0)
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
        self.asked = move * self.act_scale
        self.applied = rate_limit(self.applied, self.asked, self.rate_lim)
        term, trunc, info = self.world.step(self.applied.copy(), drop=drop)
        if self.headless:
            obs = self._obs(np.zeros((self.world.cam.h, self.world.cam.w), np.uint8))
            self.world.set_camera_feet(None)
            info.update(applied=self.applied.copy(), asked=self.asked.copy(), drop=drop,
                        reward_terms=None, **sh)
            return obs, 0.0, term, trunc, info
        img = self.world.render()
        meas = None
        if self.noise is not None:
            img = self.noise.image(img)
        self.last_frame = img
        obs = self._obs(img)
        if info["outcome"] is None:
            meas = self.world.measure(img)
        self.last_meas = meas
        self.world.set_camera_feet(self.cam_feet(meas))
        reward, terms = self.reward_fn.step(meas, self.world.true_state(), info["outcome"],
                                            info.get("touchdown"))
        terms["veto"] = -self.veto_cost if veto else 0.0
        reward += terms["veto"]
        info.update(applied=self.applied.copy(), asked=self.asked.copy(), drop=drop,
                    reward_terms=terms, **sh)
        return obs, reward, term, trunc, info
