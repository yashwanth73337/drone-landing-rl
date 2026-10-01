"""Gymnasium environment: observation and action interfaces (V5 Block 8; SPEC §5, §10).

Observation (Dict). Every key is present in every mode; the POLICY decides what the
actor may read, via ACTOR_KEYS (the privilege boundary):
  image   uint8 (160, 256)  downsampled gray frame          actor (vision modes)
  u       float32 (7,)      noisy [v_body, q], q = [x, y, z, w] with q_w >= 0   actor
  critic  float32 (13,)     [u_clean (7), s_rel (6)]         critic only   [paper: o_priv]
  target  float32 (6,)      s_rel = [dx_b, dv_b]: platform minus drone, drone body frame.
                            The L_est label only (never an actor input in vision modes)
  s_rel   float32 (6,)      = target; the actor input of variant P (privileged actor)

Action: Box(-1, 1, (4,)) -> heading-frame [vx, vy, vz, yaw_rate] =
        clip(a, -1, 1) * [10, 10, vz_max, pi/3]   (SPEC §5; yaw limit = controller clamp)
        vz_max = 1.0 m/s by default (D16, inferred from Fig. 9 descents <= ~1 m/s);
        P_smoke_s1 used 3.0 and dived at ~4 m/s impact.

Reward: reward_fn="paper" (default) -> envs.reward.ShinReward() (Table III, D3, D4);
        a callable with reset(env) and __call__(env, info); or None -> 0.0 (tests only).
Rendering: exactly one render per reset() and per step() in vision modes; none for P.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..policies.encoder import OBS_H, OBS_W, downsample
from . import dr as DR
from .landing_sim import LandingSim

VZ_MAX_DEFAULT = 1.0                               # m/s, D16
ACTION_SCALE = np.array([10.0, 10.0, VZ_MAX_DEFAULT, np.pi / 3], dtype=np.float64)


def action_scale(vz_max=VZ_MAX_DEFAULT):
    return np.array([10.0, 10.0, float(vz_max), np.pi / 3], dtype=np.float64)
ACTOR_KEYS = {
    "vision": ("image", "u"),          # proposed / ArUco+CNN / ablations
    "privileged": ("u", "s_rel"),      # variant P (SPEC §13): control ceiling
}


def relative_state_body(sim):
    """s_rel = [R^T (p_pad - p), R^T (v_pad - v)] from the TRUE state."""
    s = sim.quad.state()
    Rt = s["R"].T
    return np.concatenate([Rt @ (sim.plat.pad_center() - s["pos"]),
                           Rt @ (sim.plat.velocity() - s["v"])]), s


class ShinLandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mode="vision", c=1.0, seed=0, renderer="egl", egl=None, dr=None,
                 reward_fn="paper", gains_mode="random", vz_max=VZ_MAX_DEFAULT,
                 motor_mode="asym"):
        assert mode in ACTOR_KEYS, mode
        self.mode = mode
        self.c = float(c)
        if reward_fn == "paper":
            from .reward import ShinReward
            reward_fn = ShinReward()
        self.reward_fn = reward_fn
        self.render_images = mode == "vision"
        if not self.render_images:
            renderer = "tiny"                      # never used; avoid an EGL context
        egl = (renderer == "egl") if egl is None else egl
        self.sim = LandingSim(renderer=renderer, egl=egl, seed=seed, dr=dr, gains_mode=gains_mode,
                              motor_mode=motor_mode)      # motor_mode: evaluation diagnostic only
        self.action_scale = action_scale(vz_max)
        f32 = np.float32
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (OBS_H, OBS_W), np.uint8),
            "u": spaces.Box(-np.inf, np.inf, (7,), f32),
            "critic": spaces.Box(-np.inf, np.inf, (13,), f32),
            "target": spaces.Box(-np.inf, np.inf, (6,), f32),
            "s_rel": spaces.Box(-np.inf, np.inf, (6,), f32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (4,), f32)
        self._blank = np.zeros((OBS_H, OBS_W), np.uint8)

    # ---- curriculum hook (Block 10): takes effect at the NEXT reset ---------------
    def set_c(self, c):
        self.c = float(np.clip(c, 0.0, 1.0))

    # ---- gym API -----------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim._seed_streams(seed)           # gains stay as sampled at env init
        c = (options or {}).get("c", self.c)
        self.c_episode = float(c)                  # tags this episode for the curriculum
        info = self.sim.reset(c=c, spawn=(options or {}).get("spawn"))
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset(self)
        obs, extra = self._obs()
        return obs, {**info, **extra, "c": c}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        terminated, truncated, info = self.sim.step(a * self.action_scale)
        obs, extra = self._obs()
        info.update(extra)
        info["action_cmd"] = a * self.action_scale
        info["c_episode"] = self.c_episode
        reward = float(self.reward_fn(self, info)) if self.reward_fn is not None else 0.0
        return obs, reward, terminated, truncated, info

    # ---- observation -----------------------------------------------------------------
    def _obs(self):
        s_rel, s = relative_state_body(self.sim)
        u_clean = DR.measure(s, None, noise=False)
        u = self.sim.measure()
        img = downsample(self.sim.render()) if self.render_images else self._blank
        uv, z = self.sim.cam.project(self.sim.plat.pad_center(), s["R"], s["pos"])
        in_view = bool(z[0] > 0 and 0 <= uv[0, 0] < self.sim.cam.w and 0 <= uv[0, 1] < self.sim.cam.h)
        f = np.float32
        obs = {"image": img, "u": u.astype(f), "critic": np.concatenate([u_clean, s_rel]).astype(f),
               "target": s_rel.astype(f), "s_rel": s_rel.astype(f)}
        return obs, {"pad_centre_in_view": in_view}

    def close(self):
        self.sim.close()


def actor_view(obs, mode):
    """What the actor of `mode` is allowed to read (used by the policy in Block 11)."""
    return {k: obs[k] for k in ACTOR_KEYS[mode]}
