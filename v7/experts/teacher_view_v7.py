"""v7 S2': the VIEW-KEEPING TEACHER (21 Nov 2026; decided by Yashwanth 21 Nov 01:32).

Why (heading diagnostic of dagger_p60_r2, 21 Nov): the S2a teacher (teacher_v7.TeacherV7)
flies where the camera cannot see. From rest it falls 6 to 16 m behind a fast pad and holds
its start height (2 to 5 m). A 60 deg gimbal camera at height h sees only ~1.9 h ahead, so
the pad leaves the picture, and the teacher acts on turns and speed changes the student
cannot see. Labels there are impossible for a camera-only student.

Rule (asymmetric imitation): the privileged teacher should not rely on what the camera
cannot show. So, ONLY while the pad is FAR (d > D_VIEW m, horizontal):
  1. CLIMB so that the pad centre stays below the TOP edge of the picture, with a margin:
         height_needed = tan(top-edge angle + MARGIN_DEG) x d,   capped at H_VIEW_MAX
     top-edge angle = camera pitch - half the vertical view (60 deg: 60 - 32 = 28 deg below
     the horizon). Climb speed up to the env's vz limit (1 m/s). Never forces a descent.
  2. TURN THE NOSE TOWARD THE PAD (bearing) instead of the pad's heading, so a turning pad
     stays inside the 90 deg wide picture. Blended back to the pad's heading between
     D_YAW1 and D_YAW0 m.
Near the pad (d <= D_VIEW, and below D_YAW0 for the yaw) the actions are EXACTLY those of
TeacherV7 (a test checks this): descent, wait, abort, drop rule unchanged.

Tuned in Claude's sandbox, 21 Nov (headless, tuning seeds 31000 to 31199, 60 deg geometry):
  * cap H_VIEW_MAX: markers are read reliably up to a slant distance of ~8 to 9 m (image
    check, drone behind a standing pad). Caps 8 m and 6 m gave the same view share and
    safety (the 1 m/s climb rarely reaches the cap), so 6 m keeps the markers readable.
  * nose hand-over D_YAW0, D_YAW1: 2 to 3 m gave 3/100 extra tip-overs on waves (the nose
    was still swinging with the bearing at the start of the descent); 4 to 6 m: 100/100.
  * pad centre in the picture while far (> 3 m): 55 % -> ~72 %. Far steps without the pad
    were almost all in the first 5 s (from rest the pad pulls away faster than the 1 m/s
    climb) and none were to the side. Near the pad (0.6 to 3 m): 97 -> 99 %.
"""
import numpy as np

from v6_drop.envs.camera import vfov_deg
from v6_drop.experts.teacher import K_YAW, K_Z, wrap

from ..envs.platform_v7 import PAD_TOP_Z
from ..envs.s500 import lowest_z
from .teacher_v7 import TeacherV7

D_VIEW = 2.5          # m, horizontal distance to the pad centre: beyond it the view rules act
D_RAMP = 1.0          # m, the climb rule fades in between D_VIEW and D_VIEW + D_RAMP
MARGIN_DEG = 4.0      # deg, keep the pad centre this far below the top edge of the picture
H_VIEW_MAX = 6.0      # m, cap on the height the view rule asks for (lowest point above pad top)
D_YAW0, D_YAW1 = 4.0, 6.0   # m: nose to the pad heading below D_YAW0, to the bearing above D_YAW1


def top_edge_deg(pitch_deg):
    """Angle of the picture's top edge below the horizon (gimbal camera)."""
    return float(pitch_deg - vfov_deg() / 2.0)


class TeacherV7View(TeacherV7):
    """TeacherV7 + the two view rules while the pad is far. act(env) as TeacherV7."""

    def __init__(self, pitch_deg=60.0, margin_deg=MARGIN_DEG, h_max=H_VIEW_MAX,
                 d_view=D_VIEW, climb=True, yaw_bearing=True, d_yaw=(D_YAW0, D_YAW1), **kw):
        super().__init__(**kw)
        self.pitch_deg = float(pitch_deg)
        self.k_view = float(np.tan(np.radians(top_edge_deg(pitch_deg) + margin_deg)))
        self.h_max = float(h_max)
        self.d_view = float(d_view)
        self.climb = bool(climb)
        self.yaw_bearing = bool(yaw_bearing)
        self.d_yaw0, self.d_yaw1 = (float(x) for x in d_yaw)
        assert self.d_yaw1 > self.d_yaw0

    def h_needed(self, d):
        """Lowest-point height the view rule asks for at horizontal distance d (0 when near)."""
        w = float(np.clip((d - self.d_view) / D_RAMP, 0.0, 1.0))
        return w * min(self.h_max, self.k_view * d)

    def act(self, env):
        a, info = super().act(env)
        s = env.world.true_state()
        scale = env.act_scale
        pos, R = s["pos"], s["R"]
        rel = s["pad_center"] - pos
        d = float(np.hypot(rel[0], rel[1]))
        move = a["move"].astype(np.float64)
        view = ""
        # 1. climb while far (only raises vz; descent stays the V6/V7 rule)
        if self.climb:
            hf = float(lowest_z(R, pos) - PAD_TOP_Z)
            need = self.h_needed(d)
            if need > hf:
                vz_up = float(np.clip(K_Z * (need - hf), 0.0, scale[2]))
                if vz_up / scale[2] > move[2]:
                    move[2] = vz_up / scale[2]
                    view = "climb"
        # 2. nose toward the pad while far
        if self.yaw_bearing and d > self.d_yaw0:
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            w = float(np.clip((d - self.d_yaw0) / (self.d_yaw1 - self.d_yaw0), 0.0, 1.0))
            bearing = float(np.arctan2(rel[1], rel[0]))
            target = s["pad_psi"] + w * wrap(bearing - s["pad_psi"])
            yaw_rate = float(np.clip(K_YAW * wrap(target - yaw), -scale[3], scale[3]))
            move[3] = yaw_rate / scale[3]
        move = np.clip(move, -1.0, 1.0)
        # phase names stay TeacherV7's (record_demos_v7 / dagger_collect_v7 store them as an
        # index into a fixed list); the view rule is reported separately in info["view"]
        info = dict(info, view=view, h_need=self.h_needed(d))
        return ({"move": move.astype(np.float32), "drop": a["drop"]}, info)
