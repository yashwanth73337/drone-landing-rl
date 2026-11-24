"""v7 S1c: the gimbal camera (19 Nov 2026).

Decided 19 Nov (Yashwanth): the lab drone (S500) carries its camera on a GIMBAL, so V7
uses one. V6 (v6_drop/envs/camera.py) bolted the camera to the body at 60 deg below the
drone's nose, so every lean of the drone tilted the camera too (17 deg nose-down while
speeding up -> the camera looked 77 deg down and saw only half as far ahead).

The gimbal here is a 2-axis gimbal:
  * it cancels the drone's PITCH and ROLL: the camera always looks PITCH_DEG below the
    horizon (default 60 deg, the V6/Shin et al. angle; 45 deg is the S2 option);
  * it turns left/right WITH the drone (yaw follows the drone's heading);
  * it hangs from the body at MOUNT_B (body frame), so its position moves with the body.
Image: the same as V6 (512 x 320 grey, 90 deg horizontal view, fx = fy = 256 px).

ASSUMED until measured on the lab drone: the mount position and that the gimbal is ideal
(no lag, no limits). Real gimbals lag a little and stop at their angle limits.

S4g (25 Nov 2026, "yes gimbal"): the gimbal can also TILT during a flight: set_pitch()
changes the angle below the horizon; the camera axes, marker reading and projection of the
next frame use the new angle (WorldV7 sets it from the drone's height before each frame,
see GIMBAL_SCHEDULES in world_v7.py).

Marker reading: the V6 code (pad_board.measure) is reused. It needs the camera axes in
the BODY frame (axis_b, up_b); a gimbal camera's body-frame axes change with every lean,
so render() and pose_world() update them for the attitude of that frame, and
measure_gimbal() corrects the mount position (V6 assumed its own fixed OFFSET_B).
"""
import numpy as np

from v6_drop.envs import pad_board as PB
from v6_drop.envs.camera import OFFSET_B, Camera

PITCH_DEG = 60.0                          # below the horizon (default; 45 = S2 option)
MOUNT_B = np.array([0.15, 0.0, -0.10])    # m, body frame, ASSUMED (chosen in S1c so the
                                          # camera does not see the skids; see the tests)


def heading(R):
    """The drone's yaw: direction of its nose projected on the ground."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


class GimbalCamera(Camera):
    """Camera that keeps a fixed angle below the horizon and follows the drone's yaw."""

    def __init__(self, client, pitch_deg=PITCH_DEG, mount_b=MOUNT_B, **kw):
        super().__init__(client, **kw)
        self.pitch_deg = float(pitch_deg)
        self.mount_b = np.asarray(mount_b, float)
        self._a = np.radians(self.pitch_deg)

    def set_pitch(self, pitch_deg):
        """Tilt the gimbal: the camera looks pitch_deg below the horizon from the next frame."""
        self.pitch_deg = float(pitch_deg)
        self._a = np.radians(self.pitch_deg)

    def world_axes(self, R):
        """(forward, up) of the camera in the WORLD frame for drone attitude R."""
        y = heading(R)
        c, s = np.cos(y), np.sin(y)
        a = self._a
        fwd = np.array([c * np.cos(a), s * np.cos(a), -np.sin(a)])
        up = np.array([c * np.sin(a), s * np.sin(a), np.cos(a)])
        return fwd, up

    def pose_world(self, R, pos):
        """Eye position and (forward, up) in the world. Also stores the camera axes in the
        body frame for this attitude (axis_b, up_b), which marker reading needs."""
        fwd, up = self.world_axes(R)
        self.axis_b, self.up_b = R.T @ fwd, R.T @ up
        self.left_b = np.cross(self.up_b, self.axis_b)
        eye = np.asarray(pos, float) + R @ self.mount_b
        return eye, fwd, up


def measure_gimbal(img, cam, layout=PB.DEFAULT_LAYOUT, det=None):
    """V6 marker reading for a gimbal camera. Call right after cam.render(R, pos) of the
    same frame (the camera axes in the body frame are those of that frame).
    Returns the V6 PadMeasurement with p_body (pad centre in the body frame) corrected
    from V6's fixed camera offset to this camera's mount."""
    m = PB.measure(img, cam, layout, det)
    if m.ok:
        m.p_body = m.p_body - OFFSET_B + cam.mount_b
    return m
