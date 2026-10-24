"""Diagnostic only (no training, changes no repo file): does the moving pad
carry a drone resting on it? Motors OFF. The drone is placed on the pad with
the pad's velocity, then the pad moves for 1 s.

Modes:
  F0  current sim: mass-0 box, pose reset every physics step (teleport)
  F1  F0 plus resetBaseVelocity on the box every physics step
  F2  heavy dynamic box (1000 kg) driven by a fixed constraint

Run from the repo root:
  python -m v5_shin.scripts.diag_pad_carry
"""
import numpy as np
import pybullet as p

from v5_shin.envs import lmf2_params as P
from v5_shin.envs.platform import PAD_SIZE, PAD_TOP_Z
from v5_shin.envs import quad as Q

DT = P.PHYSICS_DT          # 0.01 s
STEPS = 100                # 1 s
G = 9.81

# name: (v0 m/s, accel m/s^2, yaw rate rad/s)
CASES = {
    "still": (0.0, 0.0, 0.0),
    "v2":    (2.0, 0.0, 0.0),
    "v5":    (5.0, 0.0, 0.0),
    "v8":    (8.0, 0.0, 0.0),
    "acc5":  (0.0, 5.0, 0.0),                 # worst dv at c = 1: 0.5 m/s per 0.1 s
    "turn":  (8.0, 0.0, np.radians(30.0)),    # W_CLIP, about 4.2 m/s^2 sideways
}


def pad_vel(v, psi):
    return np.array([v * np.cos(psi), v * np.sin(psi), 0.0])


def run(mode, v0, acc, w):
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -G, physicsClientId=cid)
    p.setTimeStep(DT, physicsClientId=cid)

    half = [PAD_SIZE / 2, PAD_SIZE / 2, PAD_TOP_Z / 2]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
    pad = p.createMultiBody(1000.0 if mode == "F2" else 0.0, col, -1,
                            [0, 0, PAD_TOP_Z / 2], physicsClientId=cid)
    con = None
    if mode == "F2":
        con = p.createConstraint(pad, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                 [0, 0, PAD_TOP_Z / 2], physicsClientId=cid)

    drone = p.loadURDF(Q.URDF, [0, 0, 3], flags=p.URDF_USE_INERTIA_FROM_FILE,
                       physicsClientId=cid)
    p.changeDynamics(drone, -1, linearDamping=P.LINEAR_DAMPING, physicsClientId=cid)
    zmin = min(p.getAABB(drone, l, physicsClientId=cid)[0][2]
               for l in range(-1, p.getNumJoints(drone, physicsClientId=cid)))
    z0 = PAD_TOP_Z + (3.0 - zmin) + 0.005
    p.resetBasePositionAndOrientation(drone, [0, 0, z0], [0, 0, 0, 1], physicsClientId=cid)
    p.resetBaseVelocity(drone, pad_vel(v0, 0.0).tolist(), [0, 0, w], physicsClientId=cid)

    info = (p.getDynamicsInfo(drone, -1, physicsClientId=cid),
            p.getDynamicsInfo(pad, -1, physicsClientId=cid))

    x = y = psi = 0.0
    v = v0
    rel0 = None
    vrel_hist, contact_hist, track_err = [], [], 0.0
    for k in range(STEPS):
        # pad kinematics: same exact unicycle arc as Platform.integrate
        v = v + acc * DT
        if abs(w) < 1e-9:
            x += v * np.cos(psi) * DT
            y += v * np.sin(psi) * DT
        else:
            x += v / w * (np.sin(psi + w * DT) - np.sin(psi))
            y += v / w * (np.cos(psi) - np.cos(psi + w * DT))
        psi += w * DT
        q = p.getQuaternionFromEuler([0, 0, psi])
        target = [x, y, PAD_TOP_Z / 2]
        if mode in ("F0", "F1"):
            p.resetBasePositionAndOrientation(pad, target, q, physicsClientId=cid)
            if mode == "F1":
                p.resetBaseVelocity(pad, pad_vel(v, psi).tolist(), [0, 0, w],
                                    physicsClientId=cid)
        else:
            p.changeConstraint(con, jointChildPivot=target, jointChildFrameOrientation=q,
                               maxForce=1e7, physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)

        # pad state: actual body for F2, analytic for F0/F1 (as the sim does)
        if mode == "F2":
            pp, pq = p.getBasePositionAndOrientation(pad, physicsClientId=cid)
            pv, pw = p.getBaseVelocity(pad, physicsClientId=cid)
            ppsi = p.getEulerFromQuaternion(pq)[2]
            track_err = max(track_err, float(np.hypot(pp[0] - x, pp[1] - y)))
            pc, pvel, pwz = np.array([pp[0], pp[1]]), np.array(pv), pw[2]
        else:
            ppsi, pc, pvel, pwz = psi, np.array([x, y]), pad_vel(v, psi), w

        dp, dv = p.getBasePositionAndOrientation(drone, physicsClientId=cid)[0], \
            p.getBaseVelocity(drone, physicsClientId=cid)[0]
        r = np.array(dp[:2]) - pc
        c, s = np.cos(ppsi), np.sin(ppsi)
        rel_pad = np.array([c * r[0] + s * r[1], -s * r[0] + c * r[1]])
        v_point = pvel + np.array([-pwz * r[1], pwz * r[0], 0.0])   # pad surface velocity under the drone
        vrel = np.array(dv) - v_point
        contact = len(p.getContactPoints(drone, pad, physicsClientId=cid)) > 0
        if rel0 is None and contact:
            rel0 = rel_pad.copy()
        vrel_hist.append(float(np.linalg.norm(vrel)))
        contact_hist.append(contact)
        z_end = dp[2]

    p.disconnect(cid)
    slip = float(np.linalg.norm(rel_pad - rel0)) if rel0 is not None else float("nan")
    on_pad = bool(np.all(np.abs(rel_pad) <= PAD_SIZE / 2) and z_end > PAD_TOP_Z - 0.05)
    return dict(slip=slip, vrel_last=float(np.mean(vrel_hist[-50:])),
                contact=float(np.mean(contact_hist)), on_pad=on_pad,
                track=track_err, info=info)


def main():
    first = True
    print(f"{'mode':4} {'case':6} {'slip_m':>7} {'vrel_m/s':>9} {'contact':>8} "
          f"{'on_pad':>7} {'F2_track_m':>10}")
    for mode in ("F0", "F1", "F2"):
        for name, (v0, acc, w) in CASES.items():
            r = run(mode, v0, acc, w)
            if first:
                d, b = r["info"]
                print(f"# drone lateral friction {d[1]}, restitution {d[5]}; "
                      f"pad lateral friction {b[1]}, restitution {b[5]}")
                first = False
            print(f"{mode:4} {name:6} {r['slip']:7.3f} {r['vrel_last']:9.3f} "
                  f"{r['contact']:8.2f} {str(r['on_pad']):>7} {r['track']:10.4f}")
    print("slip = pad-frame drift of the drone since first contact; "
          "vrel = mean |v_drone - v_pad| over the last 0.5 s; "
          "contact = fraction of steps in contact.")


if __name__ == "__main__":
    main()
