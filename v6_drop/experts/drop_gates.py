"""v6_drop Step 6d: the DROP RULES ("gates") in one place.

A gate says: in this state it is right to cut the motors. It is used for
  * the teacher's drop decision and drop LABEL (experts/teacher.py, Teacher(gate=...))
  * relabelling saved data offline (learn/dataset.py, Demos(..., drop_gate=...))
  * the oracle-drop diagnostic (scripts/oracle_drop.py)
so all three always agree.

Conditions (all must hold, for 'hold' decisions in a row):
  d       horizontal distance drone centre -> pad centre          < d      [m]
  height  |feet height - hc|                                       < hh     [m]
  vxy     horizontal speed relative to the pad                     < vxy    [m/s]
  vz      |vertical speed relative to the pad|                     < vz     [m/s]   (None = no check)
  tilt                                                             < tilt   [rad]
  impact  predicted touch speed sqrt(sink^2 + 2 g feet height)     <= impact [m/s]  (None = no check)

Gates:
  teacher  the original teacher gate (6a): 6 cm, feet 2-4 cm, 0.10 m/s both ways, 4 deg, 3 steps
  safe20   oracle test 29 Oct: 23/23 forced drops safe (20 cm, 0.20, 6 deg, feet 1.5-5 cm, <= 0.9)
  safe30   oracle test 29 Oct: 59/59 forced drops safe (30 cm, 0.30, 8 deg, feet 1.5-6 cm, <= 0.9)
  safe30m  DECIDED 30 Oct (D1 a): safe30 with impact <= 0.8 m/s, so a drop ONE decision early
           (~1.8 cm higher at the usual 0.18 m/s sink) still lands under the 1.0 m/s limit.
           Used for the labels (all data) and the teacher in DAgger round 4 onwards.
"""
import numpy as np

from ..envs import lmf2_params as P

GATES = {
    "teacher": dict(d=0.06, hc=0.03, hh=0.010, vxy=0.10, vz=0.10, tilt=np.radians(4.0), impact=None, hold=3),
    "safe20": dict(d=0.20, hc=0.0325, hh=0.0175, vxy=0.20, vz=None, tilt=np.radians(6.0), impact=0.9, hold=1),
    "safe30": dict(d=0.30, hc=0.0375, hh=0.0225, vxy=0.30, vz=None, tilt=np.radians(8.0), impact=0.9, hold=1),
    "safe30m": dict(d=0.30, hc=0.0375, hh=0.0225, vxy=0.30, vz=None, tilt=np.radians(8.0), impact=0.8, hold=1),
}


def predicted_impact(vz, hf):
    """Touch speed after cutting the motors now: free fall from the feet height with the
    current sinking speed. vz = vertical velocity drone - pad (negative = sinking)."""
    sink = np.maximum(0.0, -np.asarray(vz, dtype=np.float64))
    return np.sqrt(sink * sink + 2.0 * P.G * np.maximum(np.asarray(hf, dtype=np.float64), 0.0))


def ok(gate, d, hf, vxy, vz, tilt):
    """One-step gate test; works on numbers or on numpy arrays (element-wise)."""
    g = GATES[gate] if isinstance(gate, str) else gate
    r = (np.asarray(d) < g["d"]) & (np.abs(np.asarray(hf) - g["hc"]) < g["hh"]) \
        & (np.asarray(vxy) < g["vxy"]) & (np.asarray(tilt) < g["tilt"])
    if g["vz"] is not None:
        r = r & (np.abs(np.asarray(vz)) < g["vz"])
    if g["impact"] is not None:
        r = r & (predicted_impact(vz, hf) <= g["impact"])
    return r


def held(flags, hold):
    """flags (T,) bool -> (T,) bool: True where the last 'hold' flags were all True."""
    out = np.zeros(len(flags), bool)
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        out[i] = run >= hold
    return out
