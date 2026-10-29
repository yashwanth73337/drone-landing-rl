"""v6_drop Step 6d diagnostic: WOULD a drop be safe in the states the student reaches?

The student lands safely (95%) but by auto-cut: it never holds still at 3 cm, so the
teacher's strict drop gate (6 cm, 2-4 cm high, < 0.10 m/s, 3 steps) is never met and the
teacher never labels a drop. Before changing any label, this measures whether a WIDER,
physics-based gate gives SAFE drops from the student's own approach.

How: the student flies the dev exam exactly as usual, but a rule that reads the TRUE state
(an "oracle", like the teacher) cuts the motors the first time its gate is met. Nothing is
trained and no label changes; only the moment of the motor cut changes.

Gates (experts/drop_gates.py):
  teacher  the teacher's own gate (sanity check: expect ~0 drops, see drop_gap.py)
  safe20   centre < 20 cm, horizontal speed vs pad < 0.20 m/s, tilt < 6 deg,
           feet 1.5-5 cm above the pad, predicted impact <= 0.9 m/s, 1 step
  safe30   centre < 30 cm, < 0.30 m/s, tilt < 8 deg, feet 1.5-6 cm, impact <= 0.9 m/s, 1 step
  safe30m  safe30 with impact <= 0.8 m/s (decided 30 Oct; the new label + teacher rule)
Results 29 Oct (student dagger_r3/last.pt): teacher 0 drops; safe20 23/23 safe; safe30 59/59.
Predicted impact = sqrt(sink^2 + 2 g x feet height) (free fall from the cut; the env's
success limit is 1.0 m/s, so 0.9 leaves a margin).

Run (from the repo root; one process per gate, they can run at the same time):
  python -m v6_drop.scripts.oracle_drop --ckpt v6_drop/results/runs/dagger_r3/last.pt --gate safe20
Output: runs/<run>/oracle_<gate>_exam_<n>.csv and _summary.txt (prints the summary at the end).
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from v6_drop.envs import landing_env as E
from v6_drop.envs import lmf2_params as P
from v6_drop.experts import drop_gates as G
from v6_drop.scripts import eval_student as ES

VARIANTS = G.GATES            # gates live in experts/drop_gates.py (30 Oct); safe30m added


def predicted_impact(rec):
    return float(G.predicted_impact(rec["vz_rel"], rec["hf"]))


class OracleGate:
    """Forces the motor cut when the gate holds (on the exam's per-step record, true state)."""

    def __init__(self, gate):
        self.name = gate
        self.c = G.GATES[gate]
        self.reset()

    def reset(self):
        self.run = 0

    def ok(self, rec):
        return bool(G.ok(self.c, d=rec["d"], hf=rec["hf"], vxy=rec["vrel"], vz=rec["vz_rel"],
                         tilt=np.radians(rec["tilt"])))

    def __call__(self, rec):
        self.run = self.run + 1 if self.ok(rec) else 0
        return self.run >= self.c["hold"]


def run(model, ckpt, gate_name, n, device, seed0=9000):
    out = os.path.dirname(os.path.abspath(ckpt))
    rule = OracleGate(gate_name)
    env = E.LandingEnv(renderer="egl", seed=0)
    rows, t0 = [], time.time()
    try:
        for i in range(n):
            log, info, info0, _ = ES.fly(env, model, seed0 + i, device, drop_rule=rule)
            td = info.get("touchdown") or {}
            last = log[-1]
            forced = bool(last.get("forced_drop"))
            rows.append(dict(seed=seed0 + i, pad_speed=round(info0["pad_speed"], 3), outcome=info["outcome"],
                             reason=td.get("reason"), forced_drop=int(forced),
                             impact=None if td.get("impact") is None else round(td["impact"], 3),
                             predicted_impact=round(predicted_impact(last), 3) if forced else None,
                             offset_cm=None if not td.get("land_xy") else round(100 * float(np.linalg.norm(td["land_xy"])), 1),
                             cut_d_cm=round(100 * last["d"], 1) if forced else None,
                             cut_hf_cm=round(100 * last["hf"], 1) if forced else None,
                             cut_vxy=round(last["vrel"], 3) if forced else None,
                             cut_sink=round(-last["vz_rel"], 3) if forced else None,
                             time_s=round(len(log) * P.POLICY_DT, 1)))
            if (i + 1) % 10 == 0:
                k = sum(r["outcome"] == "success" for r in rows)
                f = sum(r["forced_drop"] for r in rows)
                print(f"[oracle {gate_name}] {i + 1}/{n} flown, safe {k}, forced drops {f} "
                      f"({time.time() - t0:.0f} s)", flush=True)
    finally:
        env.close()
    base = os.path.join(out, f"oracle_{gate_name}_exam_{n}")
    with open(base + ".csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    txt = summary(rows, ckpt, gate_name, n, seed0)
    open(base + "_summary.txt", "w").write(txt + "\n")
    print(txt, flush=True)
    return rows


def summary(rows, ckpt, gate_name, n, seed0):
    k = sum(r["outcome"] == "success" for r in rows)
    lo, hi = ES.wilson(k, n)
    outs = {}
    for r in rows:
        outs[r["outcome"]] = outs.get(r["outcome"], 0) + 1
    dr = [r for r in rows if r["forced_drop"]]
    lines = [f"Oracle-drop exam, gate '{gate_name}' {G.GATES[gate_name]}",
             f"student {os.path.abspath(ckpt)}, seeds {seed0}..{seed0 + n - 1}",
             f"safe landings: {k}/{n} = {100 * k / n:.1f}%  (95% CI {lo:.1f}-{hi:.1f}%)",
             f"outcomes: {outs}",
             f"forced drops: {len(dr)}/{n} flights; of these safe {sum(r['outcome'] == 'success' for r in dr)}"]
    if dr:
        do = {}
        for r in dr:
            do[r["outcome"]] = do.get(r["outcome"], 0) + 1
        imp = np.array([r["impact"] for r in dr if r["impact"] is not None])
        lines += [f"  outcomes of forced drops: {do}",
                  f"  at the cut: centre {np.median([r['cut_d_cm'] for r in dr]):.0f} cm, feet "
                  f"{np.median([r['cut_hf_cm'] for r in dr]):.1f} cm, speed vs pad "
                  f"{np.median([r['cut_vxy'] for r in dr]):.2f} m/s, sinking "
                  f"{np.median([r['cut_sink'] for r in dr]):.2f} m/s (medians)"]
        if len(imp):
            lines.append(f"  impact of forced drops [m/s]: mean {imp.mean():.2f}, max {imp.max():.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gate", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model, ck = ES.load(a.ckpt, a.device)
    print(f"[oracle {a.gate}] loaded {a.ckpt} (epoch {ck['epoch']})", flush=True)
    run(model, a.ckpt, a.gate, a.n, a.device)


if __name__ == "__main__":
    main()
