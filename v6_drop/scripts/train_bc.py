"""v6_drop Step 6c: behaviour cloning (the student copies the teacher).

Loss per clip step that has a label (warm-up steps excluded):
  move: mean squared error between the student's and the teacher's move (both in [-1, 1])
  drop: binary cross-entropy on the drop chance, drop=1 samples weighted DROP_WEIGHT (20x,
        decided 29 Oct) so "never drop" is not a cheap answer
  total = move + DROP_COEF x drop
Validation (episodes with id % 10 == 0) after every epoch. Reported per epoch:
  move MAE in m/s (vx, vy, vz) and deg/s (yaw rate), drop recall (share of teacher drops
  where the student's chance > 0.5) and false drops (share of non-drop steps > 0.5).
Saves runs/<name>/best.pt (lowest validation loss), last.pt, log.csv, config.json.

Step 6d (DAgger, decided 29 Oct): --data takes several built folders (old demos + DAgger
rounds) and --init continues from an earlier student (8 epochs per round, D2).
Step 6d, low weighting (proposed 29 Oct after drop_gap.py): the student never STOPS at 3 cm
(sinks 0.17 m/s through the drop window; the teacher 0.01) and the final metres are a tiny
share of the steps. --low-weight W makes every step with feet below LOW_H (0.3 m, read from
'priv', training only) count W times in the move loss (normalised by the total weight).
W = 1 (default) is exactly the old loss. The log always shows the move error on low steps
("low vz" in m/s), whatever W is.
Step 6d, wider drop rule (decided 30 Oct): --drop-gate safe30m recomputes EVERY drop label of
all the data from the saved true state with the safe30m gate (experts/drop_gates.py); the
move labels do not change. The first lines show the drop-label count before and after.

Run (from the repo root; performance mode first):
  nohup python -m v6_drop.scripts.train_bc --name bc256 > v6_drop/results/bc256.log 2>&1 &
  DAgger round 1:
  nohup python -m v6_drop.scripts.train_bc --name dagger_r1 --epochs 8 \\
        --init v6_drop/results/runs/bc256/best.pt \\
        --data v6_drop/results/bc_data v6_drop/results/dagger_data/r1 > v6_drop/results/dagger_r1.log 2>&1 &
Watch:  tail -f v6_drop/results/bc256.log | grep --line-buffered "^\\[bc"
Stop:   pkill -f "v6_drop.scripts.train_bc"
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from v6_drop.envs import lmf2_params as P
from v6_drop.envs.landing_env import ACT_SCALE, PRIV_SCALE
from v6_drop.learn.dataset import Demos, batch
from v6_drop.learn.student import Student, n_params

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
DROP_WEIGHT = 20.0
DROP_COEF = 1.0
LOW_H = 0.3                    # m, feet height below which a step is "low" (final phase)


def to_t(b, device):
    return {k: torch.as_tensor(v, device=device) for k, v in b.items()}


def feet_height(priv):
    """Feet height above the pad top [m] from the critic's 'priv' (last entry: drone centre
    height above the pad top / PRIV_SCALE[7])."""
    return priv[..., 7] * float(PRIV_SCALE[7]) - P.REST_HEIGHT


def losses(model, b, amp, device, low_weight=1.0):
    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
        move, logit, _, _ = model(b["image"], b["state"].float())
    move, logit = move.float(), logit.float()
    m = b["mask"]
    n = m.sum().clamp(min=1.0)
    low = (feet_height(b["priv"].float()) < LOW_H).float() * m
    wm = m + (low_weight - 1.0) * low                     # = m when low_weight == 1
    l_move = (((move - b["move"]) ** 2).mean(-1) * wm).sum() / wm.sum().clamp(min=1.0)
    y = b["drop"].float()
    w = torch.where(y > 0.5, torch.full_like(y, DROP_WEIGHT), torch.ones_like(y))
    l_drop = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w * m).sum() / n
    stats = dict(move_abs=((move - b["move"]).abs() * m.unsqueeze(-1)).sum((0, 1)).detach(),
                 n=n.detach(), drops=(y * m).sum().detach(),
                 move_abs_low=((move - b["move"]).abs() * low.unsqueeze(-1)).sum((0, 1)).detach(),
                 n_low=low.sum().detach(),
                 drops_hit=(((logit > 0) & (y > 0.5)).float() * m).sum().detach(),
                 nondrop=((1 - y) * m).sum().detach(),
                 false_drop=(((logit > 0) & (y < 0.5)).float() * m).sum().detach())
    return l_move, l_drop, stats


def summarize(acc):
    acc = {k: v.detach().cpu() for k, v in acc.items()}   # GPU sums -> CPU (fixed 29 Oct:
    n = acc["n"]                                           # mixing cuda and cpu tensors crashed)
    scale = torch.as_tensor(ACT_SCALE, dtype=torch.float32)
    mae = (acc["move_abs"] / n) * scale
    mae_low = (acc["move_abs_low"] / acc["n_low"].clamp(min=1)) * scale
    return dict(move_mae_vx=mae[0].item(), move_mae_vy=mae[1].item(), move_mae_vz=mae[2].item(),
                move_mae_yaw_deg=float(np.degrees(mae[3].item())),
                low_mae_vx=mae_low[0].item(), low_mae_vz=mae_low[2].item(),
                low_share=(acc["n_low"] / n).item(),
                drop_recall=(acc["drops_hit"] / acc["drops"].clamp(min=1)).item(),
                false_drop=(acc["false_drop"] / acc["nondrop"].clamp(min=1)).item())


def run_epoch(model, demos, split, rng, bs, device, amp, opt=None, low_weight=1.0):
    pairs = demos.clip_starts(split, rng)
    tot = dict(move=0.0, drop=0.0, n=0)
    acc = None
    model.train(opt is not None)
    for i in range(0, len(pairs), bs):
        b = to_t(batch(demos, pairs[i:i + bs]), device)
        with torch.set_grad_enabled(opt is not None):
            l_move, l_drop, st = losses(model, b, amp, device, low_weight)
            loss = l_move + DROP_COEF * l_drop
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        k = st["n"].item()
        tot["move"] += l_move.item() * k
        tot["drop"] += l_drop.item() * k
        tot["n"] += k
        acc = st if acc is None else {key: acc[key] + st[key] for key in acc}
    out = dict(loss_move=tot["move"] / tot["n"], loss_drop=tot["drop"] / tot["n"])
    out.update(summarize(acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bc256")
    ap.add_argument("--data", nargs="+", default=[os.path.join(RESULTS, "bc_data")],
                    help="one or more built dataset folders (6d: demos + DAgger rounds)")
    ap.add_argument("--init", default=None, help="continue from this checkpoint (6d)")
    ap.add_argument("--drop-gate", default=None,
                    help="recompute all drop labels with this gate (6d: safe30m); default: as recorded")
    ap.add_argument("--low-weight", type=float, default=1.0,
                    help=f"weight of steps with feet below {LOW_H} m in the move loss (1 = off)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16, help="clips per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--max-clips", type=int, default=0, help="tests only: limit clips per epoch")
    a = ap.parse_args()
    out = os.path.join(RESULTS, "runs", a.name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    demos = Demos(a.data, drop_gate=a.drop_gate)
    if a.max_clips:
        full = demos.clip_starts
        demos.clip_starts = lambda split, r: full(split, r)[:a.max_clips]
    model = Student(a.size).to(a.device)
    if a.init:
        ck0 = torch.load(a.init, map_location=a.device, weights_only=False)
        assert ck0["size"] == a.size, f"--init has size {ck0['size']}, --size is {a.size}"
        model.load_state_dict(ck0["model"])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    amp = (not a.no_amp) and a.device.startswith("cuda")
    cfg = dict(vars(a), n_params=n_params(model), amp=amp, drop_weight=DROP_WEIGHT, low_h=LOW_H,
               drop_labels=int(demos.a["drop"].sum()), drop_labels_recorded=demos.drop_recorded,
               train_episodes=int(len(demos.episodes("train"))), val_episodes=int(len(demos.episodes("val"))),
               parts=[dict(path=os.path.abspath(p), samples=n, episodes=m) for p, n, m in demos.part_sizes])
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"[bc] {a.name}: {cfg['n_params']:,} parameters, device {a.device}, amp {amp}, "
          f"train/val episodes {cfg['train_episodes']}/{cfg['val_episodes']}"
          + (f", start from {a.init}" if a.init else ""), flush=True)
    for p, n, m in demos.part_sizes:
        print(f"[bc]   data {p}: {n:,} samples, {m} episodes", flush=True)
    n_drop = int(demos.a["drop"].sum())
    print(f"[bc]   drop labels: {n_drop:,} ({100 * n_drop / len(demos.a['drop']):.2f}% of steps) in "
          f"{len({int(e) for e in demos.a['ep'][demos.a['drop'] > 0]})} episodes"
          + (f"; recomputed with gate '{a.drop_gate}' (recorded: {demos.drop_recorded:,})" if a.drop_gate else ""),
          flush=True)
    best = float("inf")
    log_path = os.path.join(out, "log.csv")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, demos, "train", rng, a.batch, a.device, amp, opt, a.low_weight)
        va = run_epoch(model, demos, "val", np.random.default_rng(12345), a.batch, a.device, amp,
                       low_weight=a.low_weight)
        val_loss = va["loss_move"] + DROP_COEF * va["loss_drop"]
        row = dict(epoch=ep, sec=round(time.time() - t0, 1),
                   **{f"train_{k}": round(v, 5) for k, v in tr.items()},
                   **{f"val_{k}": round(v, 5) for k, v in va.items()})
        new = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
        ck = dict(model=model.state_dict(), size=a.size, epoch=ep, val_loss=val_loss, config=cfg)
        torch.save(ck, os.path.join(out, "last.pt"))
        tag = ""
        if val_loss < best:
            best = val_loss
            torch.save(ck, os.path.join(out, "best.pt"))
            tag = "  * best"
        print(f"[bc] epoch {ep:2d}/{a.epochs} {row['sec']:5.0f} s | train move {tr['loss_move']:.4f} "
              f"drop {tr['loss_drop']:.4f} | val move {va['loss_move']:.4f} drop {va['loss_drop']:.4f} | "
              f"val MAE vx {va['move_mae_vx']:.2f} vy {va['move_mae_vy']:.2f} vz {va['move_mae_vz']:.2f} m/s "
              f"| low vx {va['low_mae_vx']:.2f} vz {va['low_mae_vz']:.2f} ({100 * va['low_share']:.0f}% of steps) "
          f"| val drop recall {100 * va['drop_recall']:.0f}% false {100 * va['false_drop']:.2f}%{tag}",
              flush=True)
    print(f"[bc] done; best val loss {best:.4f} -> {os.path.abspath(out)}/best.pt", flush=True)


if __name__ == "__main__":
    main()
