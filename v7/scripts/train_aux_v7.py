"""v7 S4h: behaviour cloning WITH A HELPER TASK (auxiliary prediction), 21 Nov 2026.

Decided by Yashwanth 21 Nov 21:39 ("yes"). Why: after DAgger rounds 1 to 3 with the view
teacher (dev 1, 5, 7 /400) the student still does not know well WHERE the pad is and HOW
FAST it moves: it flies too fast over a slow pad (half of the demos have a 6 to 9 m/s pad),
overshoots, and when the pad is behind the camera it turns the wrong way.

What changes (training only):
  A small extra head (256 -> 128 -> 8) reads the student's memory output (the LSTM features
  the move and drop heads read) and predicts the privileged pad state 'priv', which every
  recorded step already has (exactly what the critic will see later):
      priv = [pad position relative to the drone (body frame) / 5 m (3),
              pad velocity relative to the drone (body frame) / 5 m/s (3),
              pad speed / 9 m/s, drone height above the pad top / 5 m]
  loss = move + drop + AUX_W x mean squared error of the 8 predictions
  on the SAME steps as the move loss (warm-up steps and, with --blind-mask, long-blind steps
  excluded). The gradient of the helper loss flows into the memory and the camera network,
  so they must find the pad in the picture and track its motion.
What does NOT change: the network that flies (v6_drop/learn/student.Student), its inputs
(camera picture, own velocity, tilt, last command; never 'priv'), the move and drop losses
(identical to train_bc / train_bc_v7, a test checks it), the data, the settings.
The checkpoint's "model" is a plain Student (eval, DAgger, diagnostic and PPO load it
unchanged); the helper head is stored beside it under "aux" and is ignored in flight.
--init accepts a plain checkpoint (helper head starts fresh) or a helper checkpoint (both).

AUX_W 0.1 (default; a guess): the move loss is ~0.006 at the end of training, the helper
loss starts near ~0.3 and should fall to ~0.01 to 0.05, so 0.1 keeps it in the same range.
The log reports the helper errors in metres and m/s on validation steps.

Run (repo root; charger in; performance mode), e.g. round 3 data, from dagger_p60v_r3:
  nohup python -m v7.scripts.train_aux_v7 --name dagger_p60v_r3aux --init v7/results/runs/dagger_p60v_r3/last.pt \\
        --epochs 8 --blind-mask 3 --data v7/results/p60v/bc_data v7/results/p60v/dagger_data/r1 \\
        v7/results/p60v/dagger_data/r2 v7/results/p60v/dagger_data/r3 > v7/results/p60v/dagger_p60v_r3aux.log 2>&1 &
Watch:  tail -f v7/results/p60v/dagger_p60v_r3aux.log | grep --line-buffered "^\\[aux"
Stop:   pkill -f "v7.scripts.train_aux_v7"
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v6_drop.learn.dataset import batch
from v6_drop.learn.student import Student, n_params
from v6_drop.scripts import train_bc as B6

from . import train_bc_v7 as T7

AUX_W = 0.1
PRIV_DIM = 8
# priv scales (landing_env_v7.priv_scale with v_max 9): back to metres and m/s for the log
PRIV_UNITS = np.array([5, 5, 5, 5, 5, 5, 9, 5], dtype=np.float64)
AUX_NAMES = ("pos x", "pos y", "pos z", "vel x", "vel y", "vel z", "pad speed", "height")


class AuxHead(nn.Module):
    """Memory features (T, B, size) -> predicted priv (T, B, 8). Training only."""

    def __init__(self, size=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(size, 128), nn.ELU(), nn.Linear(128, PRIV_DIM))

    def forward(self, feats):
        return self.net(feats)


def losses_aux(model, aux, b, amp, device, low_weight=1.0):
    """move and drop: exactly v6_drop.scripts.train_bc.losses; plus the helper loss."""
    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
        move, logit, feats, _ = model(b["image"], b["state"].float())
        pred = aux(feats)
    move, logit, pred = move.float(), logit.float(), pred.float()
    m = b["mask"]
    n = m.sum().clamp(min=1.0)
    low = (B6.feet_height(b["priv"].float()) < B6.LOW_H).float() * m
    wm = m + (low_weight - 1.0) * low
    l_move = (((move - b["move"]) ** 2).mean(-1) * wm).sum() / wm.sum().clamp(min=1.0)
    y = b["drop"].float()
    w = torch.where(y > 0.5, torch.full_like(y, B6.DROP_WEIGHT), torch.ones_like(y))
    l_drop = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w * m).sum() / n
    err = pred - b["priv"].float()
    l_aux = ((err ** 2).mean(-1) * m).sum() / n
    stats = dict(move_abs=((move - b["move"]).abs() * m.unsqueeze(-1)).sum((0, 1)).detach(),
                 n=n.detach(), drops=(y * m).sum().detach(),
                 move_abs_low=((move - b["move"]).abs() * low.unsqueeze(-1)).sum((0, 1)).detach(),
                 n_low=low.sum().detach(),
                 drops_hit=(((logit > 0) & (y > 0.5)).float() * m).sum().detach(),
                 nondrop=((1 - y) * m).sum().detach(),
                 false_drop=(((logit > 0) & (y < 0.5)).float() * m).sum().detach(),
                 aux_abs=(err.abs() * m.unsqueeze(-1)).sum((0, 1)).detach())
    return l_move, l_drop, l_aux, stats


def run_epoch(model, aux, demos, split, rng, bs, device, amp, aux_w, opt=None, low_weight=1.0):
    pairs = demos.clip_starts(split, rng)
    tot = dict(move=0.0, drop=0.0, aux=0.0, n=0)
    acc = None
    model.train(opt is not None)
    aux.train(opt is not None)
    params = list(model.parameters()) + list(aux.parameters())
    for i in range(0, len(pairs), bs):
        b = B6.to_t(batch(demos, pairs[i:i + bs]), device)
        with torch.set_grad_enabled(opt is not None):
            l_move, l_drop, l_aux, st = losses_aux(model, aux, b, amp, device, low_weight)
            loss = l_move + B6.DROP_COEF * l_drop + aux_w * l_aux
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
        k = st["n"].item()
        tot["move"] += l_move.item() * k
        tot["drop"] += l_drop.item() * k
        tot["aux"] += l_aux.item() * k
        tot["n"] += k
        acc = st if acc is None else {key: acc[key] + st[key] for key in acc}
    out = dict(loss_move=tot["move"] / tot["n"], loss_drop=tot["drop"] / tot["n"],
               loss_aux=tot["aux"] / tot["n"])
    aux_abs = acc.pop("aux_abs").detach().cpu().numpy() / max(float(acc["n"].item()), 1.0)
    out.update(B6.summarize(acc))
    for name, v in zip(AUX_NAMES, aux_abs * PRIV_UNITS):
        out["aux_mae_" + name.replace(" ", "_")] = float(v)
    return out


def main(argv=None):
    T7.use_v7_numbers()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="aux_v7")
    ap.add_argument("--data", nargs="+", default=[os.path.join(T7.RESULTS, "bc_data")])
    ap.add_argument("--init", default=None, help="plain or helper checkpoint to continue from")
    ap.add_argument("--blind-mask", type=float, default=None,
                    help="no loss (move, drop, helper) after this many s without the pad in view")
    ap.add_argument("--aux-weight", type=float, default=AUX_W)
    ap.add_argument("--low-weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16, help="clips per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--max-clips", type=int, default=0, help="tests only: limit clips per epoch")
    ap.add_argument("--runs", default=os.path.join(T7.RESULTS, "runs"), help="tests: output root")
    a = ap.parse_args(argv)
    out = os.path.join(a.runs, a.name)
    if os.path.exists(os.path.join(out, "last.pt")):
        raise SystemExit(f"{out} already has last.pt: choose another --name")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    demos = T7.DemosV7(a.data, blind_s=a.blind_mask)
    if a.max_clips:
        full = demos.clip_starts
        demos.clip_starts = lambda split, r: full(split, r)[:a.max_clips]
    model = Student(a.size).to(a.device)
    aux = AuxHead(a.size).to(a.device)
    init_aux = False
    if a.init:
        ck0 = torch.load(a.init, map_location=a.device, weights_only=False)
        assert ck0["size"] == a.size, f"--init has size {ck0['size']}, --size is {a.size}"
        model.load_state_dict(ck0["model"])
        if "aux" in ck0:
            aux.load_state_dict(ck0["aux"])
            init_aux = True
    opt = torch.optim.Adam(list(model.parameters()) + list(aux.parameters()), lr=a.lr)
    amp = (not a.no_amp) and a.device.startswith("cuda")
    masked = None if demos.known is None else round(float(1.0 - demos.known.mean()), 4)
    cfg = dict(vars(a), n_params=n_params(model), n_params_aux=n_params(aux), amp=amp,
               drop_weight=B6.DROP_WEIGHT, low_h=B6.LOW_H, init_aux=init_aux, masked_share=masked,
               drop_labels=int(demos.a["drop"].sum()),
               train_episodes=int(len(demos.episodes("train"))), val_episodes=int(len(demos.episodes("val"))),
               parts=[dict(path=os.path.abspath(p), samples=n, episodes=m) for p, n, m in demos.part_sizes])
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"[aux] {a.name}: student {cfg['n_params']:,} + helper head {cfg['n_params_aux']:,} parameters, "
          f"device {a.device}, amp {amp}, helper weight {a.aux_weight:g}, train/val episodes "
          f"{cfg['train_episodes']}/{cfg['val_episodes']}"
          + (f", start from {a.init}" + (" (with its helper head)" if init_aux else " (new helper head)")
             if a.init else ""), flush=True)
    for p, n, m in demos.part_sizes:
        print(f"[aux]   data {p}: {n:,} samples, {m} episodes", flush=True)
    if masked is not None:
        print(f"[aux]   blind mask: no loss on steps with the pad out of view > {a.blind_mask:g} s: "
              f"{100 * masked:.1f}% of all steps", flush=True)
    best = float("inf")
    log_path = os.path.join(out, "log.csv")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, aux, demos, "train", rng, a.batch, a.device, amp, a.aux_weight, opt,
                       a.low_weight)
        va = run_epoch(model, aux, demos, "val", np.random.default_rng(12345), a.batch, a.device,
                       amp, a.aux_weight, low_weight=a.low_weight)
        # best.pt by the SAME validation loss as train_bc (move + drop), so runs compare
        val_loss = va["loss_move"] + B6.DROP_COEF * va["loss_drop"]
        row = dict(epoch=ep, sec=round(time.time() - t0, 1),
                   **{f"train_{k}": round(v, 5) for k, v in tr.items()},
                   **{f"val_{k}": round(v, 5) for k, v in va.items()})
        new = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
        ck = dict(model=model.state_dict(), aux=aux.state_dict(), size=a.size, epoch=ep,
                  val_loss=val_loss, config=cfg)
        torch.save(ck, os.path.join(out, "last.pt"))
        tag = ""
        if val_loss < best:
            best = val_loss
            torch.save(ck, os.path.join(out, "best.pt"))
            tag = "  * best"
        print(f"[aux] epoch {ep:2d}/{a.epochs} {row['sec']:5.0f} s | train move {tr['loss_move']:.4f} "
              f"drop {tr['loss_drop']:.4f} helper {tr['loss_aux']:.4f} | val move {va['loss_move']:.4f} "
              f"drop {va['loss_drop']:.4f} helper {va['loss_aux']:.4f} | "
              f"val MAE vx {va['move_mae_vx']:.2f} vy {va['move_mae_vy']:.2f} vz {va['move_mae_vz']:.2f} m/s "
              f"| low vx {va['low_mae_vx']:.2f} vz {va['low_mae_vz']:.2f} "
              f"| val drop recall {100 * va['drop_recall']:.0f}% false {100 * va['false_drop']:.2f}%{tag}",
              flush=True)
        print(f"[aux]   helper val error: pad pos x {va['aux_mae_pos_x']:.2f} y {va['aux_mae_pos_y']:.2f} "
              f"z {va['aux_mae_pos_z']:.2f} m | rel vel x {va['aux_mae_vel_x']:.2f} y {va['aux_mae_vel_y']:.2f} "
              f"m/s | pad speed {va['aux_mae_pad_speed']:.2f} m/s | height {va['aux_mae_height']:.2f} m",
              flush=True)
    json.dump(dict(blind_mask_s=a.blind_mask, masked_share=masked, aux_weight=a.aux_weight),
              open(os.path.join(out, "v7_config.json"), "w"), indent=1)
    print(f"[aux] done; best val loss {best:.4f} -> {os.path.abspath(out)}/best.pt", flush=True)


if __name__ == "__main__":
    main()
