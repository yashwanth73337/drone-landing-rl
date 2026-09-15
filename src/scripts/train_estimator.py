"""
train_estimator.py
==================

Offline supervised training of the V3b temporal estimator, plus the gate that
decides whether an RL run is worth starting at all.

THE GATE (step 9 of the agreed V3b plan)
------------------------------------------------------------------------------
The LSTM must beat the constant-velocity predictor on held-out episodes,
specifically during LONG BLIND STRETCHES and in RELATIVE-VELOCITY estimation.
If it does not, stop here -- do not spend ~6 hours training TD3.

Errors are reported in PHYSICAL units (metres, m/s), binned by blind age, with
95th-percentile and maximum values alongside the means. V3a's failure regime
contained blind stretches up to ~181 control steps, so mean error over all
frames would hide exactly the regime that matters.


WHAT ADVANTAGE IS EVEN AVAILABLE HERE
------------------------------------------------------------------------------
Worth stating before looking at any numbers: this environment samples ONE
CONSTANT platform velocity per episode
(LanderAIAviary._samplePlatformMotion). The hand-coded constant-velocity
predictor therefore has the CORRECT MODEL CLASS for this task. It is not a
strawman.

The LSTM's plausible advantages are consequently narrow:

    denoising the noisy single-step finite-difference velocity estimate
    integrating several visual measurements instead of trusting the latest one
    maintaining a better latent estimate through long dropout
    exploiting proprioceptive ego-motion temporally

It must NOT be claimed that any result here demonstrates handling of uncertain
or non-constant target dynamics. That claim would require changing the
platform dynamics and testing it explicitly, which is a separate experiment.


SEQUENCE HANDLING
------------------------------------------------------------------------------
Full episodes, padded to the batch maximum, with a loss mask. Episodes are
~300 control steps, which fits comfortably, so no truncated BPTT is used and
no hidden state is carried across chunks.

This matters: hidden state is initialised to zero ONLY at a real episode
reset, exactly as it will be at deployment. Resetting it every short training
window would train an estimator that never learns to hold information for
longer than the window -- a different model from the one deployed.


SPLIT
------------------------------------------------------------------------------
BY EPISODE, stratified by curriculum level so every split covers the same
range of tasks. No timestep and no sequence appears in more than one split.
Standardisation statistics are computed from the TRAIN split only.

USAGE
    python train_estimator.py --data estimator_data/v3b_dataset.npz \\
        --out models/temporal_estimator_v3b.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models.temporal_estimator import (                     # noqa: E402
    TemporalEstimator, constant_velocity_baseline,
    BLIND_AGE_BINS, INPUT_DIM, OUTPUT_DIM)

DT = 1.0 / 30.0


# ===========================================================================
# DATA
# ===========================================================================

def load_episodes(path):
    """Load one or more datasets. `path` may be a comma-separated list.

    Merging is by episode, so the targeted hard-regime supplement can be
    combined with the base dataset without either being resampled or
    reweighted. Training on the supplement alone would trade one
    distribution gap for another.
    """
    if ',' in path:
        eps = []
        for p in path.split(','):
            p = p.strip()
            sub = load_episodes(p)
            print(f'  loaded {len(sub):4d} episodes from {p}')
            eps += sub
        return eps

    z = np.load(path, allow_pickle=True)
    starts, lengths = z['ep_start'], z['ep_length']
    eps = []
    for i in range(len(starts)):
        a, b = int(starts[i]), int(starts[i]) + int(lengths[i])
        eps.append({
            'd_meas': z['d_meas'][a:b].astype(np.float64),
            'valid': z['valid'][a:b].astype(np.float64),
            'v': z['v'][a:b].astype(np.float64),
            'rpy': z['rpy'][a:b].astype(np.float64),
            'omega': z['omega'][a:b].astype(np.float64),
            'gt_d': z['gt_d'][a:b].astype(np.float64),
            'gt_dv': z['gt_dv'][a:b].astype(np.float64),
            'level': int(z['ep_level'][i]),
            'source': str(z['ep_source'][i]),
        })
    return eps


def build_io(ep):
    """13-D input, 6-D target. d_meas zeroed wherever valid == 0."""
    valid = ep['valid'].reshape(-1, 1)
    d_meas = ep['d_meas'] * valid          # enforce zeros during blindness
    x = np.concatenate([d_meas, valid, ep['v'], ep['rpy'], ep['omega']],
                       axis=1)
    y = np.concatenate([ep['gt_d'], ep['gt_dv']], axis=1)
    assert x.shape[1] == INPUT_DIM and y.shape[1] == OUTPUT_DIM
    return x, y


def split_by_episode(eps, seed=0, frac=(0.70, 0.15, 0.15)):
    """Stratified by level, split at the EPISODE level only."""
    rng = np.random.default_rng(seed)
    by_level = {}
    for i, e in enumerate(eps):
        by_level.setdefault(e['level'], []).append(i)

    tr, va, te = [], [], []
    for lv, idx in sorted(by_level.items()):
        idx = np.array(idx)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(frac[0] * n))
        n_va = int(round(frac[1] * n))
        tr += idx[:n_tr].tolist()
        va += idx[n_tr:n_tr + n_va].tolist()
        te += idx[n_tr + n_va:].tolist()
    return tr, va, te


def pad_batch(items, device):
    T = max(len(x) for x, _ in items)
    B = len(items)
    X = np.zeros((B, T, INPUT_DIM), dtype=np.float32)
    Y = np.zeros((B, T, OUTPUT_DIM), dtype=np.float32)
    M = np.zeros((B, T), dtype=np.float32)
    for i, (x, y) in enumerate(items):
        t = len(x)
        X[i, :t], Y[i, :t], M[i, :t] = x, y, 1.0
    return (torch.from_numpy(X).to(device),
            torch.from_numpy(Y).to(device),
            torch.from_numpy(M).to(device))


# ===========================================================================
# EVALUATION
# ===========================================================================

def blind_age_of(valid):
    age, out = 0, np.zeros(len(valid), dtype=int)
    for t, v in enumerate(valid):
        age = 0 if v else age + 1
        out[t] = age
    return out


def evaluate(model, eps, idx, device):
    """LSTM vs constant-velocity on the SAME held-out frames."""
    rows = {name: {'lstm_p': [], 'lstm_v': [], 'cv_p': [], 'cv_v': []}
            for name, _, _ in BLIND_AGE_BINS}

    model.eval()
    with torch.no_grad():
        for i in idx:
            ep = eps[i]
            x, y = build_io(ep)

            pred, _ = model.predict_physical(
                torch.from_numpy(x).float().unsqueeze(0).to(device))
            pred = pred.squeeze(0).cpu().numpy()

            cv_d, cv_dv, _ = constant_velocity_baseline(
                ep['d_meas'] * ep['valid'].reshape(-1, 1),
                ep['valid'], ep['v'], DT)

            age = blind_age_of(ep['valid'])
            lp = np.linalg.norm(pred[:, :3] - y[:, :3], axis=1)
            lv = np.linalg.norm(pred[:, 3:] - y[:, 3:], axis=1)
            cp = np.linalg.norm(cv_d - y[:, :3], axis=1)
            cv = np.linalg.norm(cv_dv - y[:, 3:], axis=1)

            for name, lo, hi in BLIND_AGE_BINS:
                m = (age >= lo) & (age <= hi)
                if not m.any():
                    continue
                rows[name]['lstm_p'] += lp[m].tolist()
                rows[name]['lstm_v'] += lv[m].tolist()
                rows[name]['cv_p'] += cp[m].tolist()
                rows[name]['cv_v'] += cv[m].tolist()
    return rows


def report(rows):
    def stat(a):
        if not a:
            return None
        a = np.asarray(a)
        return {'n': int(a.size), 'mean': float(a.mean()),
                'rmse': float(np.sqrt((a ** 2).mean())),
                'p95': float(np.percentile(a, 95)),
                'max': float(a.max())}

    print()
    print('=' * 96)
    print('  ESTIMATOR VALIDATION -- held-out episodes, PHYSICAL units')
    print('=' * 96)
    print(f'  {"bin":<14}{"n":>8}'
          f'{"CV pos":>11}{"LSTM pos":>11}{"CV p95":>10}{"LSTM p95":>10}'
          f'{"CV vel":>11}{"LSTM vel":>11}')
    print('  ' + '-' * 92)

    out = {}
    for name, _, _ in BLIND_AGE_BINS:
        r = rows[name]
        lp, lv = stat(r['lstm_p']), stat(r['lstm_v'])
        cp, cvv = stat(r['cv_p']), stat(r['cv_v'])
        if lp is None:
            continue
        out[name] = {'lstm_pos': lp, 'lstm_vel': lv,
                     'cv_pos': cp, 'cv_vel': cvv}
        print(f'  {name:<14}{lp["n"]:>8}'
              f'{cp["mean"]:>11.4f}{lp["mean"]:>11.4f}'
              f'{cp["p95"]:>10.4f}{lp["p95"]:>10.4f}'
              f'{cvv["mean"]:>11.4f}{lv["mean"]:>11.4f}')
    print('  ' + '-' * 92)
    print('  position in metres, velocity in m/s. Lower is better.')
    print()
    print('  GATE: the LSTM must beat constant velocity in the LONG blind')
    print('  bins (>30 steps) and on velocity. If it does not, STOP -- do')
    print('  not start the TD3 run.')
    print()
    print('  Reminder: this environment samples ONE CONSTANT platform')
    print('  velocity per episode, so the constant-velocity predictor has')
    print('  the correct model class. Any LSTM advantage here is denoising,')
    print('  multi-measurement integration and dropout bridging -- NOT')
    print('  evidence of handling non-constant target dynamics.')
    print('=' * 96)
    return out


# ===========================================================================
# TRAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='models/temporal_estimator_v3b.pt')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eps = load_episodes(args.data)
    tr, va, te = split_by_episode(eps, seed=args.seed)

    print('=' * 74)
    print('  V3b ESTIMATOR TRAINING')
    print('=' * 74)
    print(f'  device   : {device}')
    print(f'  episodes : {len(eps)}  (train {len(tr)} / val {len(va)} '
          f'/ test {len(te)})')
    print('  split is BY EPISODE, stratified by curriculum level.')
    lv = {}
    for i in tr:
        lv[eps[i]['level']] = lv.get(eps[i]['level'], 0) + 1
    print(f'  train episodes per level: {dict(sorted(lv.items()))}')
    print()

    io = [build_io(e) for e in eps]

    # ---- statistics: TRAIN SPLIT ONLY ----
    Xtr = np.concatenate([io[i][0] for i in tr], axis=0)
    Ytr = np.concatenate([io[i][1] for i in tr], axis=0)
    in_mean, in_std = Xtr.mean(0), Xtr.std(0)
    out_mean, out_std = Ytr.mean(0), Ytr.std(0)
    in_std[in_std < 1e-6] = 1.0
    out_std[out_std < 1e-6] = 1.0

    print('  target statistics (train split, physical units):')
    for k, nm in enumerate(['d_x', 'd_y', 'd_z', 'dv_x', 'dv_y', 'dv_z']):
        print(f'    {nm:>5}: mean {out_mean[k]:+.4f}  std {out_std[k]:.4f}')
    print()

    model = TemporalEstimator(args.hidden, args.layers).to(device)
    model.set_stats(in_mean, in_std, out_mean, out_std)
    print(f'  architecture: LSTM({INPUT_DIM} -> {args.hidden}, '
          f'{args.layers} layer) -> MLP({args.hidden} -> 128 -> 128 -> '
          f'{OUTPUT_DIM})')
    print(f'  parameters  : {sum(p.numel() for p in model.parameters()):,}')
    print()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out_mean_t = torch.tensor(out_mean, dtype=torch.float32, device=device)
    out_std_t = torch.tensor(out_std, dtype=torch.float32, device=device)

    best_val, best_state, bad = float('inf'), None, 0
    rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr))
        tot, nb = 0.0, 0
        for s in range(0, len(order), args.batch):
            batch = [io[tr[j]] for j in order[s:s + args.batch]]
            X, Y, M = pad_batch(batch, device)
            Z = (Y - out_mean_t) / out_std_t          # standardised target

            pred, _ = model(X)                        # standardised output
            loss = (((pred - Z) ** 2).mean(-1) * M).sum() / M.sum()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1

        model.eval()
        with torch.no_grad():
            vb = [io[i] for i in va]
            X, Y, M = pad_batch(vb, device)
            Z = (Y - out_mean_t) / out_std_t
            pred, _ = model(X)
            vloss = float((((pred - Z) ** 2).mean(-1) * M).sum() / M.sum())

        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1

        if epoch % 10 == 0 or bad == 0:
            print(f'  epoch {epoch:3d}  train {tot / max(nb, 1):.5f}  '
                  f'val {vloss:.5f}' + ('  *' if bad == 0 else ''))

        if bad >= args.patience:
            print(f'  early stop at epoch {epoch} '
                  f'(no val improvement for {args.patience})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'state_dict': model.state_dict(),
                'hidden_size': args.hidden, 'num_layers': args.layers},
               args.out)
    print(f'\n  saved {args.out}  (best val {best_val:.5f})')

    res = report(evaluate(model, eps, te, device))
    with open(args.out.replace('.pt', '_validation.json'), 'w') as f:
        json.dump({'test_episodes': len(te), 'bins': res}, f, indent=2)
    print(f'  wrote {args.out.replace(".pt", "_validation.json")}')


if __name__ == '__main__':
    main()
