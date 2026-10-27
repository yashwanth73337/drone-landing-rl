"""Teacher-student distillation with DAgger (SPEC D31). A privileged teacher (variant P, true
relative state, no camera) labels every visited state; a vision student learns to output the
teacher's action from its own observations (image + u), plus the usual estimation loss L_est.

Each iteration:
  collect  n_envs x n_steps decisions. At every episode start, each env draws "teacher flies"
           with probability beta (else the student flies, deterministic mean action, no noise).
  learn    epochs x minibatches over seq_len chunks (same recurrent handling as PPO):
             loss = mean (mu_student - clip(mu_teacher, -1, 1))^2 + lambda_est * L_est
           The teacher label is recomputed from the stored obs (P is a feed-forward MLP).
  beta     = beta0 * max(0, 1 - steps_since_start / beta_steps)   (linear decay; then 0)

Task: the D20 hold (success = safe landing), hold practice (D22), sit-down (D23), friction 0.5
(D24). Difficulty: env i runs at level LEVELS[i % 8] (c = level / 80) unless --c is given.
The reward is not used (reward_fn=None).

Outputs in v5_shin/runs/<name>/ (same checkpoint format as train_ppo, so evaluate.py works):
  config.json, updates.csv, episodes.csv, latest.pt, u<N>.pt

Usage (repo root):
  python -m v5_shin.scripts.distill --name D_s2_ft --teacher P_td_v5_s1 --student-init A_s2_cont5 \
      --total-steps 1000000 --beta-steps 500000 --device cuda
  python -m v5_shin.scripts.distill --name D_s1_scratch --teacher P_td_v5_s1 --student-init scratch \
      --seed 1 --total-steps 4000000 --beta-steps 1000000 --device cuda
"""
import argparse
import csv
import json
import os
import platform as pyplatform
import time

import gymnasium as gym
import numpy as np
import torch

from v5_shin.envs import landing_sim as LS
from v5_shin.envs.curriculum import LEVELS
from v5_shin.envs.dr import DRConfig
from v5_shin.envs.shin_env import VZ_MAX_DEFAULT, ShinLandingEnv
from v5_shin.policies.shin_policy import ShinPolicy

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
STUDENT_KEYS = ("image", "u", "target", "s_rel")       # s_rel only for the teacher label
OUTCOMES = ("success", "crash_ground", "crash_platform", "tilt", "drift", "timeout") + LS.HOLD_OUTCOMES


def env_levels(n_envs, c=None):
    return [float(c)] * n_envs if c is not None else [LEVELS[i % len(LEVELS)] / 80.0 for i in range(n_envs)]


def make_env_fn(seed, c, renderer, hold_practice, hold_vz_down, vz_max):
    def f():
        return ShinLandingEnv(mode="vision", renderer=renderer, egl=(renderer == "egl"), seed=seed, c=c,
                              reward_fn=None, vz_max=vz_max, dr=DRConfig(),
                              hold_steps=LS.HOLD_STEPS, hold_practice=hold_practice,
                              hold_vz_down=hold_vz_down)
    return f


def make_venv(n_envs, seed, renderer, hold_practice, hold_vz_down, vz_max=VZ_MAX_DEFAULT, c=None, sync=False):
    fns = [make_env_fn(seed * 1000 + i, ci, renderer, hold_practice, hold_vz_down, vz_max)
           for i, ci in enumerate(env_levels(n_envs, c))]
    kw = dict(autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    return gym.vector.SyncVectorEnv(fns, **kw) if sync else gym.vector.AsyncVectorEnv(fns, context="spawn", **kw)


def load_policy(run, mode, device):
    pol = ShinPolicy(mode).to(device)
    d = torch.load(os.path.join(RUNS, run, "latest.pt"), map_location=device, weights_only=False)
    pol.load_state_dict(d["policy"])
    return pol


def beta_at(step, beta0, beta_steps):
    if beta_steps <= 0:
        return 0.0
    return float(beta0) * max(0.0, 1.0 - step / beta_steps)


class Distiller:
    def __init__(self, venv, student, teacher, device="cpu", n_steps=256, seq_len=32, epochs=4,
                 minibatches=4, micro_chunks=None, lr=1e-4, lambda_est=1.0, max_grad_norm=1.0,
                 beta0=1.0, beta_steps=500_000, seed=0):
        assert n_steps % seq_len == 0
        self.venv, self.student, self.teacher = venv, student, teacher
        self.dev = torch.device(device)
        self.B = venv.num_envs
        self.n_steps, self.seq_len, self.epochs, self.minibatches = n_steps, seq_len, epochs, minibatches
        self.micro_chunks, self.lambda_est, self.max_grad_norm = micro_chunks, lambda_est, max_grad_norm
        self.beta0, self.beta_steps = beta0, beta_steps
        self.opt = torch.optim.Adam(student.parameters(), lr=lr, eps=1e-5)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.rng = np.random.default_rng(seed)
        self.update = 0
        self.steps = 0                     # decisions collected by this distillation run
        self.global_step = 0               # carried over from the student checkpoint, if any

    # ------------------------------------------------------------------ helpers
    def teacher_mu(self, obs):
        """Teacher label for (T, B, ...) or (B, ...) obs; clipped to the env's action box."""
        with torch.no_grad():
            seq = obs if obs["u"].dim() == 3 else {k: v[None] for k, v in obs.items()}
            mu, _, _ = self.teacher.actor_seq({"u": seq["u"], "s_rel": seq["s_rel"]}, None, None)
            mu = mu.clamp(-1.0, 1.0)
        return mu if obs["u"].dim() == 3 else mu[0]

    def _tensors(self, obs):
        return {k: torch.as_tensor(np.asarray(obs[k]), device=self.dev) for k in STUDENT_KEYS}

    def beta(self):
        return beta_at(self.steps, self.beta0, self.beta_steps)

    def reset(self, seeds):
        obs, _ = self.venv.reset(seed=list(seeds))
        self.obs = obs
        self.state = self.student.initial_state(self.B, self.dev)
        self.starts = torch.ones(self.B, device=self.dev)
        self.flies_teacher = self.rng.random(self.B) < self.beta()

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def collect(self):
        T, L, B = self.n_steps, self.seq_len, self.B
        buf = {k: [] for k in STUDENT_KEYS + ("starts", "teacher_flew")}
        h0, c0, episodes = [], [], []
        for t in range(T):
            if t % L == 0:
                h0.append(self.state[0].clone())
                c0.append(self.state[1].clone())
            tob = self._tensors(self.obs)
            mu_s, _, self.state = self.student.actor_seq({k: v[None] for k, v in tob.items()},
                                                         self.state, self.starts[None])
            mu_s = mu_s[0]
            mu_t = self.teacher_mu(tob)
            use_t = torch.as_tensor(self.flies_teacher, device=self.dev)[:, None]
            a = torch.where(use_t, mu_t, mu_s).clamp(-1.0, 1.0)
            for k in STUDENT_KEYS:
                buf[k].append(tob[k])
            buf["starts"].append(self.starts.clone())
            buf["teacher_flew"].append(use_t[:, 0].float())
            obs, _, te, tr, info = self.venv.step(a.cpu().numpy())
            done = te | tr
            if done.any():
                fi = info["final_info"]
                for i in np.nonzero(done)[0]:
                    e = {k: (fi[k][i] if k in fi else None) for k in
                         ("outcome", "t", "c_episode", "rel_vel", "touchdown", "practice")}
                    e["teacher_flew"] = bool(self.flies_teacher[i])
                    episodes.append(e)
                    self.flies_teacher[i] = self.rng.random() < self.beta()   # next episode
            self.obs = obs
            self.starts = torch.as_tensor(done, dtype=torch.float32, device=self.dev)
            self.steps += B
            self.global_step += B
        out = {k: torch.stack(v) for k, v in buf.items()}
        out["h0"], out["c0"] = torch.stack(h0), torch.stack(c0)          # (T/L, 1, B, H)
        return out, episodes

    # ------------------------------------------------------------------ update
    def _gather(self, buf, chunks):
        L = self.seq_len
        ks = torch.tensor([k for k, _ in chunks])
        bs = torch.tensor([b for _, b in chunks])
        tidx = ks[None, :] * L + torch.arange(L)[:, None]                  # (L, M)
        g = lambda x: x[tidx, bs[None, :]]
        obs = {k: g(buf[k]) for k in STUDENT_KEYS}
        starts = g(buf["starts"])
        state = (buf["h0"][ks, 0, bs][None].contiguous(), buf["c0"][ks, 0, bs][None].contiguous())
        return obs, starts, state

    def _loss(self, obs, starts, state):
        mu, s_est, _ = self.student.actor_seq(obs, state, starts)
        bc = ((mu - self.teacher_mu(obs)) ** 2).mean()
        est = ((s_est - obs["target"]) ** 2).mean() if s_est is not None else torch.zeros((), device=self.dev)
        return bc + self.lambda_est * est, bc.item(), est.item()

    def learn(self, buf):
        n_chunks_t = self.n_steps // self.seq_len
        all_chunks = [(k, b) for k in range(n_chunks_t) for b in range(self.B)]
        mb = max(1, len(all_chunks) // self.minibatches)
        logs = []
        for _ in range(self.epochs):
            perm = np.random.permutation(len(all_chunks))
            for i in range(0, len(all_chunks), mb):
                chunks = [all_chunks[j] for j in perm[i:i + mb]]
                micro = self.micro_chunks or len(chunks)
                self.opt.zero_grad()
                bc_sum = est_sum = 0.0
                for j in range(0, len(chunks), micro):
                    part = chunks[j:j + micro]
                    obs, starts, state = self._gather(buf, part)
                    loss, bc, est = self._loss(obs, starts, state)
                    (loss * len(part) / len(chunks)).backward()
                    bc_sum += bc * len(part)
                    est_sum += est * len(part)
                gn = torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                self.opt.step()
                logs.append((bc_sum / len(chunks), est_sum / len(chunks), float(gn)))
        self.update += 1
        a = np.array(logs)
        return dict(bc=float(a[:, 0].mean()), est=float(a[:, 1].mean()), grad_norm=float(a[:, 2].mean()),
                    bc_last=float(a[-1, 0]))

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self, extra=None):
        return dict(policy=self.student.state_dict(), opt=self.opt.state_dict(), update=self.update,
                    global_step=self.global_step, distill_steps=self.steps, extra=extra or {})

    def load_state_dict(self, d):
        self.student.load_state_dict(d["policy"])
        self.opt.load_state_dict(d["opt"])
        self.update, self.global_step = d["update"], d["global_step"]
        self.steps = d.get("distill_steps", 0)


class CSVLog:
    def __init__(self, path):
        self.path, self.f, self.w = path, None, None

    def write(self, row):
        if self.w is None:
            new = not os.path.exists(self.path)
            self.f = open(self.path, "a", newline="")
            self.w = csv.DictWriter(self.f, fieldnames=list(row))
            if new:
                self.w.writeheader()
        self.w.writerow({k: row.get(k) for k in self.w.fieldnames})
        self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--teacher", required=True, help="privileged (P) run folder with latest.pt")
    ap.add_argument("--student-init", required=True, help="vision run folder with latest.pt, or 'scratch'")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--total-steps", type=int, required=True, help="decisions collected in this run")
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--beta-steps", type=int, default=500_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--micro-chunks", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-est", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=None, help="fixed difficulty (default: envs spread over all levels)")
    ap.add_argument("--hold-practice", type=float, default=0.25)
    ap.add_argument("--hold-vz-down", type=float, default=5.0)
    ap.add_argument("--vz-max", type=float, default=VZ_MAX_DEFAULT)
    ap.add_argument("--renderer", choices=["egl", "tiny"], default="tiny")
    ap.add_argument("--ckpt-every", type=int, default=25, help="updates")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/latest.pt")
    ap.add_argument("--sync", action="store_true", help="SyncVectorEnv (tests)")
    a = ap.parse_args()

    run = os.path.join(RUNS, a.name)
    os.makedirs(run, exist_ok=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = a.device
    micro = a.micro_chunks if a.micro_chunks else (8 if dev != "cpu" else None)
    teacher = load_policy(a.teacher, "privileged", dev)
    if a.student_init == "scratch":
        student = ShinPolicy("vision").to(dev)
        start_step = 0
    else:
        student = load_policy(a.student_init, "vision", dev)
        start_step = int(torch.load(os.path.join(RUNS, a.student_init, "latest.pt"), map_location="cpu",
                                    weights_only=False).get("global_step", 0))
    venv = make_venv(a.n_envs, a.seed, a.renderer, a.hold_practice, a.hold_vz_down, a.vz_max, a.c, sync=a.sync)
    dz = Distiller(venv, student, teacher, device=dev, n_steps=a.n_steps, seq_len=a.seq_len, epochs=a.epochs,
                   minibatches=a.minibatches, micro_chunks=micro, lr=a.lr, lambda_est=a.lambda_est,
                   beta0=a.beta0, beta_steps=a.beta_steps, seed=a.seed)
    dz.global_step = start_step
    if a.resume:
        dz.load_state_dict(torch.load(os.path.join(run, "latest.pt"), map_location=dev, weights_only=False))
    per_update = a.n_envs * a.n_steps
    total_updates = int(np.ceil(a.total_steps / per_update))
    if not os.path.exists(os.path.join(run, "config.json")):
        with open(os.path.join(run, "config.json"), "w") as f:
            json.dump(dict(vars(a), mode="vision", method="distill_dagger", vz_penalty="literal",
                           motor="asym", hold_steps=LS.HOLD_STEPS, micro_chunks_used=micro,
                           levels=env_levels(a.n_envs, a.c), student_start_step=start_step,
                           host=pyplatform.node(), torch=torch.__version__, gymnasium=gym.__version__,
                           gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)),
                      f, indent=2)
    upd_log, ep_log = CSVLog(os.path.join(run, "updates.csv")), CSVLog(os.path.join(run, "episodes.csv"))
    dz.reset(seeds=[a.seed * 100000 + dz.update * 1000 + i for i in range(a.n_envs)])
    t0 = time.time()
    try:
        while dz.update < total_updates:
            beta = dz.beta()
            t_roll = time.time()
            buf, eps = dz.collect()
            t_roll = time.time() - t_roll
            t_learn = time.time()
            stats = dz.learn(buf)
            t_learn = time.time() - t_learn
            for e in eps:
                rv = e.get("rel_vel")
                ep_log.write(dict(update=dz.update, outcome=e["outcome"], steps=e["t"], c_episode=e["c_episode"],
                                  touchdown=bool(e.get("touchdown")), practice=bool(e.get("practice")),
                                  teacher_flew=e["teacher_flew"],
                                  impact_vz=(float(rv[2]) if rv is not None else None)))
            stu = [e for e in eps if not e["teacher_flew"] and not e.get("practice")]
            outc = [e["outcome"] for e in stu]
            td = [e for e in stu if e.get("touchdown")]
            row = dict(update=dz.update, distill_steps=dz.steps, global_step=dz.global_step,
                       wall_s=round(time.time() - t0, 1), sps=round(per_update / max(t_roll + t_learn, 1e-9), 1),
                       beta=round(beta, 4), episodes=len(eps), student_episodes=len(stu),
                       **{f"stu_n_{k}": outc.count(k) for k in OUTCOMES},
                       stu_n_touchdown=len(td),
                       stu_td_impact_vz_mean=(float(np.mean([e["rel_vel"][2] for e in td])) if td else None),
                       lr=dz.opt.param_groups[0]["lr"], **stats)
            upd_log.write(row)
            n = max(1, len(stu))
            print(f"d{dz.update:5d} step {dz.steps:>9d} beta {beta:.2f} | student eps {len(stu):3d} "
                  f"safe {outc.count('success'):3d} td {len(td):3d} hard {outc.count('hard_landing'):3d} "
                  f"crash {outc.count('crash_ground') + outc.count('crash_platform'):3d} "
                  f"({100 * outc.count('success') / n:4.1f}% safe) | bc {stats['bc']:.4f} est {stats['est']:.3f} "
                  f"| {row['sps']:.0f} sps", flush=True)
            if dz.update % a.ckpt_every == 0 or dz.update == total_updates:
                torch.save(dz.state_dict(), os.path.join(run, f"u{dz.update:05d}.pt"))
            torch.save(dz.state_dict(), os.path.join(run, "latest.pt"))
    finally:
        venv.close()


if __name__ == "__main__":
    main()
