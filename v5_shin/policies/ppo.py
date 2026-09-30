"""Recurrent PPO with the auxiliary estimation loss (V5 Block 12; Shin et al. Sec. III-B/D;
SPEC §11). Own implementation (D8), cleanRL-style.

Rollout (n_envs x n_steps), vector env with SAME_STEP autoreset:
  - LSTM state stored at the start of every seq_len chunk; episode starts reset it.
  - Timeouts (truncated) are bootstrapped: next_value = V(final_obs.critic).
    Terminal steps: next_value = 0. The GAE chain is cut at every done.
Update: epochs x minibatches over chunks of seq_len steps. Each minibatch can be split
into micro-batches with gradient accumulation (one optimizer step per minibatch; identical
to the unsplit update, which is tested). This is needed on a 4 GB GPU.
  loss = L_clip + vf_coef * mean((V - R)^2) - ent_coef * H + lambda_est * L_est
  L_est = mean over steps of (1/6) sum_i (s_rel_i - s~_i)^2           [paper Eq. 1]
r_active (Block 13) plugs in via r_active_fn(buffer) -> (T, B) added to the rewards before GAE.
"""
import numpy as np
import torch

from .shin_policy import ShinPolicy

BLIND_BINS = ((0, 0), (1, 15), (16, 60), (61, 120), (121, 10 ** 9))   # blind-age bins (steps)


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """All (T, B) tensors. next_values[t] = V(true s_{t+1}): values[t+1] if not done,
    V(final obs) if truncated, anything if terminated (masked). Returns (adv, returns)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        nonterm = 1.0 - terminated[t]
        delta = rewards[t] + gamma * next_values[t] * nonterm - values[t]
        cont = 1.0 - torch.clamp(terminated[t] + truncated[t], max=1.0)
        last = delta + gamma * lam * cont * last
        adv[t] = last
    return adv, adv + values


def obs_to_tensor(obs, device, keys):
    return {k: torch.as_tensor(np.asarray(obs[k]), device=device) for k in keys}


class PPOConfig:
    def __init__(self, **kw):
        self.mode = "vision"
        self.n_steps = 256
        self.seq_len = 32
        self.epochs = 5
        self.minibatches = 4
        self.micro_chunks = None          # chunks per micro-batch (None = no accumulation)
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.lr = 3e-4
        self.lr_decay = True
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.max_grad_norm = 1.0
        self.lambda_est = 1.0             # D10; 0 = the "w/o state estimation" ablation
        self.norm_adv = True
        self.log_std_init = -0.5
        for k, v in kw.items():
            assert hasattr(self, k), k
            setattr(self, k, v)
        assert self.n_steps % self.seq_len == 0


class PPOTrainer:
    OBS_KEYS = ("image", "u", "critic", "target", "s_rel")

    def __init__(self, venv, cfg, device="cpu", total_updates=1, r_active_fn=None):
        self.venv, self.cfg, self.device = venv, cfg, torch.device(device)
        self.B = venv.num_envs
        self.policy = ShinPolicy(cfg.mode, log_std_init=cfg.log_std_init).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5)
        self.total_updates = max(1, total_updates)
        self.update = 0
        self.global_step = 0
        self.r_active_fn = r_active_fn
        self.keys = self.OBS_KEYS if cfg.mode == "vision" else ("u", "critic", "target", "s_rel")
        self.obs = None
        self.state = None
        self.starts = None
        self.blind_age = np.zeros(self.B, dtype=np.int64)

    # ------------------------------------------------------------------ rollout
    def reset(self, seeds):
        obs, info = self.venv.reset(seed=list(seeds))
        self.obs = obs
        self.state = self.policy.initial_state(self.B, self.device)
        self.starts = torch.ones(self.B, device=self.device)
        self.blind_age[:] = np.where(np.asarray(info["pad_centre_in_view"]), 0, 1)
        return info

    @torch.no_grad()
    def collect(self):
        cfg, B, T, L, dev = self.cfg, self.B, self.cfg.n_steps, self.cfg.seq_len, self.device
        buf = {k: [] for k in self.keys}
        for k in ("actions", "logp", "values", "rewards", "terminated", "truncated",
                  "starts", "next_values", "s_est", "blind_age"):
            buf[k] = []
        h0, c0 = [], []
        episodes = []
        for t in range(T):
            if t % L == 0 and self.state is not None:
                h0.append(self.state[0].clone())
                c0.append(self.state[1].clone())
            tob = obs_to_tensor(self.obs, dev, self.keys)
            a, logp, v, s_est, self.state = self.policy.act(tob, self.state, self.starts)
            for k in self.keys:
                buf[k].append(tob[k])
            buf["actions"].append(a)
            buf["logp"].append(logp)
            buf["values"].append(v)
            buf["starts"].append(self.starts.clone())
            buf["s_est"].append(s_est if s_est is not None else torch.zeros(B, 6, device=dev))
            buf["blind_age"].append(torch.as_tensor(self.blind_age.copy()))

            obs, r, te, tr, info = self.venv.step(a.cpu().numpy())
            done = te | tr
            nv = torch.zeros(B, device=dev)
            if tr.any():
                fo = info["final_obs"]
                idx = np.nonzero(tr)[0]
                crit = torch.as_tensor(np.stack([fo[i]["critic"] for i in idx]), device=dev)
                nv[torch.as_tensor(idx, device=dev)] = self.policy.value(crit)
            if done.any():
                fi = info["final_info"]
                for i in np.nonzero(done)[0]:
                    episodes.append({k: (fi[k][i] if k in fi else None) for k in
                                     ("outcome", "t", "c_episode", "com_over_pad", "rel_vel",
                                      "tilt_deg")})
            buf["rewards"].append(torch.as_tensor(r, dtype=torch.float32, device=dev))
            buf["terminated"].append(torch.as_tensor(te, dtype=torch.float32, device=dev))
            buf["truncated"].append(torch.as_tensor(tr, dtype=torch.float32, device=dev))
            buf["next_values"].append(nv)
            # blind age of the NEW obs (reset obs for finished envs)
            vis = np.asarray(info["pad_centre_in_view"], dtype=bool)
            self.blind_age = np.where(vis, 0, np.where(done, 1, self.blind_age + 1))
            self.obs = obs
            self.starts = torch.as_tensor(done, dtype=torch.float32, device=dev)
            self.global_step += B

        out = {k: torch.stack(v) for k, v in buf.items()}
        # next_values for non-done steps = V(s_{t+1}) = values[t+1]; last step: V(current obs)
        last_v = self.policy.value(obs_to_tensor(self.obs, dev, ("critic",))["critic"])
        vals_next = torch.cat([out["values"][1:], last_v[None]], 0)
        done = torch.clamp(out["terminated"] + out["truncated"], max=1.0)
        out["next_values"] = torch.where(out["truncated"] > 0, out["next_values"],
                                         vals_next * (1 - done))
        if self.state is not None:
            out["h0"] = torch.stack(h0)       # (T/L, 1, B, H)
            out["c0"] = torch.stack(c0)
        out["r_active"] = (self.r_active_fn(out) if self.r_active_fn is not None
                           else torch.zeros_like(out["rewards"]))
        adv, ret = compute_gae(out["rewards"] + out["r_active"], out["values"], out["next_values"],
                               out["terminated"], out["truncated"], cfg.gamma, cfg.lam)
        out["advantages"], out["returns"] = adv, ret
        return out, episodes

    # ------------------------------------------------------------------ update
    def _gather(self, buf, chunks):
        """chunks: list of (k, b). Returns obs dict (L, M, ...), tensors (L, M), state."""
        L = self.cfg.seq_len
        ks = torch.tensor([k for k, _ in chunks])
        bs = torch.tensor([b for _, b in chunks])
        tidx = (ks[None, :] * L + torch.arange(L)[:, None])            # (L, M)
        g = lambda x: x[tidx, bs[None, :]]
        obs = {k: g(buf[k]) for k in self.keys}
        rest = {k: g(buf[k]) for k in ("actions", "logp", "values", "starts", "advantages",
                                       "returns")}
        state = None
        if "h0" in buf:
            state = (buf["h0"][ks, 0, bs][None].contiguous(), buf["c0"][ks, 0, bs][None].contiguous())
        return obs, rest, state

    def _loss(self, obs, rest, state, adv):
        cfg = self.cfg
        logp, ent, v, s_est = self.policy.evaluate(obs, rest["actions"], state, rest["starts"])
        ratio = (logp - rest["logp"]).exp()
        pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        v_loss = ((v - rest["returns"]) ** 2).mean()
        est = (((s_est - obs["target"]) ** 2).mean() if s_est is not None
               else torch.zeros((), device=self.device))
        total = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent.mean() + cfg.lambda_est * est
        with torch.no_grad():
            kl = ((ratio - 1) - (logp - rest["logp"])).mean()
            clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean()
        return total, dict(pg=pg.item(), v=v_loss.item(), ent=ent.mean().item(), est=est.item(),
                           kl=kl.item(), clipfrac=clipfrac.item())

    def update_step(self, buf, chunks):
        """One optimizer step over a minibatch of chunks (with optional accumulation)."""
        cfg = self.cfg
        adv_all = self._gather(buf, chunks)[1]["advantages"]
        mu, sd = adv_all.mean(), adv_all.std() + 1e-8
        micro = cfg.micro_chunks or len(chunks)
        parts = [chunks[i:i + micro] for i in range(0, len(chunks), micro)]
        self.opt.zero_grad()
        stats = []
        for part in parts:
            obs, rest, state = self._gather(buf, part)
            adv = (rest["advantages"] - mu) / sd if cfg.norm_adv else rest["advantages"]
            loss, st = self._loss(obs, rest, state, adv)
            (loss * len(part) / len(chunks)).backward()
            stats.append((len(part), st))
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.opt.step()
        n = sum(w for w, _ in stats)
        agg = {k: sum(w * s[k] for w, s in stats) / n for k in stats[0][1]}
        agg["grad_norm"] = float(gn)
        return agg

    def learn(self, buf):
        cfg = self.cfg
        if cfg.lr_decay:
            for g in self.opt.param_groups:
                g["lr"] = cfg.lr * max(0.0, 1.0 - self.update / self.total_updates)
        n_chunks_t = cfg.n_steps // cfg.seq_len
        all_chunks = [(k, b) for k in range(n_chunks_t) for b in range(self.B)]
        mb = max(1, len(all_chunks) // cfg.minibatches)
        logs = []
        for _ in range(cfg.epochs):
            perm = np.random.permutation(len(all_chunks))
            for i in range(0, len(all_chunks), mb):
                logs.append(self.update_step(buf, [all_chunks[j] for j in perm[i:i + mb]]))
        self.update += 1
        out = {k: float(np.mean([l[k] for l in logs])) for k in logs[0]}
        out["kl_first"] = logs[0]["kl"]
        y, yhat = buf["returns"].flatten(), buf["values"].flatten()
        out["explained_var"] = float(1 - (y - yhat).var() / (y.var() + 1e-8))
        out["lr"] = self.opt.param_groups[0]["lr"]
        out["log_std_mean"] = float(self.policy.log_std.mean())
        return out

    # ------------------------------------------------------------------ diagnostics
    @staticmethod
    def estimate_error_by_blind_age(buf):
        """Rollout-time position-estimate error (m) per blind-age bin (vision mode)."""
        err = (buf["s_est"][..., :3] - buf["target"][..., :3]).norm(dim=-1).cpu()
        age = buf["blind_age"]
        out = {}
        for lo, hi in BLIND_BINS:
            m = (age >= lo) & (age <= hi)
            out[f"est_err_blind_{lo}_{hi if hi < 10 ** 9 else 'inf'}"] = (
                float(err[m].mean()) if m.any() else float("nan"))
            out[f"n_blind_{lo}_{hi if hi < 10 ** 9 else 'inf'}"] = int(m.sum())
        return out

    # ------------------------------------------------------------------ checkpointing
    def state_dict(self, extra=None):
        return dict(policy=self.policy.state_dict(), opt=self.opt.state_dict(), update=self.update,
                    global_step=self.global_step, cfg=vars(self.cfg), extra=extra or {})

    def load_state_dict(self, d):
        self.policy.load_state_dict(d["policy"])
        self.opt.load_state_dict(d["opt"])
        self.update, self.global_step = d["update"], d["global_step"]
        return d.get("extra", {})
