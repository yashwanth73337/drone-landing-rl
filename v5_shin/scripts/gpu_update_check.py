"""GPU fit + speed check for one PPO update on synthetic data of the real size (no env,
no training run, nothing saved). Reports peak GPU memory and seconds per update, for
several micro-batch sizes, so the vision config can be chosen for a 4 GB card.

Usage:  python -m v5_shin.scripts.gpu_update_check [--micro 4 8 16 32 128]
"""
import argparse
import time
from types import SimpleNamespace

import torch

from v5_shin.policies.encoder import OBS_H, OBS_W
from v5_shin.policies.ppo import PPOConfig, PPOTrainer
from v5_shin.policies.shin_policy import H


def synthetic_buffer(T, B, L, dev):
    g = torch.Generator().manual_seed(0)
    buf = {
        "image": torch.randint(0, 256, (T, B, OBS_H, OBS_W), dtype=torch.uint8, generator=g),
        "u": torch.randn(T, B, 7, generator=g), "critic": torch.randn(T, B, 13, generator=g),
        "target": torch.randn(T, B, 6, generator=g), "s_rel": torch.randn(T, B, 6, generator=g),
        "actions": torch.randn(T, B, 4, generator=g), "logp": torch.randn(T, B, generator=g),
        "values": torch.randn(T, B, generator=g), "starts": torch.zeros(T, B),
        "advantages": torch.randn(T, B, generator=g), "returns": torch.randn(T, B, generator=g),
        "h0": torch.zeros(T // L, 1, B, H), "c0": torch.zeros(T // L, 1, B, H),
    }
    return {k: v.to(dev) for k, v in buf.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=256)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs CUDA"
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB")
    for m in a.micro:
        cfg = PPOConfig(mode="vision", n_steps=a.n_steps, seq_len=32, micro_chunks=m)
        tr = PPOTrainer(SimpleNamespace(num_envs=a.n_envs), cfg, device="cuda", total_updates=10)
        buf = synthetic_buffer(a.n_steps, a.n_envs, 32, dev)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        try:
            t0 = time.time()
            tr.learn(buf)
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"micro_chunks={m:4d} ({m * 32:5d} frames/micro-batch): peak {peak:.2f} GiB, "
                  f"one update (5 epochs x 4 minibatches) {dt:.1f} s")
        except torch.cuda.OutOfMemoryError:
            print(f"micro_chunks={m:4d}: OUT OF MEMORY")
        del tr, buf
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
