"""v6_drop Step 6c: the STUDENT network (the actor; later trained further by PPO in Step 7).

  EYES    CNN: grey picture 160 x 256 -> 256 numbers
          5 x [conv stride 2 + ELU], channels 32/64/64/128/128, kernels 5/3/3/3/3 (as V5)
          -> 128 x 5 x 8 = 5120 -> Linear -> 256 -> ELU
  MEMORY  LSTM: [256 picture numbers, 11 own-sensor numbers] -> 256 (one layer)
  HANDS   move: 256 -> 128 -> 4, tanh   (the action 'move' in [-1, 1])
          drop: 256 -> 64 -> 1 logit; bias starts at log(0.005 / 0.995), so the drop chance
                starts at 0.5% (Step 4, A3)
Size 256 decided 29 Oct "for now": if results are not good, try 512 (see SIZE).

Inputs are (T, B, ...) sequences: T time steps, B clips. hc = (h, c) LSTM state or None
(= zeros, an episode start). The LSTM output (T, B, 256) is also returned: the asymmetric
critic in Step 7 will read it DETACHED (decided 28 Oct).
"""
import math

import torch
import torch.nn as nn

SIZE = 256                     # picture summary and memory size (29 Oct: 256 for now)
CHANNELS = (32, 64, 64, 128, 128)
KERNELS = (5, 3, 3, 3, 3)
STATE_DIM = 11
P_DROP0 = 0.005


class Eyes(nn.Module):
    def __init__(self, out=SIZE, hw=(160, 256)):
        super().__init__()
        layers, c_in = [], 1
        for c, k in zip(CHANNELS, KERNELS):
            layers += [nn.Conv2d(c_in, c, k, stride=2, padding=k // 2), nn.ELU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, 1, *hw)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, out), nn.ELU())

    def forward(self, img_u8):
        """(N, H, W) uint8 -> (N, out)."""
        x = img_u8.float().unsqueeze(1) / 255.0 - 0.5
        return self.fc(self.conv(x))


class Student(nn.Module):
    def __init__(self, size=SIZE):
        super().__init__()
        self.size = size
        self.eyes = Eyes(size)
        self.memory = nn.LSTM(size + STATE_DIM, size, num_layers=1)
        self.move_head = nn.Sequential(nn.Linear(size, 128), nn.ELU(), nn.Linear(128, 4), nn.Tanh())
        self.drop_head = nn.Sequential(nn.Linear(size, 64), nn.ELU(), nn.Linear(64, 1))
        nn.init.constant_(self.drop_head[-1].bias, math.log(P_DROP0 / (1 - P_DROP0)))

    def forward(self, image, state, hc=None):
        """image (T, B, H, W) uint8, state (T, B, 11) float.
        Returns move (T, B, 4), drop_logit (T, B), features (T, B, size), (h, c)."""
        T, B = image.shape[:2]
        z = self.eyes(image.reshape(T * B, *image.shape[2:])).reshape(T, B, -1)
        feats, hc = self.memory(torch.cat([z, state], dim=-1), hc)
        return self.move_head(feats), self.drop_head(feats).squeeze(-1), feats, hc

    @torch.no_grad()
    def act(self, obs, hc=None, device="cpu"):
        """One decision from a LandingEnv observation (deterministic: mean move, drop if the
        chance is above 0.5). Returns (action dict, hc, drop chance)."""
        img = torch.as_tensor(obs["image"], device=device).reshape(1, 1, *obs["image"].shape)
        st = torch.as_tensor(obs["state"], device=device, dtype=torch.float32).reshape(1, 1, -1)
        move, logit, _, hc = self.forward(img, st, hc)
        p = torch.sigmoid(logit).item()
        return ({"move": move[0, 0].cpu().numpy(), "drop": int(p > 0.5)}, hc, p)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
