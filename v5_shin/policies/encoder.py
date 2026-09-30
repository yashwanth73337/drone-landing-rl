"""Visual front-end for the ArUco base configuration (V5 Block 7; SPEC §10, D9).

Paper (Sec. IV-B): without the keypoint encoder, "the landing pad is replaced by an
ArUco marker and the policy uses a standard CNN trained end-to-end". The output is
the image embedding l_t in R^512 (Fig. 3/4). The architecture is [unspecified];
SPEC §10 fixes it as below.

Pipeline:
  env side    512x320 uint8 gray  --cv2.INTER_AREA-->  256x160 uint8  (stored in rollouts)
  network     uint8 -> float (x/255 - 0.5) -> 5 x [conv s2 + ELU] -> Linear -> ELU -> l_t
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

OBS_W, OBS_H = 256, 160            # SPEC D9
EMBED_DIM = 512                    # [paper] l_t in R^512
CHANNELS = (32, 64, 64, 128, 128)  # [unspecified] SPEC §10
KERNELS = (5, 3, 3, 3, 3)          # [unspecified]


def downsample(gray):
    """512x320 -> 256x160 area average (exact 2x2 mean for this factor, rounded)."""
    return cv2.resize(gray, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA)


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, channels=CHANNELS, kernels=KERNELS,
                 in_hw=(OBS_H, OBS_W)):
        super().__init__()
        layers, c_in = [], 1
        for c, k in zip(channels, kernels):
            layers += [nn.Conv2d(c_in, c, k, stride=2, padding=k // 2), nn.ELU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, 1, *in_hw)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, embed_dim), nn.ELU())
        self.n_flat = n_flat

    @staticmethod
    def to_input(img_u8):
        """(B, H, W) or (B, 1, H, W) uint8 tensor -> float in [-0.5, 0.5]."""
        x = img_u8.float()
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return x / 255.0 - 0.5

    def forward(self, img_u8):
        return self.fc(self.conv(self.to_input(img_u8)))


def n_params(module):
    return sum(p.numel() for p in module.parameters())
