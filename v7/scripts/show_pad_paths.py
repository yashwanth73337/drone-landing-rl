"""v7 S1a: pictures and a video of the V7 pad's motion (no drone, no training).

  python -m v7.scripts.show_pad_paths                 # default: seeds 0, 1, 2, v_max 9
  python -m v7.scripts.show_pad_paths --v-max 8 --seconds 60 --video-seed 1

Writes to v7/results/pad_paths/ (git-ignored):
  pad_named_paths.png    top view of the 3 named test paths (straight, wave, corners)
  pad_random_paths.png   top view + speed over time of random flights, coloured by piece
  pad_random_<seed>.mp4  top-view video of one random flight (the pad as a turning square)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from matplotlib import animation

from v6_drop.envs import lmf2_params as P
from v7.envs.platform_v7 import PAD_SIZE, V_MAX, PlatformV7

OUT = os.path.join("v7", "results", "pad_paths")
COLORS = {"keep": "#8c959f", "speed up": "#2f6fdb", "slow down": "#d9822b",
          "corner-in": "#d9822b", "corner-turn": "#7a3fd1", "corner-out": "#2f6fdb", "wave": "#1a9e77"}
LABELS = {"keep": "keep speed", "speed up": "speed up", "slow down": "slow down / brake",
          "corner-turn": "90° corner", "wave": "wave"}
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"


def run(pl, seed, mode, seconds, v_max):
    pl.reset(np.random.default_rng(seed), mode=mode, v_max=v_max)
    rows = [(0.0, pl.x, pl.y, pl.psi, pl.v, pl.phase)]
    n = int(round(seconds / P.PHYSICS_DT))
    for k in range(n):
        pl.integrate(P.PHYSICS_DT)
        rows.append(((k + 1) * P.PHYSICS_DT, pl.x, pl.y, pl.psi, pl.v, pl.phase))
    return rows


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def draw_path(ax, rows, legend=False):
    t, x, y, psi, v, ph = zip(*rows)
    x, y = np.array(x), np.array(y)
    for name, c in COLORS.items():
        m = np.array([q == name for q in ph])
        if m.any():
            ax.scatter(x[m], y[m], s=3, color=c, lw=0)
    ax.plot(x[0], y[0], "o", color=INK, ms=6)
    if legend:
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], ls="", marker="o", color=COLORS[k], label=LABELS[k])
              for k in ("keep", "speed up", "slow down", "corner-turn", "wave")]
        ax.legend(handles=hs, frameon=False, fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    style(ax)


def named(pl, seconds, v_max, seed):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, mode, title in zip(axs, ("straight", "wave", "corners"),
                               ("1. Straight line", "2. Smooth wave", "3. 90° corners (brake, turn, speed up)")):
        rows = run(pl, seed, mode, seconds, v_max)
        draw_path(ax, rows, legend=(mode == "corners"))
        ax.set_title(f"{title}\nspeed at start {rows[0][4]:.1f} m/s", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("metres", color=MUTED, fontsize=9)
    fig.suptitle(f"Named test paths, top view, {seconds:.0f} s each, top speed {v_max:.0f} m/s. "
                 "Black dot = start.", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(OUT, "pad_named_paths.png")
    fig.savefig(path, dpi=140, facecolor="white")
    return path


def randoms(pl, seeds, seconds, v_max):
    fig, axs = plt.subplots(2, len(seeds), figsize=(5 * len(seeds), 8.5),
                            gridspec_kw=dict(height_ratios=[1.4, 1]))
    for i, s in enumerate(seeds):
        rows = run(pl, s, "random", seconds, v_max)
        draw_path(axs[0, i], rows, legend=(i == 0))
        axs[0, i].set_title(f"random flight, seed {s}: top view", loc="left", color=INK, fontsize=11)
        axs[0, i].set_xlabel("metres", color=MUTED, fontsize=9)
        t, _, _, _, v, ph = zip(*rows)
        t, v = np.array(t), np.array(v)
        for name, c in COLORS.items():
            m = np.array([q == name for q in ph])
            if m.any():
                axs[1, i].scatter(t[m], v[m], s=3, color=c, lw=0)
        axs[1, i].set_ylim(0, v_max + 0.5)
        axs[1, i].set_title("pad speed over time", loc="left", color=INK, fontsize=10)
        axs[1, i].set_xlabel("time (s)", color=MUTED, fontsize=9)
        axs[1, i].set_ylabel("m/s", color=MUTED, fontsize=9)
        style(axs[1, i])
    fig.suptitle("Random driver (training mode). The drone never sees this plan.",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUT, "pad_random_paths.png")
    fig.savefig(path, dpi=130, facecolor="white")
    return path


def video(pl, seed, seconds, v_max, fps=10):
    rows = run(pl, seed, "random", seconds, v_max)
    stride = int(round(1.0 / fps / P.PHYSICS_DT))
    frames = rows[::stride]
    xs, ys = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 7))
    style(ax)
    ax.plot(xs, ys, color=GRID, lw=1.5)
    ax.set_aspect("equal")
    pad = plt.Polygon(np.zeros((4, 2)), closed=True, color="#2f6fdb")
    ax.add_patch(pad)
    trail, = ax.plot([], [], color="#2f6fdb", lw=2)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", color=INK, fontsize=11,
                  family="monospace")
    half = PAD_SIZE / 2
    corners = np.array([[half, half], [-half, half], [-half, -half], [half, -half]])
    view = 25.0                                    # m, half-width of the moving window

    def upd(i):
        t, x, y, psi, v, ph = frames[i]
        c, s = np.cos(psi), np.sin(psi)
        R = np.array([[c, -s], [s, c]])
        pad.set_xy((R @ (corners * 3).T).T + [x, y])     # drawn 3x bigger so it is visible
        j = i * stride
        trail.set_data(xs[max(0, j - 300):j + 1], ys[max(0, j - 300):j + 1])
        ax.set_xlim(x - view, x + view)
        ax.set_ylim(y - view, y + view)
        txt.set_text(f"t {t:5.1f} s   speed {v:4.1f} m/s\n{LABELS.get(ph, ph)}")
        return pad, trail, txt

    ani = animation.FuncAnimation(fig, upd, frames=len(frames), blit=False)
    path = os.path.join(OUT, f"pad_random_{seed}.mp4")
    ani.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--v-max", type=float, default=V_MAX)
    ap.add_argument("--video-seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cid = p.connect(p.DIRECT)
    pl = PlatformV7(cid)
    for f in (named(pl, a.seconds, a.v_max, a.seeds[0]), randoms(pl, a.seeds, a.seconds, a.v_max),
              video(pl, a.video_seed, a.seconds, a.v_max)):
        print("wrote", f)
    p.disconnect(cid)


if __name__ == "__main__":
    main()
