"""v6_drop Step 6c: the demonstration dataset for behaviour cloning.

build(src, dst): one-time conversion of the recorded .npz chunks (results/demos/) into
  dst/images.u8      all pictures in one MEMORY-MAPPED file (N x 160 x 256 bytes, ~5.2 GB
                     for 128k samples): read from disk piece by piece, so RAM stays free
  dst/arrays.npz     state, priv, move, drop, flown, flag, phase, ep, t (small)
  dst/episodes.npz   per episode: id, first row, length (rows of one episode are contiguous
                     and in time order)
Validation split (decided 29 Oct): every episode whose id ends in 0 (ep % 10 == 0, ~10%).
Step 6d (29 Oct): DAgger rounds are built the same way into their own folders
(results/dagger_data/r1, ...); Demos([...]) reads all parts together, no copying.

Clips (decided 29 Oct): a training example is a clip of CLIP = 32 steps (3.2 s) preceded by
up to BURN = 16 steps (1.6 s) of memory warm-up. Warm-up steps get no loss. Clips are padded
to BURN + CLIP steps; 'mask' is 1 only on the clip's own steps.
"""
import glob
import os

import numpy as np

H, W = 160, 256
CLIP, BURN = 32, 16
SCALARS = ("state", "priv", "move", "drop", "flown", "flag", "phase", "ep", "t")


def build(src, dst):
    files = sorted(glob.glob(os.path.join(src, "demos_*.npz")) + glob.glob(os.path.join(src, "dagger_*.npz")))
    assert files, f"no demos_*.npz or dagger_*.npz in {src}"
    # pass 1: sizes and episode order
    index = []                                        # (ep, file, row0, n)
    for f in files:
        d = np.load(f)
        ep = d["ep"]
        starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
        ends = np.r_[starts[1:], len(ep)]
        index += [(int(ep[a]), f, int(a), int(b - a)) for a, b in zip(starts, ends)]
    index.sort(key=lambda x: x[0])
    n_total = sum(x[3] for x in index)
    os.makedirs(dst, exist_ok=True)
    img = np.lib.format.open_memmap(os.path.join(dst, "images.u8.npy"), mode="w+",
                                    dtype=np.uint8, shape=(n_total, H, W))
    out = {k: [] for k in SCALARS}
    ep_id, ep_start, ep_len = [], [], []
    row, cache_f, cache = 0, None, None
    for ep, f, a, n in index:
        if f != cache_f:
            cache, cache_f = np.load(f), f
            cache_imgs = cache["image"]
        img[row:row + n] = cache_imgs[a:a + n]
        for k in SCALARS:
            out[k].append(cache[k][a:a + n])
        t = cache["t"][a:a + n]
        assert np.array_equal(t, np.arange(n)), f"episode {ep}: steps not in order"
        ep_id.append(ep)
        ep_start.append(row)
        ep_len.append(n)
        row += n
    img.flush()
    np.savez(os.path.join(dst, "arrays.npz"), **{k: np.concatenate(v) for k, v in out.items()})
    np.savez(os.path.join(dst, "episodes.npz"), ep=np.array(ep_id), start=np.array(ep_start),
             length=np.array(ep_len))
    return n_total, len(ep_id)


class ImageStack:
    """Several memory-mapped picture files seen as ONE long array (Step 6d: the old demos
    plus one part per DAgger round, without copying pictures again). Rows are global."""

    def __init__(self, parts):
        self.parts = parts
        self.row0 = np.cumsum([0] + [len(p) for p in parts])
        self.shape = (int(self.row0[-1]),) + parts[0].shape[1:]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = np.arange(*idx.indices(len(self)))
        idx = np.asarray(idx)
        if idx.ndim == 0:
            k = int(np.searchsorted(self.row0, idx, side="right") - 1)
            return self.parts[k][int(idx) - self.row0[k]]
        out = np.empty((len(idx),) + self.shape[1:], np.uint8)
        k = np.searchsorted(self.row0, idx, side="right") - 1
        for part in np.unique(k):
            sel = k == part
            out[sel] = self.parts[part][idx[sel] - self.row0[part]]
        return out


class Demos:
    """Read-only view of one or more built datasets (a path or a list of paths).
    Step 6d: the list is [bc_data, dagger_data/r1, dagger_data/r2, ...]. Episode ids must
    be unique across the parts (demos 0..; DAgger round r: r x 10000 + i)."""

    def __init__(self, paths):
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        self.paths = [str(p) for p in paths]
        imgs, arrays, ep, start, length = [], {}, [], [], []
        row = 0
        for p in self.paths:
            im = np.load(os.path.join(p, "images.u8.npy"), mmap_mode="r")
            a = np.load(os.path.join(p, "arrays.npz"))
            for k in a.files:
                arrays.setdefault(k, []).append(a[k])
            e = np.load(os.path.join(p, "episodes.npz"))
            ep.append(e["ep"])
            start.append(e["start"] + row)
            length.append(e["length"])
            imgs.append(im)
            row += len(im)
        self.images = ImageStack(imgs)
        self.a = {k: np.concatenate(v) for k, v in arrays.items()}
        self.ep, self.start, self.length = np.concatenate(ep), np.concatenate(start), np.concatenate(length)
        assert len(np.unique(self.ep)) == len(self.ep), "episode ids repeat across the parts"
        self.part_sizes = [(p, len(im), len(e)) for p, im, e in zip(self.paths, imgs, ep)]
        self.val = (self.ep % 10) == 0

    def episodes(self, split):
        keep = self.val if split == "val" else ~self.val
        return np.flatnonzero(keep)

    def clip(self, e, s):
        """Episode index e; loss on steps max(s, 0) .. min(s + CLIP, n) - 1 (s may be < 0
        for the short first clip of an epoch). Up to BURN earlier steps are added as
        warm-up. Rows are LEFT-aligned (row 0 = first warm-up step; the memory starts from
        zero there, exactly as at a real episode start when s <= 0), padded at the end.
        Returns dict of arrays with T = BURN + CLIP rows plus 'mask' (loss rows)."""
        n, r0 = int(self.length[e]), int(self.start[e])
        lo, hi = max(s, 0), min(s + CLIP, n)
        b0 = max(0, lo - BURN)
        idx = np.arange(r0 + b0, r0 + hi)
        T = BURN + CLIP
        out = {"image": np.zeros((T, H, W), np.uint8)}
        out["image"][:len(idx)] = self.images[idx]
        for k in ("state", "priv", "move", "drop"):
            v = self.a[k][idx]
            z = np.zeros((T,) + v.shape[1:], v.dtype)
            z[:len(idx)] = v
            out[k] = z
        mask = np.zeros(T, np.float32)
        mask[lo - b0:hi - b0] = 1.0
        out["mask"] = mask
        return out

    def clip_starts(self, split, rng):
        """All clips of one epoch: every episode cut into CLIP-step pieces at a random
        offset, so each step gets its loss exactly once per epoch. Shuffled (e, s) list."""
        pairs = []
        for e in self.episodes(split):
            n = int(self.length[e])
            o = int(rng.integers(0, CLIP))
            pairs += [(e, s) for s in range(o - CLIP, n, CLIP) if s + CLIP > 0]
        idx = rng.permutation(len(pairs))
        return [pairs[i] for i in idx]


def batch(demos, pairs):
    """Stack clips into (T, B, ...) arrays."""
    clips = [demos.clip(e, s) for e, s in pairs]
    return {k: np.stack([c[k] for c in clips], axis=1) for k in clips[0]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the BC dataset from recorded demos")
    ap.add_argument("--src", default=os.path.join(os.path.dirname(__file__), "..", "results", "demos"))
    ap.add_argument("--dst", default=os.path.join(os.path.dirname(__file__), "..", "results", "bc_data"))
    # Step 6d, DAgger round 1:
    #   python -m v6_drop.learn.dataset --src v6_drop/results/dagger/r1 --dst v6_drop/results/dagger_data/r1
    args = ap.parse_args()
    n, m = build(args.src, args.dst)
    d = Demos(args.dst)
    print(f"[data] {n} samples, {m} episodes -> {os.path.abspath(args.dst)}; "
          f"train episodes {len(d.episodes('train'))}, val episodes {len(d.episodes('val'))}; "
          f"drop labels {int(d.a['drop'].sum())}")
