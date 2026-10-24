"""In-memory (and on-disk cached) dataset variants for the ensemble.

Raw variants (clean or one corruption/severity) are resized to IMG_SIZE once and
cached as uint8 arrays under CACHE_DIR; a member transform is then applied once
to the whole array, so training never touches the disk or cv2 per step.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from src.config import EXTDIR, IMG_SIZE
from src.noises.precorrupt import dsetmp

CACHE_DIR = os.path.join('data', 'cache_ens')
WORKERS = 16
MAX_CACHE_BYTES = 8 * 1024 ** 3  # stop writing new cache files beyond this; data is then just recomputed


def labels_of(ds, split):
    return np.array([lbl for _, lbl in dsetmp[ds][split].samples], dtype=np.int64)


def _read(ds, split, corruption, severity, i):
    path, label = dsetmp[ds][split].samples[i]
    if corruption is None:
        img = Image.open(path).convert('RGB')
    else:
        bname = os.path.splitext(os.path.basename(path))[0]
        npy = os.path.join(EXTDIR, corruption, str(severity), ds, split, str(label), f'{i}_{bname}.npy')
        img = Image.fromarray(np.load(npy))
    return np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))


def load_raw(ds, split, corruption=None, severity=0):
    """(N, IMG_SIZE, IMG_SIZE, 3) uint8 for the clean set or one corruption/severity."""
    name = 'clean_0' if corruption is None else f'{corruption}_{severity}'
    fp = os.path.join(CACHE_DIR, f'{ds}_{split}_{name}.npy')
    if os.path.exists(fp):
        return np.load(fp)
    n = len(dsetmp[ds][split])
    with ThreadPoolExecutor(WORKERS) as ex:
        arr = np.stack(list(ex.map(lambda i: _read(ds, split, corruption, severity, i), range(n), chunksize=256)))
    os.makedirs(CACHE_DIR, exist_ok=True)
    used = sum(f.stat().st_size for f in os.scandir(CACHE_DIR) if f.is_file())
    if used + arr.nbytes > MAX_CACHE_BYTES:
        print(f"[WARN] cache full ({used / 1024 ** 3:.1f} GiB), not caching {fp}")
        return arr
    tmp = fp[:-4] + '_tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, fp)
    return arr


def apply_tform(fn, raw):
    """Apply a per-image member transform to (N, H, W, 3) uint8 -> (N, H, W) uint8."""
    with ThreadPoolExecutor(WORKERS) as ex:
        return np.stack(list(ex.map(fn, raw, chunksize=256)))


def to_input(x):
    """uint8 (B, H, W) tensor -> float (B, 1, H, W) in [-1, 1] (== ToDtype(scale) + Normalize(0.5, 0.5))."""
    return x.float().div_(255).sub_(0.5).div_(0.5).unsqueeze(1)


class VariantBank:
    """All variants (clean + a member's corruptions x severities) of one split, transformed once.

    data[v, i] is sample i under variant v; variant 0 is always clean.
    """
    def __init__(self, ds, split, member, device, corrupt=False):
        self.variants = [(None, 0)]
        if corrupt:
            self.variants += [(c, s) for c in member['corruptions'] for s in member['severities']]
        shape = (len(self.variants), len(dsetmp[ds][split]), IMG_SIZE, IMG_SIZE)
        data = np.empty(shape, dtype=np.uint8)
        for v, (c, s) in enumerate(self.variants):
            data[v] = apply_tform(member['tform'], load_raw(ds, split, c, s))
        # keep on the GPU when it fits comfortably, else gather on the CPU
        self.device = device if data.nbytes < 1.5e9 else torch.device('cpu')
        self.data = torch.from_numpy(data).to(self.device)
        self.labels = torch.from_numpy(labels_of(ds, split)).to(device)

    def __len__(self):
        return self.data.shape[1]

    def draw(self, idx, p_corrupt, generator=None):
        """Batch for sample indices idx; each sample is corrupted with prob p_corrupt (random variant)."""
        idx = idx.to(self.device)
        if len(self.variants) == 1:
            return self.data[0, idx]
        v = torch.randint(1, len(self.variants), idx.shape, device=self.device, generator=generator)
        v[torch.rand(idx.shape, device=self.device, generator=generator) >= p_corrupt] = 0
        return self.data[v, idx]
