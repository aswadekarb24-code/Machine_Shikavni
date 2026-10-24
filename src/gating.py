"""Fog gate: boosts the `fog` ensemble member on images that look fogged.

A fogged image has lower contrast and a brighter median than clean images. Both are measured as z-scores
against the dataset's clean training images (no corrupted data involved); fog_score is large only when
both deviate. The knob FOG_BOOST was set on BTSD/CTSD severities 1-3 and validated on severities 4-5 and GTSRB.
"""
import numpy as np

from src.preproc.cache import load_raw

FOG_BOOST = 4.0
_ref = {}


def _feats(raw):
    gray = raw.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)  # (N, H, W)
    flat = gray.reshape(len(gray), -1)
    return np.stack([np.log1p(flat.std(1)), np.median(flat, 1)], 1)


def clean_ref(ds):
    """Mean/std of the features over the clean training images of a dataset."""
    if ds not in _ref:
        f = _feats(load_raw(ds, 'train'))
        _ref[ds] = (f.mean(0), f.std(0) + 1e-6)
    return _ref[ds]


def fog_score(raw, ds):
    """(N,) score >= 0: min(contrast drop, median brightening) in clean-image standard deviations."""
    mu, sd = clean_ref(ds)
    z = (_feats(raw) - mu) / sd
    return np.maximum(np.minimum(-z[:, 0], z[:, 1]), 0.0)
