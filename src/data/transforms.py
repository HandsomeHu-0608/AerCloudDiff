from __future__ import annotations

import numpy as np


def clean_finite(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)


def log1p_cot(cot: np.ndarray, clip_max: float = 200.0) -> np.ndarray:
    cot = np.asarray(cot, dtype=np.float32)
    cot = np.where(np.isfinite(cot), cot, np.nan)
    cot = np.clip(cot, 0.0, clip_max)
    return np.log1p(cot).astype(np.float32)


def normalize_log_cot(log_cot: np.ndarray, clip_max: float = 200.0) -> np.ndarray:
    scale = float(np.log1p(clip_max))
    if scale <= 0:
        return log_cot.astype(np.float32)
    return (log_cot / scale).astype(np.float32)


def channelwise_standardize(arr: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Standardize [C,T,H,W] arrays per channel using finite values.

    The arithmetic is deliberately in-place on a single output copy.  A
    vectorized ``(flat - means) / stds`` creates two full-size temporaries;
    for the 347-channel full-grid B13 condition that costs more than 600 MiB
    in addition to the input and can fail on workstations with constrained
    virtual memory.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    flat = arr.reshape(arr.shape[0], -1)
    means = np.zeros((arr.shape[0], 1), dtype=np.float32)
    stds = np.ones((arr.shape[0], 1), dtype=np.float32)
    for i in range(arr.shape[0]):
        finite = np.isfinite(flat[i])
        vals = flat[i, finite]
        if vals.size:
            means[i, 0] = vals.mean(dtype=np.float64)
            std = vals.std(dtype=np.float64)
            stds[i, 0] = std if std > eps else 1.0
    out = flat.copy()
    # Process one channel at a time. NumPy's whole-array nan_to_num creates
    # several CxTxHxW boolean work arrays even with copy=False.
    for i in range(out.shape[0]):
        channel = out[i]
        channel[~np.isfinite(channel)] = 0.0
        np.subtract(channel, means[i, 0], out=channel)
        np.divide(channel, stds[i, 0], out=channel)
        np.clip(channel, -10.0, 10.0, out=channel)
    return out.reshape(arr.shape)


def sincos_encode(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    radians = 2.0 * np.pi * np.asarray(values, dtype=np.float32) / float(period)
    return np.sin(radians).astype(np.float32), np.cos(radians).astype(np.float32)
