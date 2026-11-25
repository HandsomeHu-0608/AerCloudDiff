from __future__ import annotations

import numpy as np


DEFAULT_COT_CLIP_MAX = 200.0


def target_to_physical(target: np.ndarray, cot_clip_max: float = DEFAULT_COT_CLIP_MAX) -> tuple[np.ndarray, np.ndarray]:
    cloud_probability = np.clip(target[..., 0], 0.0, 1.0)
    log_cot_norm = np.clip(target[..., 1], 0.0, 1.0)
    log_cot = log_cot_norm * np.log1p(cot_clip_max)
    cot = np.expm1(log_cot)
    return cloud_probability.astype(np.float32), cot.astype(np.float32)

