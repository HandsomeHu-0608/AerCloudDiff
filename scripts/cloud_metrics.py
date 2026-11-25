"""Alignment and event metrics for cloud-forecast evaluation."""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
from scipy.ndimage import uniform_filter


def equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and bool(
        np.array_equal(left, right, equal_nan=np.issubdtype(left.dtype, np.inexact))
    )


def samples(directory: Path) -> dict[str, Path]:
    paths = [directory] if directory.is_file() else list(directory.glob("*_pred.npz"))
    values = {}
    for path in paths:
        match = re.search(r"(20\d{8})", path.stem)
        if match is None:
            raise ValueError(f"Prediction filename must contain its YYYYMMDDHH initialization: {path.name}")
        key = match.group(1)
        if key in values:
            raise ValueError(f"Multiple predictions for initialization {key}: {directory}")
        values[key] = path
    if not values:
        raise ValueError(f"No prediction arrays under {directory}")
    return values


def validate(models: dict[str, Path]) -> tuple[list[str], dict[str, dict[str, Path]]]:
    maps = {name: samples(directory) for name, directory in models.items()}
    names = sorted(next(iter(maps.values())))
    for model, mapping in maps.items():
        if sorted(mapping) != names:
            raise ValueError(f"{model}: prediction sample names differ")
    reference = next(iter(models))
    for name in names:
        with np.load(maps[reference][name]) as left:
            for key in ("truth_k", "valid", "lon", "lat", "future_hours"):
                if key not in left:
                    raise ValueError(f"{reference}/{name}: missing {key}")
            if left["pred_k"].shape != (24, *left["truth_k"].shape[1:]) or left["truth_k"].shape[0] != 24:
                raise ValueError(f"{reference}/{name}: expected 24 lead-time prediction and truth arrays")
            for model in list(models)[1:]:
                with np.load(maps[model][name]) as right:
                    for key in ("truth_k", "valid", "lon", "lat", "future_hours"):
                        if key not in right or not equal(left[key], right[key]):
                            raise ValueError(f"{name}: {key} differs between {reference} and {model}")
    return names, maps


def div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def fractions_skill_score(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray, window: int) -> float:
    """Masked FSS for binary cloud masks in a square neighborhood window."""
    if window < 1 or window % 2 != 1:
        raise ValueError("FSS neighborhood window must be a positive odd integer")
    valid = np.asarray(valid, bool)
    if not np.any(valid):
        return float("nan")
    weight = uniform_filter(valid.astype(float), size=window, mode="constant")
    pred_fraction = uniform_filter((pred & valid).astype(float), size=window, mode="constant") / np.maximum(weight, 1e-12)
    truth_fraction = uniform_filter((truth & valid).astype(float), size=window, mode="constant") / np.maximum(weight, 1e-12)
    core = weight >= 0.999
    if not np.any(core):
        core = valid
    numerator = float(np.mean((pred_fraction[core] - truth_fraction[core]) ** 2))
    denominator = float(np.mean(pred_fraction[core] ** 2 + truth_fraction[core] ** 2))
    return 1.0 if denominator == 0 and numerator == 0 else 1.0 - numerator / denominator if denominator else 0.0
