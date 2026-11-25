from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.data.dataset_b13cascade import CONDITION_GROUP_IDS


def stable_sample_seed(base_seed: int, sample_key: str, member: int, block: int) -> int:
    payload = f"{int(base_seed)}|{sample_key}|{int(member)}|{int(block)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def resolved_sample_seed(base_seed: int, sample_path: str | Path, member: int, block: int) -> int:
    """Read stored sampling seeds, or derive a seed from the sample filename."""
    path = Path(sample_path)
    if int(base_seed) == 42 and int(member) == 0 and path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as handle:
            values = handle.attrs.get("sampling_seeds_42")
        if values is not None:
            values = np.asarray(values)
            if (values.shape != (5,) or values.dtype.kind not in "ui"
                    or np.any(values < 0) or any(int(value) >= 2**63 - 1 for value in values)):
                raise ValueError("sampling_seeds_42 must contain five integers in [0, 2**63 - 1)")
            if not 0 <= int(block) < 5:
                raise ValueError("Stored sampling seeds cover forecast blocks 0 through 4")
            return int(values[int(block)])
    return stable_sample_seed(base_seed, path.name, member, block)


def make_torch_generator(device, seed: int):
    import torch

    generator = torch.Generator(device=str(device))
    generator.manual_seed(int(seed))
    return generator


def build_unconditional_inputs(condition, wrf_cloud_condition, condition_group_ids, cfg: dict[str, Any]):
    """Build the unconditional branch while optionally preserving static/time."""

    ids = condition_group_ids
    if ids.ndim == 1:
        ids = ids.unsqueeze(0).expand(condition.shape[0], -1)
    groups = []
    if bool(cfg.get("drop_history", True)):
        groups.append(CONDITION_GROUP_IDS["history_b13"])
    if bool(cfg.get("drop_wrf_regular", True)):
        groups.append(CONDITION_GROUP_IDS["wrf_regular"])
    if not bool(cfg.get("keep_static_time", True)):
        groups.append(CONDITION_GROUP_IDS["static_time"])
    unconditional = condition.clone()
    for group_id in groups:
        channel_mask = ids == group_id
        unconditional = unconditional.masked_fill(channel_mask[:, :, None, None, None], 0.0)
    unconditional_wrf = wrf_cloud_condition
    if wrf_cloud_condition is not None and bool(cfg.get("drop_wrf_cloud", True)):
        unconditional_wrf = wrf_cloud_condition.new_zeros(wrf_cloud_condition.shape)
    return unconditional, unconditional_wrf


def build_unconditional_aerosol_3d(aerosol_3d_condition, cfg: dict[str, Any]):
    """Build the CFG unconditional 3-D aerosol input alongside AOD/summary."""

    if aerosol_3d_condition is None:
        return None
    if bool(cfg.get("drop_aerosol", cfg.get("drop_wrf_regular", True))):
        return aerosol_3d_condition.new_zeros(aerosol_3d_condition.shape)
    return aerosol_3d_condition


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def masked_ssim(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray, data_range: float = 150.0) -> float:
    from scipy.ndimage import uniform_filter

    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(pred) & np.isfinite(truth)
    if valid.sum() < 25:
        return float("nan")
    fill_pred = np.where(valid, pred, 0.0)
    fill_truth = np.where(valid, truth, 0.0)
    weight = uniform_filter(valid.astype(np.float64), size=7, mode="constant")
    weight = np.maximum(weight, 1.0e-6)
    mu_x = uniform_filter(fill_pred, size=7, mode="constant") / weight
    mu_y = uniform_filter(fill_truth, size=7, mode="constant") / weight
    ex2 = uniform_filter(fill_pred**2, size=7, mode="constant") / weight
    ey2 = uniform_filter(fill_truth**2, size=7, mode="constant") / weight
    exy = uniform_filter(fill_pred * fill_truth, size=7, mode="constant") / weight
    var_x = np.maximum(ex2 - mu_x**2, 0.0)
    var_y = np.maximum(ey2 - mu_y**2, 0.0)
    cov = exy - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    core = valid & (weight > 0.95)
    return float(np.mean(score[core])) if np.any(core) else float(np.mean(score[valid]))


def spatial_gradient_rmse(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    values = []
    for axis in (0, 1):
        dp = np.diff(pred, axis=axis)
        dt = np.diff(truth, axis=axis)
        v = np.take(valid, range(1, valid.shape[axis]), axis=axis) & np.take(
            valid, range(0, valid.shape[axis] - 1), axis=axis
        )
        if np.any(v):
            values.append((dp[v] - dt[v]) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(values)))) if values else float("nan")


def high_frequency_energy(field: np.ndarray, valid: np.ndarray, cutoff: float = 0.25) -> float:
    field = np.asarray(field, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(field)
    if valid.sum() < 64:
        return float("nan")
    centered = np.where(valid, field - np.mean(field[valid]), 0.0)
    window = np.outer(np.hanning(field.shape[0]), np.hanning(field.shape[1]))
    spectrum = np.abs(np.fft.fft2(centered * window)) ** 2
    fy = np.fft.fftfreq(field.shape[0])[:, None]
    fx = np.fft.fftfreq(field.shape[1])[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    total = spectrum[radius > 0].sum()
    return float(spectrum[radius >= cutoff].sum() / total) if total > 0 else float("nan")


def cold_cloud_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    threshold_k: float,
    lon: np.ndarray | None = None,
    lat: np.ndarray | None = None,
) -> dict[str, float]:
    from scipy.ndimage import label

    valid = np.asarray(valid, dtype=bool) & np.isfinite(pred) & np.isfinite(truth)
    p = (np.asarray(pred) <= threshold_k) & valid
    t = (np.asarray(truth) <= threshold_k) & valid
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t & valid))
    fn = int(np.sum(~p & t & valid))
    pred_count = int(p.sum())
    truth_count = int(t.sum())
    pl, pn = label(p)
    tl, tn = label(t)
    p_areas = np.bincount(pl.ravel())[1:] if pn else np.asarray([], dtype=int)
    t_areas = np.bincount(tl.ravel())[1:] if tn else np.asarray([], dtype=int)
    centroid_distance = float("nan")
    if pn and tn:
        p_id = int(np.argmax(p_areas)) + 1
        t_id = int(np.argmax(t_areas)) + 1
        py, px = np.mean(np.argwhere(pl == p_id), axis=0)
        ty, tx = np.mean(np.argwhere(tl == t_id), axis=0)
        if lon is not None and lat is not None:
            pi, pj = int(round(py)), int(round(px))
            ti, tj = int(round(ty)), int(round(tx))
            lon1, lat1 = float(lon[pi, pj]), float(lat[pi, pj])
            lon2, lat2 = float(lon[ti, tj]), float(lat[ti, tj])
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
            centroid_distance = 6371.0 * 2.0 * math.asin(min(1.0, math.sqrt(a)))
        else:
            centroid_distance = float(math.hypot(py - ty, px - tx))
    pred_min = float(np.min(np.asarray(pred)[valid])) if np.any(valid) else float("nan")
    truth_min = float(np.min(np.asarray(truth)[valid])) if np.any(valid) else float("nan")
    prefix = f"cold_{int(threshold_k)}k_"
    return {
        prefix + "csi": _safe_div(tp, tp + fp + fn),
        prefix + "pod": _safe_div(tp, tp + fn),
        prefix + "far": _safe_div(fp, tp + fp),
        prefix + "area_ratio": _safe_div(pred_count, truth_count),
        prefix + "area_bias_fraction": _safe_div(pred_count - truth_count, truth_count),
        prefix + "component_count_pred": float(pn),
        prefix + "component_count_truth": float(tn),
        prefix + "largest_area_pred": float(p_areas.max()) if p_areas.size else 0.0,
        prefix + "largest_area_truth": float(t_areas.max()) if t_areas.size else 0.0,
        prefix + "largest_centroid_distance_km": centroid_distance,
        prefix + "minimum_bt_bias_k": pred_min - truth_min,
    }


def cold_core_intensity_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    primary_threshold_k: float = 230.0,
    fallback_threshold_k: float = 240.0,
    min_pixels: int = 32,
) -> dict[str, float]:
    """Score intensity retention inside the largest truth-defined cold core.

    The primary threshold is preferred.  If it contains too few valid pixels,
    the fallback threshold is used.  Metrics are always evaluated on the
    largest connected truth component so false cold clouds elsewhere cannot
    make the core-intensity score look artificially good.
    """

    from scipy.ndimage import label

    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    finite = np.asarray(valid, dtype=bool) & np.isfinite(pred) & np.isfinite(truth)
    primary = finite & (truth <= float(primary_threshold_k))
    if int(primary.sum()) >= int(min_pixels):
        threshold = float(primary_threshold_k)
        candidate = primary
        used_fallback = 0.0
    else:
        threshold = float(fallback_threshold_k)
        candidate = finite & (truth <= threshold)
        used_fallback = 1.0

    labels, count = label(candidate)
    prefix = "cold_core_"
    empty = {
        prefix + "threshold_k": threshold,
        prefix + "used_fallback": used_fallback,
        prefix + "pixel_count": 0.0,
        prefix + "rmse_k": float("nan"),
        prefix + "mae_k": float("nan"),
        prefix + "mean_bias_k": float("nan"),
        prefix + "minimum_bt_bias_k": float("nan"),
        prefix + "retention_fraction": float("nan"),
    }
    if count == 0:
        return empty

    areas = np.bincount(labels.ravel())[1:]
    largest_id = int(np.argmax(areas)) + 1
    core = labels == largest_id
    if int(core.sum()) < int(min_pixels):
        return empty

    diff = pred[core] - truth[core]
    return {
        prefix + "threshold_k": threshold,
        prefix + "used_fallback": used_fallback,
        prefix + "pixel_count": float(core.sum()),
        prefix + "rmse_k": float(np.sqrt(np.mean(diff**2))),
        prefix + "mae_k": float(np.mean(np.abs(diff))),
        prefix + "mean_bias_k": float(np.mean(diff)),
        prefix + "minimum_bt_bias_k": float(np.min(pred[core]) - np.min(truth[core])),
        prefix + "retention_fraction": float(np.mean(pred[core] <= threshold)),
    }


def ensemble_metrics(members: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    members = np.asarray(members, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(truth)
    if members.ndim != 3 or members.shape[0] < 2 or not np.any(valid):
        return {
            "ensemble_crps_k": float("nan"),
            "ensemble_spread_skill_ratio": float("nan"),
            "ensemble_member_diversity_k": float("nan"),
            "ensemble_mean_rmse_k": float("nan"),
        }
    x = members[:, valid]
    y = truth[valid][None, :]
    term1 = np.mean(np.abs(x - y), axis=0)
    pairwise = np.abs(x[:, None, :] - x[None, :, :]).mean(axis=(0, 1))
    crps = np.mean(term1 - 0.5 * pairwise)
    mean = np.mean(x, axis=0)
    spread = float(np.sqrt(np.mean(np.var(x, axis=0, ddof=1))))
    skill = float(np.sqrt(np.mean((mean - y[0]) ** 2)))
    diversity = float(np.sqrt(np.mean((x - mean[None, :]) ** 2)))
    result = {
        "ensemble_crps_k": float(crps),
        "ensemble_spread_skill_ratio": _safe_div(spread, skill),
        "ensemble_member_diversity_k": diversity,
        "ensemble_mean_rmse_k": skill,
        "ensemble_spread_k": spread,
    }
    ranks = np.sum(x < y, axis=0)
    counts = np.bincount(ranks, minlength=members.shape[0] + 1)
    for rank, count in enumerate(counts):
        result[f"rank_histogram_bin_{rank}"] = float(count / max(1, ranks.size))
    return result


def stability_metrics_for_lead(
    result: dict[str, np.ndarray],
    lead_index: int,
    metrics_min_lon: float | None = None,
) -> dict[str, float]:
    pred = result["pred_k"][lead_index]
    members = result.get("ensemble_pred_k")
    if members is not None:
        pred = np.mean(members[:, lead_index], axis=0)
    truth = result["truth_k"][lead_index]
    valid = result["valid"][lead_index]
    finite = valid & np.isfinite(pred) & np.isfinite(truth)
    lon = result.get("lon")
    if metrics_min_lon is not None and lon is not None:
        finite &= np.asarray(lon) >= float(metrics_min_lon)
    out = {
        "ssim": masked_ssim(pred, truth, finite),
        "spatial_gradient_rmse_k": spatial_gradient_rmse(pred, truth, finite),
        "high_frequency_energy_pred": high_frequency_energy(pred, finite),
        "high_frequency_energy_truth": high_frequency_energy(truth, finite),
    }
    out["high_frequency_energy_ratio"] = _safe_div(
        out["high_frequency_energy_pred"], out["high_frequency_energy_truth"]
    )
    for threshold in (220.0, 230.0, 240.0):
        out.update(cold_cloud_metrics(pred, truth, finite, threshold, result.get("lon"), result.get("lat")))
    out.update(cold_core_intensity_metrics(pred, truth, finite))
    if members is not None:
        out.update(ensemble_metrics(members[:, lead_index], truth, finite))
    return out
