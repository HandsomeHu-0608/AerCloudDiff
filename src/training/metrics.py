from __future__ import annotations

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(prob: np.ndarray, truth: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    prob = np.asarray(prob)
    truth = np.asarray(truth)
    mask = np.asarray(mask).astype(bool)
    if prob.ndim == 4:
        prob = prob[:, 0]
    pred = prob >= threshold
    obs = truth >= threshold
    valid = mask & np.isfinite(prob) & np.isfinite(truth)
    pred = pred[valid]
    obs = obs[valid]
    prob_v = np.clip(prob[valid], 0.0, 1.0)
    truth_v = obs.astype(np.float32)
    tp = float((pred & obs).sum())
    tn = float((~pred & ~obs).sum())
    fp = float((pred & ~obs).sum())
    fn = float((~pred & obs).sum())
    return {
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_div(tp, tp + fp + fn),
        "csi": _safe_div(tp, tp + fp + fn),
        "pod": _safe_div(tp, tp + fn),
        "far": _safe_div(fp, tp + fp),
        "brier": float(np.mean((prob_v - truth_v) ** 2)) if prob_v.size else float("nan"),
        "n": float(valid.sum()),
    }


def regression_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, prefix: str = "") -> dict[str, float]:
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    mask = np.asarray(mask).astype(bool) & np.isfinite(pred) & np.isfinite(truth)
    p = pred[mask]
    y = truth[mask]
    if p.size == 0:
        return {f"{prefix}{k}": float("nan") for k in ["rmse", "mae", "bias", "r", "n"]}
    diff = p - y
    if p.size > 1 and np.std(p) > 0 and np.std(y) > 0:
        r = float(np.corrcoef(p, y)[0, 1])
    else:
        r = float("nan")
    return {
        f"{prefix}rmse": float(np.sqrt(np.mean(diff**2))),
        f"{prefix}mae": float(np.mean(np.abs(diff))),
        f"{prefix}bias": float(np.mean(diff)),
        f"{prefix}r": r,
        f"{prefix}n": float(p.size),
    }


def per_lead_metrics(pred: np.ndarray, truth: np.ndarray, valid_mask: np.ndarray, cloudy_mask: np.ndarray) -> list[dict[str, float]]:
    out = []
    t_count = pred.shape[1] if pred.ndim == 4 else pred.shape[0]
    for t in range(t_count):
        prob_pred = pred[0, t] if pred.ndim == 4 else pred[t, 0]
        prob_true = truth[0, t] if truth.ndim == 4 else truth[t, 0]
        cot_pred = pred[1, t] if pred.ndim == 4 else pred[t, 1]
        cot_true = truth[1, t] if truth.ndim == 4 else truth[t, 1]
        vm = valid_mask[t] if valid_mask.ndim == 3 else valid_mask[0, t]
        cm = cloudy_mask[t] if cloudy_mask.ndim == 3 else cloudy_mask[0, t]
        row = {"lead_index": float(t)}
        row.update(binary_metrics(prob_pred, prob_true, vm))
        row.update(regression_metrics(cot_pred, cot_true, vm, prefix="valid_logcot_"))
        row.update(regression_metrics(cot_pred, cot_true, cm, prefix="cloudy_logcot_"))
        out.append(row)
    return out

