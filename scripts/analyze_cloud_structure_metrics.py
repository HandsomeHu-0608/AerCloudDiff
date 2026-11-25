#!/usr/bin/env python3
"""Compute full-field MS-SSIM and PCC from matched saved prediction arrays."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cloud_metrics import fractions_skill_score, validate


def parse_models(values: list[str]) -> dict[str, Path]:
    """Parse repeated NAME=PATH arguments without silently replacing a model."""
    models: dict[str, Path] = {}
    for value in values:
        name, separator, directory = value.partition("=")
        name, directory = name.strip(), directory.strip()
        if not separator or not name or not directory:
            raise ValueError(f"Invalid --model {value!r}: expected a nonempty NAME=PATH")
        if name in models:
            raise ValueError(f"Duplicate model name in --model: {name!r}")
        models[name] = Path(directory)
    return models


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=EVALUATION_ROOT")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metrics-min-lon", type=float, default=80.0)
    parser.add_argument("--thresholds-k", default="220,230,240")
    return parser.parse_args()


def local_terms(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray, data_range: float = 150.0) -> tuple[float, float]:
    valid = np.asarray(valid, bool)
    if valid.sum() < 25:
        return float("nan"), float("nan")
    weight = uniform_filter(valid.astype(float), size=7, mode="constant")
    weight = np.maximum(weight, 1e-12)
    pred_fill, truth_fill = np.where(valid, pred, 0.0), np.where(valid, truth, 0.0)
    mx, my = uniform_filter(pred_fill, 7, mode="constant") / weight, uniform_filter(truth_fill, 7, mode="constant") / weight
    vx = np.maximum(uniform_filter(pred_fill**2, 7, mode="constant") / weight - mx**2, 0.0)
    vy = np.maximum(uniform_filter(truth_fill**2, 7, mode="constant") / weight - my**2, 0.0)
    covariance = uniform_filter(pred_fill * truth_fill, 7, mode="constant") / weight - mx * my
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    luminance = (2 * mx * my + c1) / (mx**2 + my**2 + c1)
    contrast_structure = (2 * covariance + c2) / (vx + vy + c2)
    core = weight >= 0.95
    if not np.any(core):
        core = valid
    return float(np.mean(luminance[core])), float(np.mean(contrast_structure[core]))


def downsample(field: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = field.shape
    height, width = height - height % 2, width - width % 2
    field, valid = field[:height, :width], valid[:height, :width]
    field4, valid4 = field.reshape(height // 2, 2, width // 2, 2), valid.reshape(height // 2, 2, width // 2, 2)
    count = valid4.sum(axis=(1, 3))
    result = np.where(count > 0, (field4 * valid4).sum(axis=(1, 3)) / np.maximum(count, 1), 0.0)
    return result, count == 4


def ms_ssim(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> float:
    # Standard five-scale MS-SSIM weights (Wang et al., 2003).
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    terms: list[tuple[float, float]] = []
    for _ in weights:
        luminance, contrast_structure = local_terms(pred, truth, valid)
        if not np.isfinite(luminance) or not np.isfinite(contrast_structure):
            return float("nan")
        terms.append((luminance, contrast_structure))
        # Downsample both fields against the same current-resolution mask;
        # the next mask requires all four source cells to be valid.
        pred, next_valid = downsample(pred, valid)
        truth, _ = downsample(truth, valid)
        valid = next_valid
    # Numerical noise can make local terms slightly negative; MS-SSIM is conventionally bounded at zero.
    value = np.prod([max(0.0, term[1]) ** weight for term, weight in zip(terms[:-1], weights[:-1])])
    value *= max(0.0, terms[-1][0] * terms[-1][1]) ** weights[-1]
    return float(value)


def main() -> None:
    args = arguments()
    try:
        models = parse_models(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    names, maps = validate(models)
    thresholds = [float(value) for value in args.thresholds_k.split(",")]
    rows = []
    for model, mapping in maps.items():
        for sample in names:
            with np.load(mapping[sample]) as data:
                domain = data["lon"] >= args.metrics_min_lon
                for index in range(24):
                    valid = domain & data["valid"][index] & np.isfinite(data["pred_k"][index]) & np.isfinite(data["truth_k"][index])
                    pred, truth = data["pred_k"][index].astype(float), data["truth_k"][index].astype(float)
                    pcc = float(np.corrcoef(pred[valid], truth[valid])[0, 1]) if valid.sum() > 1 else float("nan")
                    row = {"model": model, "sample": sample, "lead_hour": index + 1, "pcc": pcc, "ms_ssim": ms_ssim(pred, truth, valid)}
                    for threshold in thresholds:
                        cold_pred, cold_truth = pred <= threshold, truth <= threshold
                        for window in (5, 11, 21):
                            row[f"fss_{int(threshold)}k_{window}px"] = fractions_skill_score(cold_pred, cold_truth, valid, window)
                    rows.append(row)
    detail = pd.DataFrame(rows)
    summary = detail.groupby("model", as_index=False).mean(numeric_only=True).drop(columns=["lead_hour"], errors="ignore")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_csv.with_name(args.output_csv.stem + "_per_lead.csv"), index=False)
    summary.to_csv(args.output_csv, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
