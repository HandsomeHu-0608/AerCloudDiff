#!/usr/bin/env python3
"""Audit and compare a trained residual EMA against its supplied 2DAer reference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cloud_metrics import validate
from scripts.eval_3daer import target_policy_binding
from src.data.final_target_policy import load_policy
from src.evaluation.aer_residual_final import (
    BASELINE_SHA256, METRICS, NORMALIZATION_SHA256, SAMPLING, SELECTION_TOLERANCE_K,
    aggregate_rows, prediction_rows, validate_dates,
)
from src.utils.b13_checkpoint import checkpoint_fingerprint, write_json_atomic
from src.utils.config import load_config

WINDOWS = ((1, 6), (7, 12), (13, 18), (19, 24), (13, 24), (1, 24))


def paired_interval(values, *, replicates=10000, seed=20260920, block_length=1):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Paired bootstrap requires at least two finite date differences")
    if replicates < 100 or block_length < 1:
        raise ValueError("Invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = rng.integers(0, n, size=(replicates, (n + block_length - 1) // block_length))
    indices = ((starts[..., None] + np.arange(block_length)) % n).reshape(replicates, -1)[:, :n]
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, (.025, .975))
    return float(values.mean()), float(low), float(high)


def read_product(root: Path, expected_kind: str, *, expected_checkpoint_sha=None) -> dict:
    complete = json.loads((root / "evaluation_complete.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "sampling_metadata.json").read_text(encoding="utf-8"))
    request = json.loads((root / "evaluation_request.json").read_text(encoding="utf-8"))
    if complete != meta:
        raise ValueError(f"Completion metadata differs from sampling metadata: {root}")
    required_request = {"schema", "kind", "checkpoint_sha256", "checkpoint_epoch",
                        "normalization_sha256", "split", "config_sha256", "sampling"}
    optional_policy_keys = {"known_missing_targets_policy", "known_missing_targets_manifest_sha256"}
    if (set(request) not in (required_request, required_request | optional_policy_keys)
            or any(meta.get(key) != request[key] for key in request)):
        raise ValueError(f"Evaluation request differs from completion metadata: {root}")
    config = load_config(root / "resolved_config.yaml")
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    if hashlib.sha256(config_json.encode()).hexdigest() != meta.get("config_sha256"):
        raise ValueError(f"Resolved evaluation configuration changed: {root}")
    policy_binding = target_policy_binding(config)
    if (set(request) != required_request | set(policy_binding)
            or any(meta.get(key) != policy_binding.get(key) for key in optional_policy_keys)):
        raise ValueError(f"Missing-target policy or manifest changed: {root}")
    if meta.get("schema") != "diffcloud_aer_residual_final_evaluation_v1":
        raise ValueError(f"Unrecognized evaluation protocol: {root}")
    if meta.get("kind") != expected_kind or meta.get("split") != "test":
        raise ValueError(f"Unexpected evaluation role or split: {root}")
    if meta.get("normalization_sha256") != NORMALIZATION_SHA256 or meta.get("sampling") != SAMPLING:
        raise ValueError(f"Normalization/sampling protocol mismatch: {root}")
    if meta.get("weights_requested") != "ema" or meta.get("weights_loaded") != "ema":
        raise ValueError(f"Comparison requires EMA weights without fallback: {root}")
    if (meta.get("n_dates"), meta.get("n_date_leads")) != (31, 744):
        raise ValueError(f"Incomplete December evaluation: {root}")
    if meta.get("n_scored_date_leads", 744 if not policy_binding else None) != 744:
        raise ValueError(f"December must retain all 744 scored frames: {root}")
    if expected_checkpoint_sha is not None and meta.get("checkpoint_sha256") != expected_checkpoint_sha:
        raise ValueError("Reference checkpoint differs from the explicitly requested fingerprint")
    if expected_kind == "trained_residual_candidate":
        residual = float(meta.get("residual_output_max_abs", 0))
        if not 1 <= int(meta.get("checkpoint_epoch", 0)) <= 80 or not np.isfinite(residual) or residual <= 0:
            raise ValueError("Final candidate lacks trained, active nonzero residual evidence")
    arrays = sorted((root / "prediction_arrays").glob("*_pred.npz"))
    if {path.name for path in arrays} != set(meta["prediction_sha256"]):
        raise ValueError(f"Prediction inventory changed: {root}")
    for path in arrays:
        if checkpoint_fingerprint(path) != meta["prediction_sha256"][path.name]:
            raise ValueError(f"Prediction hash mismatch: {path}")
    return meta


def comparison_tables(detail: pd.DataFrame, *, replicates: int, seed: int):
    summaries, intervals, leads = [], [], []
    for metric in METRICS:
        paired = detail.pivot(index=["sample", "lead_hour"], columns="model", values=metric)
        paired = paired[["2DAer", "3DAer"]].replace([np.inf, -np.inf], np.nan).dropna()
        for lead in range(1, 25):
            frame = paired[paired.index.get_level_values("lead_hour") == lead]
            delta = frame["3DAer"].to_numpy() - frame["2DAer"].to_numpy()
            if len(delta) < 2:
                leads.append({"metric": metric, "lead_hour": lead,
                              "reference_mean": float(frame["2DAer"].mean()),
                              "candidate_mean": float(frame["3DAer"].mean()),
                              "delta_mean": float(delta.mean()) if len(delta) else float("nan"),
                              "ci95_low": float("nan"), "ci95_high": float("nan"), "n_dates": len(frame)})
                continue
            mean, lo, hi = paired_interval(delta, replicates=replicates, seed=seed)
            leads.append({"metric": metric, "lead_hour": lead, "reference_mean": frame["2DAer"].mean(),
                          "candidate_mean": frame["3DAer"].mean(), "delta_mean": mean,
                          "ci95_low": lo, "ci95_high": hi, "n_dates": len(frame)})
        for first, last in WINDOWS:
            sub = paired[paired.index.get_level_values("lead_hour").to_series(index=paired.index).between(first, last)]
            daily = sub.groupby(level="sample").mean().sort_index()
            if len(daily) < 2:
                delta = float((daily["3DAer"]-daily["2DAer"]).mean())
                summaries.append({"window": f"{first:02d}-{last:02d}", "metric": metric,
                                  "reference_mean": float(daily["2DAer"].mean()),
                                  "candidate_mean": float(daily["3DAer"].mean()),
                                  "delta_mean": delta,
                                  "n_dates": len(daily), "n_date_leads": len(sub)})
                for block in (1, 3, 5):
                    intervals.append({"window": f"{first:02d}-{last:02d}", "metric": metric,
                                      "delta_mean": delta, "ci95_low": float("nan"),
                                      "ci95_high": float("nan"), "block_length_days": block,
                                      "bootstrap_replicates": replicates, "n_dates": len(daily)})
                continue
            window = f"{first:02d}-{last:02d}"
            reference, candidate = daily["2DAer"].to_numpy(), daily["3DAer"].to_numpy()
            summaries.append({"window": window, "metric": metric, "reference_mean": float(reference.mean()),
                              "candidate_mean": float(candidate.mean()), "delta_mean": float((candidate-reference).mean()),
                              "n_dates": len(daily), "n_date_leads": len(sub)})
            for block in (1, 3, 5):
                mean, lo, hi = paired_interval(candidate-reference, replicates=replicates, seed=seed, block_length=block)
                intervals.append({"window": window, "metric": metric, "delta_mean": mean,
                                  "ci95_low": lo, "ci95_high": hi, "block_length_days": block,
                                  "bootstrap_replicates": replicates, "n_dates": len(daily)})
    return pd.DataFrame(summaries), pd.DataFrame(intervals), pd.DataFrame(leads)


def build_report(candidate_dir: Path, reference_dir: Path, output_dir: Path, *, replicates=10000,
                 seed=20260920, reference_sha256=None):
    sources = {"2DAer": reference_dir.resolve(), "3DAer": candidate_dir.resolve()}
    metadata = {"2DAer": read_product(sources["2DAer"], "baseline_reference",
                                     expected_checkpoint_sha=reference_sha256),
                "3DAer": read_product(sources["3DAer"], "trained_residual_candidate")}
    policy_keys = ("known_missing_targets_policy", "known_missing_targets_manifest_sha256")
    if any(metadata["2DAer"].get(key) != metadata["3DAer"].get(key) for key in policy_keys):
        raise ValueError("Reference and candidate use different missing-target policies")
    names, maps = validate({key: root / "prediction_arrays" for key, root in sources.items()})
    validate_dates(names, "test")
    output_dir.mkdir(parents=True, exist_ok=True)
    request = {"schema": "diffcloud_aer_residual_final_comparison_v1",
               "evaluation_sha256": {key: checkpoint_fingerprint(root / "evaluation_complete.json") for key, root in sources.items()},
               "bootstrap_replicates": replicates, "seed": seed}
    request_path = output_dir / "comparison_request.json"
    if request_path.exists() and json.loads(request_path.read_text(encoding="utf-8")) != request:
        raise ValueError("Comparison directory is bound to different inputs or statistics settings")
    write_json_atomic(request_path, request)
    all_rows, metrics = [], {}
    for key, mapping in maps.items():
        policy = load_policy(load_config(sources[key] / "resolved_config.yaml"))
        rows = []
        for name in names:
            with np.load(mapping[name], allow_pickle=False) as data:
                rows.extend(prediction_rows(data, name, target_policy=policy)
                            if policy is not None else prediction_rows(data, name))
        metrics[key] = aggregate_rows(rows, samples=names, target_policy=policy)
        saved = json.loads((sources[key] / "metrics.json").read_text(encoding="utf-8"))
        for metric in ("rmse_k", "mae_k", "bias_k", "abs_bias_k"):
            saved_value = float(saved[metric])
            if not np.isfinite(saved_value) or abs(metrics[key][metric] - saved_value) > 1e-8:
                raise ValueError(f"Saved {key}/{metric} differs from recomputed arrays")
        all_rows.extend({"model": key, **row} for row in rows)
    detail = pd.DataFrame(all_rows)
    summary, intervals, leads = comparison_tables(detail, replicates=replicates, seed=seed)
    detail.to_csv(output_dir / "per_sample_lead_metrics.csv", index=False)
    summary.to_csv(output_dir / "window_summary.csv", index=False)
    intervals.to_csv(output_dir / "paired_deltas.csv", index=False)
    leads.to_csv(output_dir / "per_lead_summary.csv", index=False)
    overall = summary[summary.window == "01-24"]
    overall.to_csv(output_dir / "overall_table.csv", index=False)
    delta = metrics["3DAer"]["rmse_k"] - metrics["2DAer"]["rmse_k"]
    passed = delta <= SELECTION_TOLERANCE_K
    result = {**request, "status": "passed_fixed_benchmark" if passed else "failed_primary_goal",
              "primary_metric": "date_lead_macro_rmse_k", "delta_rmse_k": delta,
              "reference_rmse_k": metrics["2DAer"]["rmse_k"], "candidate_rmse_k": metrics["3DAer"]["rmse_k"],
              "tolerance_k": SELECTION_TOLERANCE_K, "trained_active_nonzero_residual": True,
              "all_other_metrics_are_reported_not_gates": True,
              "independent_unseen_test": False, "automatic_retraining": False}
    table = ["| 指标 | 原2DAer | 本轮残差3DAer | 差值 |", "|---|---:|---:|---:|"]
    for row in overall.itertuples(index=False):
        table.append(f"| {row.metric} | {row.reference_mean:.6f} | {row.candidate_mean:.6f} | {row.delta_mean:+.6f} |")
    report = "\n".join([
        "# 残差3DAer最后一轮：固定12月基准比较", "",
        f"主目标{'达到' if passed else '未达到'}：全域RMSE相对原2DAer变化 {delta:+.6f} K。",
        "本结论仅指固定评估集上的日期/时效等权RMSE，不等于所有指标均改善或统计学非劣证明。", "",
        *table, "", "统计域：共同观测有效区域、经度≥80°E，31天×24时效；真值、掩膜、经纬网格和时效轴已逐数组核对。",
        "每帧先计算空间指标，再日期/时效等权平均。Bias为预测减观测；abs_bias_k先逐帧取绝对值再平均。",
        "FSS沿用240/230/220 K事件阈值及11像素窗口，未将事件阈值用于主误差的像元筛选。", "",
        "paired_deltas.csv含日期配对bootstrap，以及3天、5天环形区块敏感性区间；区间为逐项95%区间，未做多重比较修正。",
        "区间跨零不能写成已证明不劣。12月曾参与前期诊断，属于固定基准比较，非完全未接触的独立泛化验证。",
        "本轮组合了分支增强、条件融合层解冻和训练配方调整；不能仅凭该比较归因三维气溶胶信息的净收益。", "",
        "## 文件调用关系", "",
        "- 重新计算指标：asset_manifest.json中的两组prediction_arrays及原指标实现。",
        "- 重新制作整体/逐时效图：overall_table.csv、per_lead_summary.csv。",
        "- 时段比较与统计分析：window_summary.csv、paired_deltas.csv。",
        "- 推理复现：两组sampling_metadata.json、resolved_config.yaml及其绑定的checkpoint/归一化/输入缓存。", "",
        "未达标时保留本次真实最佳训练模型及结果，不把原2DAer或零残差替换为训练成果，也不自动开启第二轮。", "",
    ])
    (output_dir / "report_zh.md").write_text(report, encoding="utf-8")
    manifest = {"schema": "diffcloud_aer_residual_final_analysis_assets_v1",
                "evaluations": {key: {"root": str(root), "metadata": metadata[key]} for key, root in sources.items()},
                "analysis_sources": {str(path.relative_to(ROOT)): checkpoint_fingerprint(path) for path in
                    (Path(__file__), ROOT / "src/evaluation/aer_residual_final.py",
                     ROOT / "src/data/final_target_policy.py", ROOT / "scripts/eval_3daer.py",
                     ROOT / "scripts/analyze_cloud_structure_metrics.py", ROOT / "scripts/cloud_metrics.py")},
                "outputs": {p.name: checkpoint_fingerprint(p) for p in sorted(output_dir.iterdir())
                            if p.is_file() and p.suffix in (".csv", ".md")}}
    write_json_atomic(output_dir / "asset_manifest.json", manifest)
    write_json_atomic(output_dir / "comparison_complete.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260920)
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--reference-sha256", help="Expected SHA-256 of the independently supplied 2DAer reference")
    baseline.add_argument("--verify-paper-baseline", action="store_true",
                          help="Require the paper's original unpublished 2DAer reference")
    args = parser.parse_args()
    result = build_report(args.candidate_dir, args.reference_dir, args.output_dir,
                          replicates=args.bootstrap_replicates, seed=args.seed,
                          reference_sha256=BASELINE_SHA256 if args.verify_paper_baseline else args.reference_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
