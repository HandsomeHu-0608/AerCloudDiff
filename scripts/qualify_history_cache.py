#!/usr/bin/env python3
"""Fast admission checks for Round3f-Full checkpoints and cache manifests.

This script never runs inference.  Checkpoint qualification consumes the fixed
eight-file, DDIM-100 rollout metrics already stored by the trainer.  Cache
qualification verifies the cache builder's provenance manifest and file count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _qualify_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        return {"passed": False, "reasons": [f"missing checkpoint: {checkpoint_path}"]}

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on server environment
        return {"passed": False, "reasons": [f"PyTorch import failed: {exc}"]}

    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        return {"passed": False, "reasons": [f"checkpoint load failed: {exc}"]}

    epoch = int(checkpoint.get("epoch", -1))
    rmse = checkpoint.get("rollout_val_rmse_k")
    bias = checkpoint.get("rollout_val_bias_k")
    pearson_r = checkpoint.get("rollout_val_pearson_r")
    config = checkpoint.get("config", {})
    training = config.get("training", {}) if isinstance(config, dict) else {}
    sampling = config.get("sampling", {}) if isinstance(config, dict) else {}
    reasons: list[str] = []

    if args.expected_epoch is not None and epoch != args.expected_epoch:
        reasons.append(f"epoch={epoch}, expected {args.expected_epoch}")
    if args.require_ema and "ema_state" not in checkpoint and "ema_model_state" not in checkpoint:
        reasons.append("EMA state is absent")
    if args.expected_condition_channels is not None:
        actual = int(checkpoint.get("condition_channels", -1))
        if actual != args.expected_condition_channels:
            reasons.append(f"condition_channels={actual}, expected {args.expected_condition_channels}")
    if args.expected_sampling_steps is not None:
        actual = int(training.get("rollout_validation_sampling_steps", -1))
        if actual != args.expected_sampling_steps:
            reasons.append(
                f"rollout_validation_sampling_steps={actual}, expected {args.expected_sampling_steps}"
            )
    if args.expected_max_files is not None:
        actual = int(training.get("rollout_validation_max_files", -1))
        if actual != args.expected_max_files:
            reasons.append(f"rollout_validation_max_files={actual}, expected {args.expected_max_files}")
    if args.expected_seed is not None:
        actual = int(sampling.get("seed", training.get("seed", -1)))
        if actual != args.expected_seed:
            reasons.append(f"sampling seed={actual}, expected {args.expected_seed}")

    metric_gate_requested = any(
        value is not None for value in (args.max_rmse_k, args.max_abs_bias_k, args.min_pearson_r)
    )
    if metric_gate_requested:
        for name, value in (
            ("rollout_val_rmse_k", rmse),
            ("rollout_val_bias_k", bias),
            ("rollout_val_pearson_r", pearson_r),
        ):
            if not _finite_number(value):
                reasons.append(f"{name} is missing or non-finite")
        if _finite_number(rmse) and args.max_rmse_k is not None and float(rmse) > args.max_rmse_k:
            reasons.append(f"RMSE={float(rmse):.4f} K > {args.max_rmse_k:.4f} K")
        if _finite_number(bias) and args.max_abs_bias_k is not None and abs(float(bias)) > args.max_abs_bias_k:
            reasons.append(f"|Bias|={abs(float(bias)):.4f} K > {args.max_abs_bias_k:.4f} K")
        if _finite_number(pearson_r) and args.min_pearson_r is not None and float(pearson_r) < args.min_pearson_r:
            reasons.append(f"R={float(pearson_r):.4f} < {args.min_pearson_r:.4f}")

    return {
        "mode": "checkpoint",
        "passed": not reasons,
        "reasons": reasons,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "epoch": epoch,
        "ema_present": "ema_state" in checkpoint or "ema_model_state" in checkpoint,
        "rollout": {"rmse_k": rmse, "bias_k": bias, "pearson_r": pearson_r},
        "protocol": {
            "sampling_steps": training.get("rollout_validation_sampling_steps"),
            "max_files": training.get("rollout_validation_max_files"),
            "seed": sampling.get("seed", training.get("seed")),
        },
    }


def _qualify_cache(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir).resolve()
    manifest_path = cache_dir / "manifest.json"
    reasons: list[str] = []
    if not manifest_path.is_file():
        return {
            "mode": "cache",
            "passed": False,
            "reasons": [f"missing complete manifest: {manifest_path}"],
            "cache_dir": str(cache_dir),
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "mode": "cache",
            "passed": False,
            "reasons": [f"manifest parse failed: {exc}"],
            "cache_dir": str(cache_dir),
        }

    if manifest.get("schema") != "diffcloud_b13_predicted_history_cache_v2":
        reasons.append(f"unexpected schema: {manifest.get('schema')!r}")
    if manifest.get("split") != "train":
        reasons.append(f"cache split is {manifest.get('split')!r}, expected 'train'")
    if manifest.get("weights_loaded") != "ema":
        reasons.append(f"weights_loaded={manifest.get('weights_loaded')!r}, expected 'ema'")
    if int(manifest.get("sampling_steps", -1)) != args.cache_sampling_steps:
        reasons.append(
            f"sampling_steps={manifest.get('sampling_steps')}, expected {args.cache_sampling_steps}"
        )
    sampling = manifest.get("sampling", {})
    if float(sampling.get("guidance_scale", float("nan"))) != 0.0:
        reasons.append(f"guidance_scale={sampling.get('guidance_scale')}, expected 0")
    if sampling.get("clamp_mode") != "per_step":
        reasons.append(f"clamp_mode={sampling.get('clamp_mode')!r}, expected 'per_step'")
    if int(sampling.get("seed", -1)) != args.expected_seed:
        reasons.append(f"seed={sampling.get('seed')}, expected {args.expected_seed}")

    expected_files = int(manifest.get("expected_files", -1))
    declared_files = int(manifest.get("cache_files", -1))
    actual_files = sum(1 for _ in cache_dir.glob(args.cache_glob))
    if expected_files < args.min_cache_files:
        reasons.append(f"expected_files={expected_files} < minimum {args.min_cache_files}")
    if declared_files != expected_files or actual_files != expected_files:
        reasons.append(
            f"cache count mismatch: expected={expected_files}, manifest={declared_files}, actual={actual_files}"
        )

    expected_source_sha = None
    if args.expected_source_checkpoint:
        source_checkpoint = Path(args.expected_source_checkpoint).resolve()
        if not source_checkpoint.is_file():
            reasons.append(f"missing expected source checkpoint: {source_checkpoint}")
        else:
            expected_source_sha = _sha256(source_checkpoint)
            if manifest.get("source_checkpoint_sha256") != expected_source_sha:
                reasons.append("source checkpoint SHA256 does not match cache provenance")

    return {
        "mode": "cache",
        "passed": not reasons,
        "reasons": reasons,
        "cache_dir": str(cache_dir),
        "manifest": str(manifest_path),
        "cache_fingerprint": manifest.get("cache_fingerprint"),
        "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
        "expected_source_checkpoint_sha256": expected_source_sha,
        "expected_files": expected_files,
        "actual_files": actual_files,
        "sampling": sampling,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify a Round3f-Full rollout checkpoint or predicted-history cache."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", help="Checkpoint containing fixed rollout metrics")
    mode.add_argument("--cache-dir", help="Completed predicted-history cache directory")
    parser.add_argument("--result-json", help="Write the machine-readable result atomically")
    parser.add_argument("--expected-epoch", type=int)
    parser.add_argument("--expected-condition-channels", type=int)
    parser.add_argument("--require-ema", action="store_true")
    parser.add_argument("--max-rmse-k", type=float)
    parser.add_argument("--max-abs-bias-k", type=float)
    parser.add_argument("--min-pearson-r", type=float)
    parser.add_argument("--expected-sampling-steps", type=int)
    parser.add_argument("--expected-max-files", type=int)
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--expected-source-checkpoint")
    parser.add_argument("--cache-sampling-steps", type=int, default=100)
    parser.add_argument("--min-cache-files", type=int, default=300)
    parser.add_argument(
        "--cache-glob",
        default="sample_2025????12.npz",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = _qualify_checkpoint(args) if args.checkpoint else _qualify_cache(args)
    if args.result_json:
        _write_json_atomic(Path(args.result_json), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("passed") else 3


if __name__ == "__main__":
    sys.exit(main())
