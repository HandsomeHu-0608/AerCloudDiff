#!/usr/bin/env python3
"""Complete-month, distributed EMA evaluation with immutable source binding."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.aer_residual_final import (
    BASELINE_SHA256, NORMALIZATION_SHA256, COMPATIBLE_NORMALIZATION_SHA256, SAMPLING, evaluate_rollout, evaluation_config,
)
from src.models.b13_factory import build_b13_model
from src.models.diffusion import DDPMScheduler
from src.utils.aerosol_residual_checkpoint import ADAPTER_ONLY_CHECKPOINT_SCHEMA
from src.utils.b13_checkpoint import checkpoint_fingerprint, load_checkpoint_weights, write_json_atomic
from src.utils.config import load_config, save_config
from src.data.final_target_policy import POLICY_ID, load_policy


def target_policy_binding(config: dict) -> dict:
    """Bind the approved policy and exact audited manifest; no implicit opt-in."""
    policy = load_policy(config)
    if policy is None:
        return {}
    manifest = Path(config["dataset"]["known_missing_targets_manifest"])
    return {"known_missing_targets_policy": POLICY_ID,
            "known_missing_targets_manifest_sha256": checkpoint_fingerprint(manifest)}


def build_evaluation_request(config: dict, checkpoint_sha: str, kind: str, epoch: int,
                             split: str = "val") -> dict:
    """Canonical request used by evaluation and audited pre-training recovery."""
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return {"schema": "diffcloud_aer_residual_final_evaluation_v1", "kind": kind,
            "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": int(epoch),
            "normalization_sha256": NORMALIZATION_SHA256, "split": split,
            "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(), "sampling": SAMPLING,
            **target_policy_binding(config)}


def check_checkpoint(checkpoint: dict, fingerprint: str, config: dict,
                     *, verify_paper_baseline: bool = False) -> str:
    architecture = config["model"]["architecture"]
    policy_binding = target_policy_binding(config)
    if architecture == "grouped_conditional_unet3d_wrfcloud":
        if verify_paper_baseline and fingerprint != BASELINE_SHA256:
            raise ValueError("Reference is not the paper's original 2DAer EMA checkpoint")
        if (int(checkpoint.get("epoch", 0)) <= 0 or
                (checkpoint.get("condition_channels"), checkpoint.get("wrf_cloud_channels")) != (376, 23)):
            raise ValueError("Reference requires trained 2DAer weights, condition376/WRF23")
        if not isinstance(checkpoint.get("ema_model_state"), dict):
            raise ValueError("Reference requires exported EMA weights without fallback")
        return "baseline_reference"
    if verify_paper_baseline:
        raise ValueError("Paper baseline verification applies only to a 2DAer reference")
    if architecture != "grouped_aerosol_residual_st48_v2":
        raise ValueError("Final candidate requires the ST48 residual architecture")
    if not config["model"].get("residual_enabled", False) or int(checkpoint.get("epoch", 0)) <= 0:
        raise ValueError("An untrained or disabled residual cannot be a final candidate")
    if checkpoint.get("schema") != ADAPTER_ONLY_CHECKPOINT_SCHEMA:
        raise ValueError("Evaluation requires an adapter-only 3DAer checkpoint")
    if checkpoint.get("normalization_sha256") not in COMPATIBLE_NORMALIZATION_SHA256:
        raise ValueError("Unsupported checkpoint normalization identity")
    if checkpoint.get("architecture") != architecture or checkpoint.get("trained") is not True:
        raise ValueError("Candidate architecture or training provenance differs")
    for key in ("known_missing_targets_policy", "known_missing_targets_manifest_sha256"):
        if checkpoint.get(key) != policy_binding.get(key):
            raise ValueError("Candidate missing-target training policy or manifest differs")
    if checkpoint.get("residual_enabled") is not True or checkpoint.get("residual_nonzero") is not True:
        raise ValueError("Candidate was not validated as an active nonzero trained residual")
    state = checkpoint.get("ema_model_state")
    if not isinstance(state, dict):
        raise ValueError("Full exported EMA model state is required")
    import torch
    head = [tensor for name, tensor in state.items()
            if name.startswith("aerosol_residual_adapter.output_projection.")]
    if not head or not any(bool(torch.count_nonzero(tensor)) for tensor in head):
        raise ValueError("Zero residual output head cannot pass candidate validation")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in state.values()):
        raise ValueError("Nonfinite EMA state")
    return "trained_residual_candidate"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--verify-paper-baseline", action="store_true",
                        help="Require the original unpublished 2DAer reference fingerprint")
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = args.distributed or world > 1
    if args.distributed and world == 1:
        raise ValueError("Use torchrun to launch distributed evaluation")
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" and os.name != "nt" else "gloo",
                                timeout=timedelta(seconds=int(os.environ.get("DIFFCLOUD_DDP_TIMEOUT_SECONDS", "7200"))))
    try:
        config = evaluation_config(load_config(args.config))
        norm = Path(config["dataset"]["wrf_cloud_normalization_file"])
        norm_sha = checkpoint_fingerprint(norm)
        if norm_sha != NORMALIZATION_SHA256:
            raise ValueError("Final evaluation requires the original 304-file normalization")
        checkpoint_sha = checkpoint_fingerprint(args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        kind = check_checkpoint(checkpoint, checkpoint_sha, config,
                                verify_paper_baseline=args.verify_paper_baseline)
        request = build_evaluation_request(config, checkpoint_sha, kind,
                                           int(checkpoint.get("epoch", 0)), args.split)
        request_file = args.output_dir / "evaluation_request.json"
        error, resume = None, False
        if rank == 0:
            try:
                if request_file.exists():
                    if json.loads(request_file.read_text(encoding="utf-8")) != request:
                        raise ValueError("Evaluation directory belongs to a different checkpoint/protocol")
                    resume = True
                    complete_path = args.output_dir / "evaluation_complete.json"
                    if complete_path.exists():
                        complete = json.loads(complete_path.read_text(encoding="utf-8"))
                        for name, digest in complete["prediction_sha256"].items():
                            if checkpoint_fingerprint(args.output_dir / "prediction_arrays" / name) != digest:
                                raise ValueError(f"Previously completed prediction was changed: {name}")
                else:
                    if (args.output_dir / "prediction_arrays").exists():
                        raise ValueError("Unbound prediction arrays already exist in the requested output directory")
                    write_json_atomic(request_file, request)
                save_config(config, args.output_dir / "resolved_config.yaml")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        status = [error, resume]
        if distributed:
            dist.broadcast_object_list(status, src=0)
        error, resume = status
        if error:
            raise RuntimeError(error)
        model = build_b13_model(config["model"], int(checkpoint.get("target_channels", 1)),
                                int(checkpoint.get("condition_channels", 376)),
                                int(checkpoint.get("wrf_cloud_channels", 23))).to(device)
        load_info = load_checkpoint_weights(model, checkpoint, weights="ema", strict=True, allow_fallback=False)
        scheduler = DDPMScheduler(steps=1000, device=str(device))
        metrics = evaluate_rollout(config, model, scheduler, device, split=args.split,
                                   distributed=distributed, output_dir=args.output_dir,
                                   allow_resume_predictions=resume)
        if rank == 0:
            products = {path.name: checkpoint_fingerprint(path)
                        for path in sorted((args.output_dir / "prediction_arrays").glob("*_pred.npz"))}
            metadata = {**request, "checkpoint": str(args.checkpoint.resolve()),
                        "weights_requested": "ema", "weights_loaded": load_info["weights_loaded"],
                        "normalization_file": str(norm.resolve()), "prediction_sha256": products,
                        "n_dates": metrics["n_dates"], "n_date_leads": metrics["n_date_leads"],
                        "n_scored_date_leads": metrics["n_scored_date_leads"],
                        "per_date_valid_lead_counts": metrics["per_date_valid_lead_counts"],
                        "architecture": config["model"]["architecture"],
                        "comparison_role": kind, "config": str((args.output_dir / "resolved_config.yaml").resolve())}
            actual_residual = metrics["residual_output_max_abs"]
            if kind == "trained_residual_candidate" and actual_residual <= 0:
                raise ValueError("No evidence of a nonzero active residual for this trained EMA")
            metadata["residual_output_max_abs"] = actual_residual
            write_json_atomic(args.output_dir / "sampling_metadata.json", metadata)
            write_json_atomic(args.output_dir / "evaluation_complete.json", metadata)
            print(json.dumps({k: metrics[k] for k in ("rmse_k", "mae_k", "n_dates")}), flush=True)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
