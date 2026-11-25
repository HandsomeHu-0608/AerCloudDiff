from __future__ import annotations

import argparse
from datetime import timedelta
import json
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_b13cascade import predict_file
from src.data.dataset_b13cascade import B13CascadeDataset
from src.models.b13_factory import build_b13_model
from src.models.diffusion import DDPMScheduler, SamplingConfig
from src.utils.b13_checkpoint import checkpoint_fingerprint, load_checkpoint_weights, write_json_atomic
from src.utils.config import load_config
from src.utils.seed import set_seed


def fixed_history_validity(result: dict) -> np.ndarray:
    """Validate predictions within the declared spatial domain."""
    pred = np.asarray(result["pred_k"])
    spatial = np.asarray(result["spatial_valid"], dtype=bool)
    if pred.ndim != 3 or spatial.shape != pred.shape[1:]:
        raise ValueError("History prediction and spatial mask shapes differ")
    valid = np.repeat(spatial[None], pred.shape[0], axis=0)
    if not valid.any():
        raise ValueError("History spatial domain is empty")
    if not np.isfinite(pred[valid]).all():
        raise FloatingPointError("Nonfinite prediction in the fixed history domain")
    return valid


def _validate_cache_provenance(
    output_dir: Path,
    existing_manifest: Path,
    partial_manifest: Path,
    cache_fingerprint: str,
) -> None:
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("cache_fingerprint") != cache_fingerprint:
            raise RuntimeError(f"Existing cache manifest does not match requested provenance: {existing_manifest}")
        return
    if partial_manifest.is_file():
        existing = json.loads(partial_manifest.read_text(encoding="utf-8"))
        if existing.get("cache_fingerprint") != cache_fingerprint:
            raise RuntimeError(f"Partial cache provenance does not match requested run: {partial_manifest}")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Cache directory has files but no matching provenance manifest: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build round2a full-grid predicted-history cache for round3a.")
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--sample-glob", default="sample_2025????12.h5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--weights", choices=["auto", "ema", "raw"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.overwrite:
        raise ValueError("Cache overwrite is disabled; use a new versioned output directory")
    if args.split != "train":
        raise ValueError("Predicted-history training cache must use --split train")

    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        raise ImportError(f"PyTorch is required to build predicted-history cache: {exc}") from exc

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(seconds=int(os.environ.get("DIFFCLOUD_DDP_TIMEOUT_SECONDS", "7200"))),
        )

    config = load_config(args.source_config)
    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg.update(
        {
            "sample_dir": args.sample_dir,
            "sample_glob": args.sample_glob,
            "patch_size": None,
            "patch_sampling": "center",
            "patches_per_file": 1,
            "teacher_forcing": False,
            "require_predicted_history_cache": False,
            "predicted_history_max_probability": 0.0,
        }
    )
    if args.max_files is not None:
        dataset_cfg["max_files"] = int(args.max_files)
    model_cfg = config.setdefault("model", {})
    model_cfg["sampling_steps"] = int(args.sampling_steps)
    sampler_name = str(model_cfg.get("sampler", "ddpm")).lower()
    config["sampling"] = {
        "sampler": sampler_name,
        "steps": int(args.sampling_steps),
        "eta": float(model_cfg.get("ddim_eta", 0.0)),
        "guidance_scale": 0.0,
        "clamp_mode": "per_step" if sampler_name == "ddim" else "final",
        "clamp_min": 0.0,
        "clamp_max": 1.0,
        "use_ema": args.weights != "raw",
        "seed": int(args.seed if args.seed is not None else config.get("training", {}).get("seed", 42)),
        "ensemble_members": 1,
        "seed_mode": "sample_keyed",
    }
    sampling = SamplingConfig.from_config(config)
    set_seed(int(sampling.seed) + rank)

    dataset = B13CascadeDataset(config, split=args.split)
    if args.max_files is not None:
        dataset.files = dataset.files[: int(args.max_files)]
    # The caller binds this trusted local checkpoint by SHA-256. Full training
    # checkpoints may contain optimizer/RNG metadata, not just tensor weights.
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    first = dataset.read_block(0, 0)
    condition_channels = int(checkpoint.get("condition_channels", first["condition"].shape[0]))
    target_channels = int(checkpoint.get("target_channels", first["target"].shape[0]))
    wrf_cloud_channels = int(
        checkpoint.get(
            "wrf_cloud_channels",
            first["wrf_cloud_condition"].shape[0] if "wrf_cloud_condition" in first else 0,
        )
    )
    model = build_b13_model(model_cfg, target_channels, condition_channels, wrf_cloud_channels).to(device)
    load_info = load_checkpoint_weights(
        model,
        checkpoint,
        weights=args.weights,
        strict=True,
        allow_fallback=args.weights == "auto",
    )
    model.eval()
    diffusion_steps = int(model_cfg.get("diffusion_steps", 1000))
    model.diffusion_steps = diffusion_steps
    scheduler = DDPMScheduler(steps=diffusion_steps, device=str(device))

    output_dir = Path(args.output_dir)
    source_sha256 = checkpoint_fingerprint(args.source_checkpoint)
    fingerprint_payload = {
        "schema": "diffcloud_b13_predicted_history_cache_v2",
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "sample_dir": str(Path(args.sample_dir).resolve()),
        "sample_glob": args.sample_glob,
        "split": args.split,
        "sampling": sampling.to_dict(),
        "weights_loaded": load_info["weights_loaded"],
    }
    cache_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    existing_manifest = output_dir / "manifest.json"
    partial_manifest = output_dir / "manifest.partial.json"
    provenance_error = None
    if rank == 0:
        try:
            _validate_cache_provenance(output_dir, existing_manifest, partial_manifest, cache_fingerprint)
            output_dir.mkdir(parents=True, exist_ok=True)
            if not existing_manifest.is_file():
                write_json_atomic(
                    partial_manifest,
                    {**fingerprint_payload, "cache_fingerprint": cache_fingerprint, "status": "in_progress"},
                )
        except Exception as exc:
            provenance_error = str(exc)
    if world_size > 1:
        error_box = [provenance_error]
        dist.broadcast_object_list(error_box, src=0)
        provenance_error = error_box[0]
        dist.barrier()
    if provenance_error is not None:
        raise RuntimeError(provenance_error)
    started = time.time()
    processed = 0
    skipped = 0
    for file_index in range(rank, len(dataset.files), world_size):
        sample_path = dataset.files[file_index]
        output_path = output_dir / f"{sample_path.stem}.npz"
        if output_path.is_file() and not args.overwrite:
            skipped += 1
            continue
        result = predict_file(
            dataset,
            file_index,
            model,
            scheduler,
            device,
            model_cfg,
            mode="autoregressive",
            sampling_config=sampling,
            cfg_cfg=config.get("cfg", {}),
        )
        cache_valid = fixed_history_validity(result)
        temp_path = output_dir / f"{sample_path.stem}.tmp.npz"
        np.savez_compressed(
            temp_path,
            pred_k=result["pred_k"].astype(np.float16),
            valid=cache_valid.astype(np.uint8),
        )
        temp_path.replace(output_path)
        processed += 1
        print(f"rank={rank} cached={sample_path.name} files_done={processed}", flush=True)

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        cache_files = sorted(output_dir.glob("sample_2025????12.npz"))
        manifest = {
            "schema": "diffcloud_b13_predicted_history_cache_v2",
            "cache_fingerprint": cache_fingerprint,
            "source_config": str(Path(args.source_config).resolve()),
            "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
            "source_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "source_checkpoint_sha256": source_sha256,
            "sample_dir": str(Path(args.sample_dir).resolve()),
            "sample_glob": args.sample_glob,
            "split": args.split,
            "sampling_steps": int(args.sampling_steps),
            "sampler": str(model_cfg.get("sampler", "ddpm")),
            "sampling": sampling.to_dict(),
            "sampling_timestep_sequence": scheduler.sampling_schedule(sampling.steps),
            "weights_requested": args.weights,
            "weights_loaded": load_info["weights_loaded"],
            "cfg": config.get("cfg", {"enabled": False}),
            "b13_clip": dataset_cfg.get("b13_clip", [180.0, 330.0]),
            "dtype": "float16",
            "expected_files": len(dataset.files),
            "cache_files": len(cache_files),
            "world_size": world_size,
            "elapsed_seconds": time.time() - started,
        }
        write_json_atomic(output_dir / "manifest.json", manifest)
        if partial_manifest.is_file():
            partial_manifest.unlink()
        if len(cache_files) != len(dataset.files):
            raise RuntimeError(
                f"Predicted-history cache incomplete: {len(cache_files)} files, expected {len(dataset.files)}"
            )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
