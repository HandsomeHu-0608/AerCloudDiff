from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_b13cascade import B13CascadeDataset
from src.evaluation.b13_stability import (
    build_unconditional_aerosol_3d,
    build_unconditional_inputs,
    make_torch_generator,
    stability_metrics_for_lead,
    resolved_sample_seed,
)
from src.models.b13_factory import b13_model_kwargs, build_b13_model
from src.models.diffusion import DDPMScheduler, SamplingConfig
from src.utils.b13_checkpoint import checkpoint_fingerprint, load_checkpoint_weights, write_json_atomic
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AerCloudDiff B13 forecasts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", "--base-checkpoint", dest="checkpoint", required=True)
    parser.add_argument(
        "--sampling-seed-mode",
        choices=["legacy_global", "sample_keyed"],
        default=None,
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "eval"])
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--sample-glob", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--weights", choices=["auto", "ema", "raw"], default=None)
    parser.add_argument("--clamp-mode", choices=["none", "final", "per_step", "dynamic_threshold"], default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--ensemble-members", type=int, default=None)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument(
        "--inference-amp",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use CUDA float16 autocast for lower inference memory use.",
    )
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=float,
        default=None,
        help="Optional per-process CUDA allocator limit. Helps keep a desktop responsive.",
    )
    parser.add_argument("--mode", choices=["autoregressive", "teacher_forced"], default="autoregressive")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-file prediction/truth/mask/lon/lat arrays as compressed NPZ products.",
    )
    parser.add_argument("--initialization-only", action="store_true",
                        help="Evaluate the epoch-zero 3DAer initialization with its zero residual bypassed.")
    parser.add_argument("--full-grid", action="store_true", help="Evaluate the full H/W grid instead of config patch_size.")
    parser.add_argument(
        "--metrics-min-lon",
        type=float,
        default=None,
        help="When set, compute main per-lead/mean metrics only for pixels with lon >= this value.",
    )
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        raise ImportError(f"PyTorch is required for B13 cascade evaluation: {exc}") from exc

    config = load_config(args.config)
    legacy_sampling_config = "sampling" not in config
    config.setdefault("dataset", {})
    config.setdefault("output", {})
    if args.initialization_only:
        if config.get("model", {}).get("architecture") != "grouped_aerosol_residual_st48_v2":
            parser.error("--initialization-only requires the 3DAer architecture")
        if args.weights == "raw":
            parser.error("--initialization-only requires the EMA initialization")
        config["model"]["residual_enabled"] = False
        config["dataset"].update(include_aerosol_full3d=False, require_aerosol_full3d_cache=False,
                                 require_predicted_history_cache=False, predicted_history_cache_dir=None,
                                 predicted_history_max_probability=0.0)
        config.setdefault("sampling", {})["use_ema"] = True
    if args.sample_dir is not None:
        config["dataset"]["sample_dir"] = args.sample_dir
    if args.sample_glob is not None:
        config["dataset"]["sample_glob"] = args.sample_glob
    if args.output_dir is not None:
        config["output"]["output_dir"] = args.output_dir
    if args.max_files is not None:
        config["dataset"]["max_files"] = args.max_files
    if args.full_grid:
        config["dataset"]["patch_size"] = None
        config["dataset"]["patch_sampling"] = "center"
        config["dataset"]["patches_per_file"] = 1
    if args.sampling_steps is not None:
        config.setdefault("sampling", {})["steps"] = int(args.sampling_steps)
    for key, value in (
        ("clamp_mode", args.clamp_mode),
        ("guidance_scale", args.guidance_scale),
        ("ensemble_members", args.ensemble_members),
        ("seed", args.sampling_seed),
        ("seed_mode", args.sampling_seed_mode),
    ):
        if value is not None:
            config.setdefault("sampling", {})[key] = value
    if args.weights in {"ema", "raw"}:
        config.setdefault("sampling", {})["use_ema"] = args.weights == "ema"
    resolve_wrfcloud_normalization(config, Path(args.checkpoint), Path(config["output"].get("output_dir", ".")))
    if args.mode == "autoregressive":
        config["dataset"]["teacher_forcing"] = False
    set_seed(int(config.get("training", {}).get("seed", 42)))

    out_dir = Path(config.get("output", {}).get("output_dir", "outputs/b13cascade_debug")) / f"eval_b13_{args.split}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = out_dir / "prediction_arrays"
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    dataset = B13CascadeDataset(config, split=args.split)
    if args.max_files is not None:
        dataset.files = dataset.files[: int(args.max_files)]
    if args.mode == "teacher_forced":
        dataset.teacher_forcing = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.inference_amp == "on" and device.type != "cuda":
        raise RuntimeError("--inference-amp on requires a CUDA GPU.")
    inference_amp = device.type == "cuda" and args.inference_amp != "off"
    if args.max_gpu_memory_gib is not None:
        if device.type != "cuda":
            raise RuntimeError("--max-gpu-memory-gib requires a CUDA GPU.")
        total_gib = float(torch.cuda.get_device_properties(device).total_memory) / (1024**3)
        if not 0.0 < float(args.max_gpu_memory_gib) < total_gib:
            raise ValueError(
                f"--max-gpu-memory-gib must be in (0, {total_gib:.2f}); "
                f"received {args.max_gpu_memory_gib}"
            )
        torch.cuda.set_per_process_memory_fraction(
            float(args.max_gpu_memory_gib) / total_gib,
            device=device,
        )
    # mmap avoids materialising the complete multi-gigabyte checkpoint in
    # pageable CPU memory before its selected tensors are copied into the
    # model.  Fall back for older PyTorch releases that lack mmap support.
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_epoch = int(ckpt.get("epoch", -1))
    checkpoint_initialization_only = ckpt.get("initialization_only") is True
    if args.initialization_only:
        if (not checkpoint_initialization_only or checkpoint_epoch != 0
                or ckpt.get("schema") != "diffcloud_aer_residual_final_initial_v2"):
            raise ValueError("--initialization-only requires an epoch-zero 3DAer initialization checkpoint")
        state = ckpt.get("ema_model_state", {})
        prefix = "aerosol_residual_adapter.output_projection."
        head = [state.get(prefix + name) for name in ("weight", "bias")]
        if any(not isinstance(tensor, torch.Tensor) or torch.count_nonzero(tensor).item() for tensor in head):
            raise ValueError("The epoch-zero 3DAer checkpoint must have a zero residual output head")
    elif checkpoint_initialization_only:
        raise ValueError("Use --initialization-only to evaluate an epoch-zero 3DAer checkpoint")
    condition_channels = ckpt.get("condition_channels")
    wrf_cloud_channels = ckpt.get("wrf_cloud_channels")
    target_channels = ckpt.get("target_channels")
    # Do not construct a full-grid condition tensor while the multi-gigabyte
    # checkpoint is still resident in CPU memory when its metadata is complete.
    # Python evaluates dict.get's default argument eagerly, so the previous
    # one-line get(..., sample0[...]) form defeated this memory safeguard.
    if condition_channels is None or wrf_cloud_channels is None or target_channels is None:
        sample0 = dataset.read_block(0, 0)
        if condition_channels is None:
            condition_channels = sample0["condition"].shape[0]
        if wrf_cloud_channels is None:
            wrf_cloud_channels = (
                sample0["wrf_cloud_condition"].shape[0] if "wrf_cloud_condition" in sample0 else 0
            )
        if target_channels is None:
            target_channels = sample0["target"].shape[0]
    condition_channels = int(condition_channels)
    wrf_cloud_channels = int(wrf_cloud_channels)
    target_channels = int(target_channels)
    model_cfg = config.get("model", {})
    sampling = SamplingConfig.from_config(config, legacy_config=legacy_sampling_config)
    rank = int(os.environ.get("RANK", "0"))
    set_seed(int(sampling.seed) + rank * 1000003)
    if sampling.guidance_scale != 0.0:
        if not bool(config.get("cfg", {}).get("enabled", False)):
            raise ValueError("guidance_scale != 0 requires cfg.enabled=true")
        if not bool(ckpt.get("cfg", {}).get("enabled", False)):
            raise ValueError("Checkpoint was not trained with CFG dropout; non-zero guidance is invalid")
    model = build_b13_model(model_cfg, target_channels, condition_channels, wrf_cloud_channels).to(device)
    weights_requested = args.weights or ("auto" if sampling.legacy_config else ("ema" if sampling.use_ema else "raw"))
    load_info = load_checkpoint_weights(
        model,
        ckpt,
        weights=weights_requested,
        strict=args.initialization_only,
        allow_fallback=sampling.legacy_config or weights_requested == "auto",
    )
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        print(f"Loaded checkpoint with {load_info}")
    # model now owns the selected weight tensors.  Releasing the CPU
    # checkpoint before the first full-grid H5 read substantially reduces
    # peak host-memory use on teaching workstations.
    del ckpt
    gc.collect()
    diffusion_steps = int(model_cfg.get("diffusion_steps", 1000))
    model.diffusion_steps = diffusion_steps
    scheduler = DDPMScheduler(steps=diffusion_steps, device=str(device))
    all_rows: list[dict[str, Any]] = []
    all_region_rows: list[dict[str, Any]] = []
    all_continuity_rows: list[dict[str, Any]] = []
    all_stability_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    total_inference_seconds = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for file_index, path in enumerate(dataset.files):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        try:
            result = predict_file(
                dataset,
                file_index,
                model,
                scheduler,
                device,
                model_cfg,
                mode=args.mode,
                sampling_config=sampling,
                cfg_cfg=config.get("cfg", {}),
                inference_amp=inference_amp,
            )
        except torch.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA memory limit reached. The process stopped before continuing "
                "to another date. Close other GPU programs and retry the one-date "
                "safe command; do not raise the configured memory limit on an 8 GiB GPU."
            ) from exc
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        total_inference_seconds += elapsed
        rows = per_lead_rows(result, path=str(path), metrics_min_lon=args.metrics_min_lon)
        all_rows.extend(rows)
        all_region_rows.extend(region_metric_rows(result, path=str(path)))
        continuity_rows = continuity_metric_rows(result, path=str(path), metrics_min_lon=args.metrics_min_lon)
        all_continuity_rows.extend(continuity_rows)
        stability_rows = []
        for lead_index in range(result["pred_k"].shape[0]):
            row = {
                "path": str(path),
                "lead_hour": lead_index + 1,
                **stability_metrics_for_lead(result, lead_index, metrics_min_lon=args.metrics_min_lon),
            }
            stability_rows.append(row)
        all_stability_rows.extend(stability_rows)
        summaries.append(
            {
                "path": str(path),
                "metrics": mean_numeric_rows(rows),
                "continuity": continuity_metrics(continuity_rows),
                "inference_seconds": elapsed,
            }
        )
        if args.save_predictions:
            prediction_path = prediction_dir / f"{path.stem}_pred.npz"
            if prediction_path.exists():
                raise FileExistsError(f"Refusing to overwrite saved prediction array: {prediction_path}")
            temporary_path = prediction_path.with_suffix(".partial.npz")
            np.savez_compressed(
                temporary_path,
                pred_k=result["pred_k"],
                pred_norm=result["pred_norm"],
                truth_k=result["truth_k"],
                valid=result["valid"],
                coszen=result["coszen"],
                daylight=result["daylight"],
                future_hours=result["future_hours"],
                cascade_block=result["cascade_block"],
                blend_weight_sum=result["blend_weight_sum"],
                source_window_count=result["source_window_count"],
                spatial_valid=result["spatial_valid"],
                lon=result["lon"],
                lat=result["lat"],
            )
            os.replace(temporary_path, prediction_path)
        print(
            f"Evaluated {path.name}: b13_rmse={summaries[-1]['metrics'].get('rmse_k'):.4f} "
            f"inference_seconds={elapsed:.2f}"
        )

    num_files = len(dataset.files)
    forecast_hours = int(getattr(dataset, "forecast_hours", 24))
    timing = {
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "num_files": num_files,
        "forecast_hours_per_file": forecast_hours,
        "sampling_steps": int(sampling.steps),
        "total_inference_seconds": total_inference_seconds,
        "seconds_per_file": total_inference_seconds / max(1, num_files),
        "seconds_per_forecast_hour": total_inference_seconds / max(1, num_files * forecast_hours),
        "files_per_minute": 60.0 * num_files / total_inference_seconds if total_inference_seconds > 0 else float("nan"),
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "inference_amp": bool(inference_amp),
        "max_gpu_memory_gib": args.max_gpu_memory_gib,
    }

    metrics = {
        "mode": args.mode,
        "split": args.split,
        "num_files": num_files,
        "metrics_min_lon": args.metrics_min_lon,
        "timing": timing,
        "mean": mean_numeric_rows(all_rows),
        "continuity": continuity_metrics(all_continuity_rows),
        "stability": mean_numeric_rows(all_stability_rows),
        "block_error_growth": block_error_growth_metrics(all_rows),
        "files": summaries,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    write_csv(out_dir / "per_lead_metrics.csv", all_rows)
    write_csv(out_dir / "region_metrics.csv", all_region_rows)
    write_csv(out_dir / "continuity_metrics.csv", all_continuity_rows)
    write_csv(out_dir / "stability_metrics.csv", all_stability_rows)
    provenance = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(args.checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_initialization_only": checkpoint_initialization_only,
        **load_info,
        "sampling": sampling.to_dict(),
        "sampling_timestep_sequence": scheduler.sampling_schedule(sampling.steps),
        "cfg": config.get("cfg", {"enabled": False}),
    }
    write_json_atomic(out_dir / "sampling_metadata.json", provenance)
    save_config(config, out_dir / "resolved_config.yaml")
    root_metrics_csv = Path(config.get("output", {}).get("output_dir", "outputs/b13cascade_debug")) / f"metrics_b13_{args.split}_{args.mode}.csv"
    root_mean = dict(metrics["mean"])
    root_mean.update(metrics["continuity"])
    write_csv(root_metrics_csv, [flatten_metrics(root_mean, args.split, args.mode, num_files, timing=timing)])
    print(json.dumps(metrics["mean"], indent=2, ensure_ascii=False))
    print(json.dumps({"timing": timing}, indent=2, ensure_ascii=False))
    print(f"Wrote B13 cascade evaluation outputs under {out_dir}")


def predict_file(
    dataset: B13CascadeDataset,
    file_index: int,
    model,
    scheduler,
    device,
    model_cfg: dict[str, Any],
    mode: str,
    sampling_config: SamplingConfig | None = None,
    cfg_cfg: dict[str, Any] | None = None,
    inference_amp: bool = False,
) -> dict[str, np.ndarray]:
    sampling = sampling_config or SamplingConfig(
        sampler=str(model_cfg.get("sampler", "ddpm")),
        steps=int(model_cfg.get("sampling_steps", scheduler.steps)),
        eta=float(model_cfg.get("ddim_eta", 0.0)),
        clamp_mode="per_step" if str(model_cfg.get("sampler", "ddpm")).lower() == "ddim" else "final",
        legacy_config=True,
    )
    members = []
    for member_index in range(sampling.ensemble_members):
        members.append(
            _predict_file_member(
                dataset,
                file_index,
                model,
                scheduler,
                device,
                model_cfg,
                mode,
                sampling,
                cfg_cfg or {},
                member_index,
                inference_amp,
            )
        )
    result = dict(members[0])
    if len(members) > 1:
        ensemble = np.stack([item["pred_k"] for item in members], axis=0)
        ensemble_norm = np.stack([item["pred_norm"] for item in members], axis=0)
        result["ensemble_pred_k"] = ensemble
        result["ensemble_pred_norm"] = ensemble_norm
        result["pred_k"] = np.mean(ensemble, axis=0).astype(np.float32)
        result["pred_norm"] = np.mean(ensemble_norm, axis=0).astype(np.float32)
    result["ensemble_members"] = np.asarray(len(members), dtype=np.int16)
    return result


def _predict_file_member(
    dataset: B13CascadeDataset,
    file_index: int,
    model,
    scheduler,
    device,
    model_cfg: dict[str, Any],
    mode: str,
    sampling: SamplingConfig,
    cfg_cfg: dict[str, Any],
    member_index: int,
    inference_amp: bool = False,
) -> dict[str, np.ndarray]:
    import torch

    sample0 = dataset.read_block(file_index, 0)
    y0 = int(sample0["y0"])
    x0 = int(sample0["x0"])
    h = int(sample0["target"].shape[-2])
    w = int(sample0["target"].shape[-1])
    y_slice = slice(y0, y0 + h)
    x_slice = slice(x0, x0 + w)
    history_k = sample0["bt13_history_k"].numpy()
    history_mask = sample0["bt13_history_valid_mask"].numpy().astype(bool)
    spatial_valid = sample0["spatial_valid_mask"].numpy().astype(bool)
    forecast_hours = int(getattr(dataset, "forecast_hours", 24))
    pred_sum = np.zeros((forecast_hours, h, w), dtype=np.float64)
    pred_norm_sum = np.zeros((forecast_hours, h, w), dtype=np.float64)
    blend_weight_sum = np.zeros(forecast_hours, dtype=np.float64)
    source_window_count = np.zeros(forecast_hours, dtype=np.int16)
    owner_weight = np.full(forecast_hours, -np.inf, dtype=np.float32)
    owner_block = np.full(forecast_hours, -1, dtype=np.int16)
    truth = np.full((forecast_hours, h, w), np.nan, dtype=np.float32)
    valid_out = np.zeros((forecast_hours, h, w), dtype=bool)
    coszen_out = np.zeros((forecast_hours, h, w), dtype=np.float32)
    daylight_out = np.zeros((forecast_hours, h, w), dtype=bool)
    history_time = 0
    blend_weights = np.asarray(dataset.cascade_blend_weights, dtype=np.float32)

    for block, window_start in enumerate(dataset.cascade_start_hours):
        if mode == "autoregressive" and window_start > history_time:
            if np.any(blend_weight_sum[history_time:window_start] <= 0.0):
                raise RuntimeError(
                    f"Cannot advance autoregressive history to hour {window_start}; fused predictions are incomplete"
                )
            append = (
                pred_sum[history_time:window_start]
                / blend_weight_sum[history_time:window_start, None, None]
            ).astype(np.float32)
            advance = int(window_start - history_time)
            append_mask = np.repeat(spatial_valid[None, :, :], advance, axis=0)
            history_k = np.concatenate([history_k[advance:], append], axis=0)
            history_mask = np.concatenate([history_mask[advance:], append_mask], axis=0)
            history_time = int(window_start)
        if mode == "teacher_forced":
            sample = dataset.read_block(file_index, block, y_slice=y_slice, x_slice=x_slice)
        else:
            sample = dataset.read_block(
                file_index,
                block,
                y_slice=y_slice,
                x_slice=x_slice,
                history_override=history_k,
                history_mask_override=history_mask,
            )
        condition = sample["condition"].unsqueeze(0).to(device)
        condition_group_ids = sample["condition_group_ids"].unsqueeze(0).to(device)
        wrf_cloud_condition = sample.get("wrf_cloud_condition")
        if wrf_cloud_condition is not None:
            wrf_cloud_condition = wrf_cloud_condition.unsqueeze(0).to(device)
        aerosol_3d_condition = sample.get("aerosol_3d_condition")
        if aerosol_3d_condition is not None:
            aerosol_3d_condition = aerosol_3d_condition.unsqueeze(0).to(device)
        unconditional_condition = None
        unconditional_wrf = None
        unconditional_aerosol = None
        if sampling.guidance_scale != 0.0:
            unconditional_condition, unconditional_wrf = build_unconditional_inputs(
                condition,
                wrf_cloud_condition,
                condition_group_ids,
                cfg_cfg,
            )
            unconditional_aerosol = build_unconditional_aerosol_3d(aerosol_3d_condition, cfg_cfg)
        generator = None
        if sampling.seed_mode == "sample_keyed" or sampling.ensemble_members > 1:
            seed = resolved_sample_seed(
                sampling.seed,
                dataset.files[int(file_index)],
                member_index,
                block,
            )
            generator = make_torch_generator(device, seed)
        # Use float16 activations for CUDA sampling when requested.
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if inference_amp and device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), amp_context:
            if hasattr(scheduler, "sample"):
                sampled = scheduler.sample(
                    model,
                    condition,
                    target_channels=1,
                    sampling=sampling,
                    model_kwargs=b13_model_kwargs(wrf_cloud_condition, aerosol_3d_condition),
                    unconditional_condition=unconditional_condition,
                    unconditional_model_kwargs=b13_model_kwargs(unconditional_wrf, unconditional_aerosol),
                    generator=generator,
                )
            else:  # test doubles and pre-Round3b external schedulers
                sampled = scheduler.p_sample_loop(
                    model,
                    condition,
                    target_channels=1,
                    num_steps=sampling.steps,
                    sampler=sampling.sampler,
                    eta=sampling.eta,
                    model_kwargs=b13_model_kwargs(wrf_cloud_condition, aerosol_3d_condition),
                )
            pred_norm = sampled.detach().cpu().numpy()[0, 0]
        pred_k = dataset.denormalize_bt13(pred_norm, clip=sampling.clamp_mode != "none")
        truth_k = sample["target_k"].numpy()[0]
        valid_window = sample["valid_mask"].numpy()[0].astype(bool)
        coszen_window = sample["coszen_future"].numpy()
        daylight_window = sample["daylight_mask_future"].numpy().astype(bool)
        hours = sample["future_hours"].numpy().astype(int)

        expected_hours = np.arange(window_start + 1, window_start + dataset.cascade_horizon + 1)
        if not np.array_equal(hours, expected_hours):
            raise ValueError(f"Window {block} hours {hours.tolist()} != expected {expected_hours.tolist()}")
        for local_index, hour in enumerate(hours):
            lead_index = int(hour) - 1
            weight = float(blend_weights[local_index])
            pred_sum[lead_index] += pred_k[local_index].astype(np.float64) * weight
            pred_norm_sum[lead_index] += pred_norm[local_index].astype(np.float64) * weight
            blend_weight_sum[lead_index] += weight
            source_window_count[lead_index] += 1
            if weight > owner_weight[lead_index]:
                owner_weight[lead_index] = weight
                owner_block[lead_index] = block
            truth[lead_index] = truth_k[local_index]
            valid_out[lead_index] |= valid_window[local_index]
            coszen_out[lead_index] = coszen_window[local_index]
            daylight_out[lead_index] = daylight_window[local_index]

    if np.any(blend_weight_sum <= 0.0):
        raise RuntimeError(f"Overlap cascade did not cover all lead hours: weights={blend_weight_sum.tolist()}")
    pred = (pred_sum / blend_weight_sum[:, None, None]).astype(np.float32)
    pred_norm_out = (pred_norm_sum / blend_weight_sum[:, None, None]).astype(np.float32)

    return {
        "pred_k": pred,
        "pred_norm": pred_norm_out,
        "normalized_out_of_range_fraction": np.asarray(
            np.mean((pred_norm_out < sampling.clamp_min) | (pred_norm_out > sampling.clamp_max)),
            dtype=np.float32,
        ),
        "truth_k": truth,
        "valid": valid_out,
        "coszen": coszen_out,
        "daylight": daylight_out,
        "future_hours": np.arange(1, forecast_hours + 1, dtype=np.int16),
        "cascade_block": owner_block,
        "blend_weight_sum": blend_weight_sum.astype(np.float32),
        "source_window_count": source_window_count,
        "spatial_valid": spatial_valid,
        "lon": read_lon_grid_for_eval(dataset, dataset.files[int(file_index)], y_slice, x_slice),
        "lat": read_lat_grid_for_eval(dataset, dataset.files[int(file_index)], y_slice, x_slice),
    }


def per_lead_rows(result: dict[str, np.ndarray], path: str, metrics_min_lon: float | None = None) -> list[dict[str, Any]]:
    rows = []
    pred = result["pred_k"]
    truth = result["truth_k"]
    valid = result["valid"]
    coszen = result["coszen"]
    daylight = result["daylight"]
    hours = result["future_hours"]
    blocks = result["cascade_block"]
    blend_weight_sum = result.get("blend_weight_sum")
    source_window_count = result.get("source_window_count")
    lon = result.get("lon")
    if metrics_min_lon is not None and lon is not None:
        domain_mask = lon >= float(metrics_min_lon)
    else:
        domain_mask = None
    for i in range(pred.shape[0]):
        metric_valid = valid[i] & domain_mask if domain_mask is not None else valid[i]
        base = {
            "path": path,
            "lead_hour": int(hours[i]),
            "lead_index": i,
            "cascade_block": int(blocks[i]),
            "blend_weight_sum": float(blend_weight_sum[i]) if blend_weight_sum is not None else 1.0,
            "source_window_count": int(source_window_count[i]) if source_window_count is not None else 1,
        }
        base.update(regression_metrics(pred[i], truth[i], metric_valid, prefix=""))
        base.update(regression_metrics(pred[i], truth[i], metric_valid & daylight[i], prefix="day_"))
        twilight = metric_valid & (coszen[i] > 0.0) & (coszen[i] <= 0.05)
        base.update(regression_metrics(pred[i], truth[i], twilight, prefix="twilight_"))
        night = metric_valid & (coszen[i] <= 0.0)
        base.update(regression_metrics(pred[i], truth[i], night, prefix="night_"))
        rows.append(base)
    return rows


def region_metric_rows(result: dict[str, np.ndarray], path: str) -> list[dict[str, Any]]:
    lon = result.get("lon")
    if lon is None:
        return []
    regions = {
        "west_lt80": lon < 80.0,
        "lon80_90": (lon >= 80.0) & (lon < 90.0),
        "east_ge80": lon >= 80.0,
    }
    rows = []
    pred = result["pred_k"]
    truth = result["truth_k"]
    valid = result["valid"]
    daylight = result["daylight"]
    hours = result["future_hours"]
    blocks = result["cascade_block"]
    for i in range(pred.shape[0]):
        for region_name, region_mask in regions.items():
            mask = valid[i] & region_mask
            base = {
                "path": path,
                "region": region_name,
                "lead_hour": int(hours[i]),
                "lead_index": i,
                "cascade_block": int(blocks[i]),
            }
            base.update(regression_metrics(pred[i], truth[i], mask, prefix=""))
            base.update(regression_metrics(pred[i], truth[i], mask & daylight[i], prefix="day_"))
            rows.append(base)
    return rows


def continuity_metric_rows(
    result: dict[str, np.ndarray],
    path: str,
    metrics_min_lon: float | None = None,
) -> list[dict[str, Any]]:
    pred = result["pred_k"]
    truth = result["truth_k"]
    valid = result["valid"]
    lon = result.get("lon")
    domain = np.ones(valid.shape[-2:], dtype=bool)
    if metrics_min_lon is not None and lon is not None:
        domain &= lon >= float(metrics_min_lon)
    rows: list[dict[str, Any]] = []
    for i in range(1, pred.shape[0]):
        transition_mask = valid[i - 1] & valid[i] & domain
        pred_increment = pred[i] - pred[i - 1]
        truth_increment = truth[i] - truth[i - 1]
        increment_error = pred_increment - truth_increment
        finite = (
            transition_mask
            & np.isfinite(pred_increment)
            & np.isfinite(truth_increment)
            & np.isfinite(increment_error)
        )
        lead_hour = i + 1
        if np.any(finite):
            error = increment_error[finite]
            pred_abs = np.abs(pred_increment[finite])
            truth_abs = np.abs(truth_increment[finite])
            row = {
                "path": path,
                "from_lead_hour": i,
                "to_lead_hour": lead_hour,
                "legacy_boundary": int(lead_hour in {7, 13, 19}),
                "increment_rmse_k": float(np.sqrt(np.mean(error**2))),
                "increment_mae_k": float(np.mean(np.abs(error))),
                "pred_abs_increment_k": float(np.mean(pred_abs)),
                "truth_abs_increment_k": float(np.mean(truth_abs)),
                "valid_count": int(np.count_nonzero(finite)),
            }
        else:
            row = {
                "path": path,
                "from_lead_hour": i,
                "to_lead_hour": lead_hour,
                "legacy_boundary": int(lead_hour in {7, 13, 19}),
                "increment_rmse_k": float("nan"),
                "increment_mae_k": float("nan"),
                "pred_abs_increment_k": float("nan"),
                "truth_abs_increment_k": float("nan"),
                "valid_count": 0,
            }
        rows.append(row)
    return rows


def continuity_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "temporal_gradient_rmse_k": float("nan"),
            "boundary_increment_rmse_k": float("nan"),
            "non_boundary_increment_rmse_k": float("nan"),
            "boundary_jump_ratio": float("nan"),
        }

    def root_mean_square_for(selected: list[dict[str, Any]]) -> float:
        values = np.asarray([row.get("increment_rmse_k", np.nan) for row in selected], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")

    boundary = [row for row in rows if int(row.get("legacy_boundary", 0)) == 1]
    non_boundary = [row for row in rows if int(row.get("legacy_boundary", 0)) == 0]
    overall_rmse = root_mean_square_for(rows)
    boundary_rmse = root_mean_square_for(boundary)
    non_boundary_rmse = root_mean_square_for(non_boundary)
    ratio = boundary_rmse / non_boundary_rmse if np.isfinite(non_boundary_rmse) and non_boundary_rmse > 0 else float("nan")
    metrics = {
        "temporal_gradient_rmse_k": overall_rmse,
        "boundary_increment_rmse_k": boundary_rmse,
        "non_boundary_increment_rmse_k": non_boundary_rmse,
        "boundary_jump_ratio": float(ratio),
    }
    for lead_hour in (7, 13, 19):
        selected = [row for row in rows if int(row.get("to_lead_hour", -1)) == lead_hour]
        metrics[f"increment_{lead_hour - 1:02d}_to_{lead_hour:02d}_rmse_k"] = root_mean_square_for(selected)
    return metrics


def read_lon_grid_for_eval(dataset: B13CascadeDataset, sample_path: Path, y_slice: slice, x_slice: slice) -> np.ndarray | None:
    import h5py

    with h5py.File(sample_path, "r") as f:
        if "grid_lon" in f:
            return f["grid_lon"][y_slice, x_slice].astype(np.float32)
        if "static_feat" in f:
            names = dataset._read_names(f, "static_var_names")
            for candidate in ["XLONG", "lon", "longitude", "LON"]:
                if candidate in names:
                    return f["static_feat"][y_slice, x_slice, names.index(candidate)].astype(np.float32)
    return None


def read_lat_grid_for_eval(dataset: B13CascadeDataset, sample_path: Path, y_slice: slice, x_slice: slice) -> np.ndarray | None:
    import h5py

    with h5py.File(sample_path, "r") as f:
        if "grid_lat" in f:
            return f["grid_lat"][y_slice, x_slice].astype(np.float32)
        if "static_feat" in f:
            names = dataset._read_names(f, "static_var_names")
            for candidate in ["XLAT", "lat", "latitude", "LAT"]:
                if candidate in names:
                    return f["static_feat"][y_slice, x_slice, names.index(candidate)].astype(np.float32)
    return None


def regression_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, prefix: str = "") -> dict[str, float]:
    valid = np.asarray(mask).astype(bool) & np.isfinite(truth)
    invalid_predictions = valid & ~np.isfinite(pred)
    if np.any(invalid_predictions):
        raise ValueError(
            f"Nonfinite predictions at {np.count_nonzero(invalid_predictions)} valid observation pixels"
        )
    p = np.asarray(pred)[valid]
    y = np.asarray(truth)[valid]
    if p.size == 0:
        return {f"{prefix}{key}": float("nan") for key in ["rmse_k", "mae_k", "bias_k", "pearson_r", "valid_count"]}
    diff = p - y
    if p.size > 1 and np.std(p) > 0.0 and np.std(y) > 0.0:
        r = float(np.corrcoef(p, y)[0, 1])
    else:
        r = float("nan")
    return {
        f"{prefix}rmse_k": float(np.sqrt(np.mean(diff**2))),
        f"{prefix}mae_k": float(np.mean(np.abs(diff))),
        f"{prefix}bias_k": float(np.mean(diff)),
        f"{prefix}pearson_r": r,
        f"{prefix}valid_count": float(p.size),
    }


def mean_numeric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.floating))})
    out = {}
    for key in keys:
        if key in {"lead_hour", "lead_index", "cascade_block"}:
            continue
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        out[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return out


def block_error_growth_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    blocks = sorted({int(row["cascade_block"]) for row in rows if int(row.get("cascade_block", -1)) >= 0})
    means = {}
    for block in blocks:
        values = np.asarray(
            [row.get("rmse_k", np.nan) for row in rows if int(row.get("cascade_block", -1)) == block],
            dtype=np.float64,
        )
        means[block] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    out = {f"block_{block}_rmse_k": value for block, value in means.items()}
    if blocks:
        baseline = means[blocks[0]]
        for block in blocks[1:]:
            out[f"block_{block}_rmse_growth_k"] = means[block] - baseline
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_metrics(metrics: dict[str, float], split: str, mode: str, num_files: int, timing: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"split": split, "mode": mode, "num_files": num_files}
    out.update(metrics)
    if timing:
        for key in ["device", "cuda_device_name", "sampling_steps", "total_inference_seconds", "seconds_per_file", "seconds_per_forecast_hour", "files_per_minute"]:
            out[f"inference_{key}"] = timing.get(key)
    return out


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def resolve_wrfcloud_normalization(config: dict[str, Any], checkpoint: Path, output_dir: Path) -> None:
    dataset_cfg = config.setdefault("dataset", {})
    if not dataset_cfg.get("include_wrf_cloud"):
        return
    configured = dataset_cfg.get("wrf_cloud_normalization_file")
    if configured and Path(configured).is_file():
        return
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    candidates = [
        run_dir / "reports" / "wrfcloud_v2_normalization.json",
        output_dir / "reports" / "wrfcloud_v2_normalization.json",
        PROJECT_ROOT / "assets/normalization/wrf_cloud.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            dataset_cfg["wrf_cloud_normalization_file"] = str(candidate)
            return


if __name__ == "__main__":
    main()
