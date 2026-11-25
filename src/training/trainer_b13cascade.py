from __future__ import annotations

import csv
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler, random_split
except Exception as exc:  # pragma: no cover
    torch = None
    dist = None
    F = None
    DataLoader = None
    DistributedSampler = None
    DistributedDataParallel = None
    random_split = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from src.data.dataset_b13cascade import B13CascadeDataset
from src.models.b13_factory import b13_model_kwargs, build_b13_model
from src.models.diffusion import DDPMScheduler, SamplingConfig
from src.training.ema import ExponentialMovingAverage
from src.training.trainer import (
    _make_generator,
    _validation_timesteps,
    cleanup_distributed,
    distributed_context,
    move_batch,
)
from src.utils.b13_checkpoint import load_checkpoint_weights, select_checkpoint_state
from src.utils.config import save_config


def _require_torch() -> None:
    if torch is None:
        raise ImportError(f"PyTorch is required for B13 cascade training: {_TORCH_IMPORT_ERROR}")


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def rollout_guard_should_stop(action: str, bad_count: int, patience: int) -> bool:
    """Return whether a triggered rollout guard should terminate training."""

    normalized = str(action).strip().lower()
    if normalized not in {"stop", "warn"}:
        raise ValueError(
            "training.rollout_guard_action must be 'stop' or 'warn', "
            f"got {normalized!r}"
        )
    return normalized == "stop" and int(bad_count) >= max(1, int(patience))


def update_rollout_early_stopping(
    *,
    rmse_k: float,
    gate_passed: bool,
    best_rmse_k: float,
    bad_count: int,
    min_delta_k: float,
    patience: int,
) -> tuple[float, int, bool, bool, str]:
    """Update plateau early-stopping state for one scheduled rollout check."""

    value = float(rmse_k)
    minimum_delta = max(0.0, float(min_delta_k))
    eligible = bool(gate_passed) and np.isfinite(value)
    improved = eligible and (
        not np.isfinite(best_rmse_k) or value <= float(best_rmse_k) - minimum_delta
    )
    if improved:
        return value, 0, False, True, "improved"
    next_bad = int(bad_count) + 1
    if not np.isfinite(value):
        reason = "non_finite_rollout_rmse"
    elif not gate_passed:
        reason = "rollout_quality_gate_failed"
    else:
        reason = (
            f"rmse_improvement<{minimum_delta:.3f}K "
            f"(current={value:.4f}, best={float(best_rmse_k):.4f})"
        )
    return float(best_rmse_k), next_bad, next_bad >= max(1, int(patience)), False, reason


def evaluate_rollout_best_gate(
    rollout_values: dict[str, float],
    *,
    max_abs_bias_k: float = float("inf"),
    min_pearson_r: float = -float("inf"),
    max_late_cold_core_rmse_k: float = float("inf"),
) -> tuple[bool, str]:
    """Evaluate the physical-quality gates used when selecting ``best.pt``.

    An unconstrained (infinite) threshold deliberately ignores its metric,
    including a missing/NaN value, which preserves legacy checkpoint
    selection.  Once a finite threshold is configured, the corresponding
    rollout metric must be finite and satisfy it.
    """

    reasons: list[str] = []

    def _finite_metric(name: str) -> float | None:
        value = float(rollout_values.get(name, float("nan")))
        if not np.isfinite(value):
            reasons.append(f"{name}=non-finite")
            return None
        return value

    if np.isfinite(max_abs_bias_k):
        bias_k = _finite_metric("bias_k")
        if bias_k is not None and abs(bias_k) > max_abs_bias_k:
            reasons.append(f"abs_bias={abs(bias_k):.3f}>{max_abs_bias_k:.3f}K")

    if np.isfinite(min_pearson_r):
        pearson_r = _finite_metric("pearson_r")
        if pearson_r is not None and pearson_r < min_pearson_r:
            reasons.append(f"pearson_r={pearson_r:.4f}<{min_pearson_r:.4f}")

    if np.isfinite(max_late_cold_core_rmse_k):
        late_rmse_k = _finite_metric("late_cold_core_rmse_k")
        if late_rmse_k is not None and late_rmse_k > max_late_cold_core_rmse_k:
            reasons.append(
                f"late_core_rmse={late_rmse_k:.3f}>{max_late_cold_core_rmse_k:.3f}K"
            )

    return not reasons, "; ".join(reasons)


def is_rollout_best_candidate(
    *,
    epoch: int,
    start_epoch: int,
    rmse_k: float,
    best_rmse_k: float,
    guard_passed: bool,
    best_gate_passed: bool,
) -> bool:
    """Return whether the current rollout is eligible to replace ``best.pt``."""

    return bool(
        int(epoch) >= max(1, int(start_epoch))
        and np.isfinite(rmse_k)
        and guard_passed
        and best_gate_passed
        and float(rmse_k) < float(best_rmse_k)
    )


def build_l2sp_reference(
    model,
    checkpoint: dict[str, Any],
    *,
    weights: str = "ema",
) -> tuple[dict[str, Any], str]:
    """Materialize a frozen L2-SP reference aligned to trainable parameters."""

    state, loaded = select_checkpoint_state(checkpoint, weights=weights, allow_fallback=False)
    reference: dict[str, Any] = {}
    missing: list[str] = []
    for name, parameter in _unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue
        tensor = state.get(name)
        if tensor is None:
            missing.append(name)
            continue
        reference[name] = tensor.detach().to(device=parameter.device, dtype=torch.float32).clone()
    if missing:
        raise KeyError(f"L2-SP reference is missing {len(missing)} trainable parameters; first={missing[0]}")
    return reference, loaded


def l2sp_parameter_loss(model, reference: dict[str, Any]):
    """Unnormalised squared distance to a frozen source checkpoint."""

    total = None
    for name, parameter in _unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue
        target = reference[name]
        term = torch.sum((parameter.float() - target) ** 2)
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), device=next(_unwrap_model(model).parameters()).device)
    return total


def apply_cfg_condition_dropout(
    condition,
    wrf_cloud_condition,
    condition_group_ids,
    cfg: dict[str, Any],
    *,
    generator=None,
):
    """Apply sample-level grouped condition dropout for CFG training.

    Group ids are 0=history observations, 1=regular WRF, 2=static/time.
    WRF-cloud is a separate tensor. Disabled CFG is an exact no-op.
    """

    if not bool(cfg.get("enabled", False)):
        return condition, wrf_cloud_condition, torch.zeros(condition.shape[0], dtype=torch.bool, device=condition.device)
    probability = float(cfg.get("dropout_prob", 0.1))
    if probability <= 0.0:
        return condition, wrf_cloud_condition, torch.zeros(condition.shape[0], dtype=torch.bool, device=condition.device)
    selected = torch.rand(condition.shape[0], device=condition.device, generator=generator) < probability
    if not torch.any(selected):
        return condition, wrf_cloud_condition, selected
    ids = condition_group_ids
    if ids.ndim == 1:
        ids = ids.unsqueeze(0).expand(condition.shape[0], -1)
    dropped = condition.clone()
    groups = []
    if bool(cfg.get("drop_history", True)):
        groups.append(0)
    if bool(cfg.get("drop_wrf_regular", True)):
        groups.append(1)
    if not bool(cfg.get("keep_static_time", True)):
        groups.append(2)
    if groups:
        channel_drop = torch.zeros_like(ids, dtype=torch.bool)
        for group_id in groups:
            channel_drop |= ids == group_id
        mask = selected[:, None] & channel_drop
        dropped = dropped.masked_fill(mask[:, :, None, None, None], 0.0)
    dropped_wrf = wrf_cloud_condition
    if wrf_cloud_condition is not None and bool(cfg.get("drop_wrf_cloud", True)):
        dropped_wrf = wrf_cloud_condition.clone()
        dropped_wrf[selected] = 0.0
    return dropped, dropped_wrf, selected


def apply_cfg_aerosol_3d_dropout(aerosol_3d_condition, selected, cfg: dict[str, Any]):
    """Synchronize full-profile aerosol dropout with AOD/column summaries."""

    if aerosol_3d_condition is None:
        return None
    enabled = bool(cfg.get("enabled", False)) and bool(
        cfg.get("drop_aerosol", cfg.get("drop_wrf_regular", True))
    )
    if not enabled or not torch.any(selected):
        return aerosol_3d_condition
    dropped = aerosol_3d_condition.clone()
    dropped[selected] = 0.0
    return dropped


def make_training_noise(x0, diffusion_cfg: dict[str, Any], *, generator=None):
    if generator is None:
        noise = torch.randn_like(x0)
    else:
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    if not bool(diffusion_cfg.get("offset_noise_enabled", False)):
        return noise
    scale = float(diffusion_cfg.get("offset_noise_scale", 0.0))
    if scale == 0.0:
        return noise
    if str(diffusion_cfg.get("offset_noise_mode", "per_channel_time")) != "per_channel_time":
        raise ValueError("Only diffusion.offset_noise_mode=per_channel_time is supported")
    shape = (x0.shape[0], x0.shape[1], x0.shape[2], 1, 1)
    offset = torch.randn(shape, device=x0.device, dtype=x0.dtype, generator=generator)
    return noise + scale * offset


def _count_trainable_parameters(model) -> int:
    return sum(int(p.numel()) for p in _unwrap_model(model).parameters() if p.requires_grad)


def _distributed_average(total: float, count: int, device) -> float:
    tensor = torch.tensor([float(total), float(count)], device=device)
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor[0] / torch.clamp(tensor[1], min=1.0)).detach().cpu())


def _write_split_manifest(output_dir: Path, train_ds, val_ds) -> None:
    def paths(ds) -> list[str]:
        if hasattr(ds, "files"):
            return [str(p) for p in ds.files]
        if hasattr(ds, "dataset") and hasattr(ds.dataset, "files") and hasattr(ds, "indices"):
            dataset = ds.dataset
            stride = int(getattr(dataset, "cascade_blocks", 1)) * int(getattr(dataset, "patches_per_file", 1))
            file_indices = sorted({int(i) // max(1, stride) for i in ds.indices})
            return [str(dataset.files[i]) for i in file_indices if i < len(dataset.files)]
        return []

    train_files = paths(train_ds)
    val_files = paths(val_ds)
    manifest = {
        "train_count": len(train_files),
        "val_count": len(val_files),
        "train_dataset_len": len(train_ds),
        "val_dataset_len": len(val_ds),
        "train_files": train_files,
        "val_files": val_files,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _make_datasets(config: dict[str, Any]):
    dataset_cfg = config.get("dataset", {})
    if dataset_cfg.get("splits"):
        train_ds = B13CascadeDataset(config, split="train")
        val_ds = B13CascadeDataset(config, split="val")
    else:
        if dataset_cfg.get("require_explicit_splits"):
            raise ValueError("dataset.require_explicit_splits=true but dataset.splits is missing")
        dataset = B13CascadeDataset(config, split="train")
        val_size = max(1, len(dataset) // 5) if len(dataset) > 1 else 1
        train_size = max(1, len(dataset) - val_size)
        if len(dataset) > 1:
            generator = torch.Generator().manual_seed(int(config.get("training", {}).get("seed", 42)))
            train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
        else:
            train_ds = dataset
            val_ds = dataset
    return train_ds, val_ds


def _set_dataset_epoch(dataset, epoch: int) -> None:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    elif hasattr(dataset, "dataset"):
        _set_dataset_epoch(dataset.dataset, epoch)


def _weighted_sample_mean(values, active, sample_weight=None):
    values = values * active
    if sample_weight is not None:
        weights = sample_weight.to(device=values.device, dtype=values.dtype).view(-1) * active
        return (values * weights).sum() / torch.clamp(active.sum(), min=1.0)
    return values.sum() / torch.clamp(active.sum(), min=1.0)


def _masked_elementwise_mean(element_loss, valid_mask, sample_weight=None):
    if sample_weight is None:
        loss = element_loss * valid_mask
        return loss.sum() / torch.clamp(valid_mask.sum(), min=1.0)
    dims = tuple(range(1, element_loss.ndim))
    masked_sum = (element_loss * valid_mask).sum(dim=dims)
    valid_sum = valid_mask.sum(dim=dims)
    per_sample = masked_sum / torch.clamp(valid_sum, min=1.0)
    active = (valid_sum > 0).to(element_loss.dtype)
    return _weighted_sample_mean(per_sample, active, sample_weight=sample_weight)


def _block_loss_weights(cascade_block, weights: list[float] | None, device):
    if not weights:
        return None
    blocks = cascade_block.to(device=device, dtype=torch.long).view(-1)
    table = torch.tensor([float(x) for x in weights], device=device, dtype=torch.float32)
    blocks = torch.clamp(blocks, min=0, max=max(0, table.numel() - 1))
    return table[blocks]


def b13_reconstruction_loss(pred, target, valid_mask, mode: str = "smooth_l1", sample_weight=None):
    if mode == "mse":
        loss = (pred - target) ** 2
    elif mode == "l1":
        loss = torch.abs(pred - target)
    else:
        loss = F.smooth_l1_loss(pred, target, reduction="none")
    return _masked_elementwise_mean(loss, valid_mask, sample_weight=sample_weight)


def b13_mean_bias_loss(pred, target, valid_mask, mode: str = "mse", sample_weight=None):
    if sample_weight is not None:
        dims = tuple(range(1, pred.ndim))
        valid_sum = valid_mask.sum(dim=dims)
        per_sample_bias = ((pred - target) * valid_mask).sum(dim=dims) / torch.clamp(valid_sum, min=1.0)
        active = (valid_sum > 0).to(pred.dtype)
        if mode in {"l1", "mae", "abs"}:
            per_sample_loss = torch.abs(per_sample_bias)
        else:
            per_sample_loss = per_sample_bias ** 2
        loss = _weighted_sample_mean(per_sample_loss, active, sample_weight=sample_weight)
        bias_metric = (per_sample_bias * active).sum() / torch.clamp(active.sum(), min=1.0)
        return loss, bias_metric
    bias = ((pred - target) * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1.0)
    if mode in {"l1", "mae", "abs"}:
        return torch.abs(bias), bias
    return bias ** 2, bias


def b13_lead_bias_loss(pred, target, valid_mask, mode: str = "mse", sample_weight=None):
    dims = (-2, -1)
    valid_sum = valid_mask.sum(dim=dims)
    per_lead_bias = ((pred - target) * valid_mask).sum(dim=dims) / torch.clamp(valid_sum, min=1.0)
    active = (valid_sum > 0).to(pred.dtype)
    if mode in {"l1", "mae", "abs"}:
        per_lead_loss = torch.abs(per_lead_bias)
    else:
        per_lead_loss = per_lead_bias ** 2
    if sample_weight is not None:
        weights = sample_weight.to(device=pred.device, dtype=pred.dtype).view(-1, 1, 1) * active
        loss = (per_lead_loss * weights).sum() / torch.clamp(weights.sum(), min=1.0)
    else:
        loss = (per_lead_loss * active).sum() / torch.clamp(active.sum(), min=1.0)
    bias_metric = (per_lead_bias * active).sum() / torch.clamp(active.sum(), min=1.0)
    return loss, bias_metric


def b13_gradient_reconstruction_loss(pred, target, valid_mask, mode: str = "smooth_l1", sample_weight=None):
    dx_pred = pred[..., 1:] - pred[..., :-1]
    dx_target = target[..., 1:] - target[..., :-1]
    dx_mask = valid_mask[..., 1:] * valid_mask[..., :-1]

    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_target = target[..., 1:, :] - target[..., :-1, :]
    dy_mask = valid_mask[..., 1:, :] * valid_mask[..., :-1, :]

    if mode == "mse":
        dx_loss = (dx_pred - dx_target) ** 2
        dy_loss = (dy_pred - dy_target) ** 2
    elif mode == "l1":
        dx_loss = torch.abs(dx_pred - dx_target)
        dy_loss = torch.abs(dy_pred - dy_target)
    else:
        dx_loss = F.smooth_l1_loss(dx_pred, dx_target, reduction="none")
        dy_loss = F.smooth_l1_loss(dy_pred, dy_target, reduction="none")

    if sample_weight is None:
        numerator = (dx_loss * dx_mask).sum() + (dy_loss * dy_mask).sum()
        denominator = dx_mask.sum() + dy_mask.sum()
        return numerator / torch.clamp(denominator, min=1.0)

    dims = tuple(range(1, pred.ndim))
    dx_sum = (dx_loss * dx_mask).sum(dim=dims)
    dy_sum = (dy_loss * dy_mask).sum(dim=dims)
    valid_sum = dx_mask.sum(dim=dims) + dy_mask.sum(dim=dims)
    per_sample = (dx_sum + dy_sum) / torch.clamp(valid_sum, min=1.0)
    active = (valid_sum > 0).to(pred.dtype)
    return _weighted_sample_mean(per_sample, active, sample_weight=sample_weight)


def b13_boundary_consistency_loss(
    pred,
    target,
    history_last,
    boundary_prev_target,
    boundary_mask,
    mode: str = "smooth_l1",
    sample_weight=None,
):
    pred_delta = pred[:, :, 0:1] - history_last
    target_delta = target[:, :, 0:1] - boundary_prev_target
    if mode == "mse":
        loss = (pred_delta - target_delta) ** 2
    elif mode == "l1":
        loss = torch.abs(pred_delta - target_delta)
    else:
        loss = F.smooth_l1_loss(pred_delta, target_delta, reduction="none")
    return _masked_elementwise_mean(loss, boundary_mask, sample_weight=sample_weight)


def b13_temporal_consistency_losses(
    pred,
    target,
    history_last,
    boundary_prev_target,
    valid_mask,
    boundary_mask,
    mode: str = "smooth_l1",
    sample_weight=None,
):
    pred_sequence = torch.cat([history_last, pred], dim=2)
    target_sequence = torch.cat([boundary_prev_target, target], dim=2)
    first_mask = boundary_mask
    pair_mask = valid_mask[:, :, 1:] * valid_mask[:, :, :-1]
    delta_mask = torch.cat([first_mask, pair_mask], dim=2)
    pred_delta = pred_sequence[:, :, 1:] - pred_sequence[:, :, :-1]
    target_delta = target_sequence[:, :, 1:] - target_sequence[:, :, :-1]

    if mode == "mse":
        delta_loss = (pred_delta - target_delta) ** 2
    elif mode == "l1":
        delta_loss = torch.abs(pred_delta - target_delta)
    else:
        delta_loss = F.smooth_l1_loss(pred_delta, target_delta, reduction="none")
    temporal_gradient = _masked_elementwise_mean(delta_loss, delta_mask, sample_weight=sample_weight)

    pred_accel = pred_delta[:, :, 1:] - pred_delta[:, :, :-1]
    target_accel = target_delta[:, :, 1:] - target_delta[:, :, :-1]
    accel_mask = delta_mask[:, :, 1:] * delta_mask[:, :, :-1]
    if mode == "mse":
        accel_loss = (pred_accel - target_accel) ** 2
    elif mode == "l1":
        accel_loss = torch.abs(pred_accel - target_accel)
    else:
        accel_loss = F.smooth_l1_loss(pred_accel, target_accel, reduction="none")
    temporal_acceleration = _masked_elementwise_mean(accel_loss, accel_mask, sample_weight=sample_weight)
    return temporal_gradient, temporal_acceleration


def _weighted_active_mean(values, active, sample_weight=None):
    values = values * active
    if sample_weight is not None:
        view_shape = [values.shape[0]] + [1] * (values.ndim - 1)
        weights = sample_weight.to(device=values.device, dtype=values.dtype).view(*view_shape) * active
        return (values * weights).sum() / torch.clamp(weights.sum(), min=1.0)
    return values.sum() / torch.clamp(active.sum(), min=1.0)


def b13_cold_cloud_losses(
    pred,
    target,
    valid_mask,
    thresholds_norm: list[float],
    softness_norm: float,
    threshold_weights=None,
    sample_weight=None,
):
    if not thresholds_norm:
        zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
        return zero, zero, zero
    softness = max(float(softness_norm), 1.0e-4)
    if threshold_weights is None or len(threshold_weights) == 0:
        threshold_weights = [1.0] * len(thresholds_norm)
    else:
        threshold_weights = [float(w) for w in threshold_weights]
        if len(threshold_weights) < len(thresholds_norm):
            threshold_weights = threshold_weights + [threshold_weights[-1]] * (
                len(thresholds_norm) - len(threshold_weights)
            )
        threshold_weights = threshold_weights[: len(thresholds_norm)]
    loss_weights = torch.as_tensor(threshold_weights, device=pred.device, dtype=pred.dtype)
    if torch.sum(loss_weights) <= 0:
        loss_weights = torch.ones_like(loss_weights)
    loss_weights = loss_weights / torch.clamp(loss_weights.sum(), min=1.0e-6)
    area_losses = []
    dice_losses = []
    retention_losses = []
    spatial_dims = (-2, -1)
    for threshold in thresholds_norm:
        thr = torch.as_tensor(float(threshold), device=pred.device, dtype=pred.dtype)
        pred_cloud = torch.sigmoid((thr - pred) / softness)
        target_cloud = torch.sigmoid((thr - target) / softness)
        valid_sum = valid_mask.sum(dim=spatial_dims)
        active = (valid_sum > 0).to(pred.dtype)

        pred_area = (pred_cloud * valid_mask).sum(dim=spatial_dims) / torch.clamp(valid_sum, min=1.0)
        target_area = (target_cloud * valid_mask).sum(dim=spatial_dims) / torch.clamp(valid_sum, min=1.0)
        area_losses.append(_weighted_active_mean((pred_area - target_area) ** 2, active, sample_weight=sample_weight))

        intersection = (pred_cloud * target_cloud * valid_mask).sum(dim=spatial_dims)
        pred_mass = (pred_cloud * valid_mask).sum(dim=spatial_dims)
        target_mass = (target_cloud * valid_mask).sum(dim=spatial_dims)
        target_active = (target_mass > 1.0).to(pred.dtype)
        dice = 1.0 - (2.0 * intersection + 1.0e-6) / torch.clamp(pred_mass + target_mass + 1.0e-6, min=1.0e-6)
        dice_losses.append(_weighted_active_mean(dice, target_active, sample_weight=sample_weight))

        retention_den = torch.clamp(target_mass, min=1.0)
        retention = (torch.relu(target_cloud - pred_cloud) * target_cloud * valid_mask).sum(dim=spatial_dims) / retention_den
        retention_losses.append(_weighted_active_mean(retention, target_active, sample_weight=sample_weight))
    return (
        (torch.stack(area_losses) * loss_weights).sum(),
        (torch.stack(dice_losses) * loss_weights).sum(),
        (torch.stack(retention_losses) * loss_weights).sum(),
    )


def b13_cold_cloud_false_alarm_loss(
    pred,
    target,
    valid_mask,
    thresholds_norm: list[float],
    softness_norm: float,
    threshold_weights=None,
    sample_weight=None,
):
    if not thresholds_norm:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    softness = max(float(softness_norm), 1.0e-4)
    weights = threshold_weights or [1.0] * len(thresholds_norm)
    weights = list(weights)[: len(thresholds_norm)]
    if len(weights) < len(thresholds_norm):
        weights.extend([weights[-1] if weights else 1.0] * (len(thresholds_norm) - len(weights)))
    weight_tensor = torch.as_tensor(weights, device=pred.device, dtype=pred.dtype)
    weight_tensor = weight_tensor / torch.clamp(weight_tensor.sum(), min=1.0e-6)
    losses = []
    spatial_dims = (-2, -1)
    for threshold in thresholds_norm:
        thr = torch.as_tensor(float(threshold), device=pred.device, dtype=pred.dtype)
        pred_cloud = torch.sigmoid((thr - pred) / softness)
        target_cloud = torch.sigmoid((thr - target) / softness)
        false_mass = (pred_cloud * (1.0 - target_cloud) * valid_mask).sum(dim=spatial_dims)
        valid_sum = valid_mask.sum(dim=spatial_dims)
        active = (valid_sum > 0).to(pred.dtype)
        losses.append(
            _weighted_active_mean(false_mass / torch.clamp(valid_sum, min=1.0), active, sample_weight=sample_weight)
        )
    return (torch.stack(losses) * weight_tensor).sum()


def _apply_west_edge_weight(mask, batch: dict[str, Any]):
    weight = batch.get("west_edge_weight_mask")
    if weight is None:
        return mask
    weight = weight.to(device=mask.device, dtype=mask.dtype)
    if weight.ndim == mask.ndim and weight.shape[2] != mask.shape[2]:
        weight = weight[:, :, : mask.shape[2]]
    return mask * weight


def _normalize_bt13_tensor(values, clip_min: float, clip_max: float):
    return torch.clamp((values - clip_min) / max(clip_max - clip_min, 1.0e-6), 0.0, 1.0)


def _apply_future_hour_multiplier(
    mask,
    future_hours,
    start_hour: int | None,
    multiplier: float,
):
    if start_hour is None or float(multiplier) == 1.0:
        return mask
    hours = future_hours.to(device=mask.device, dtype=mask.dtype)
    if hours.ndim == 1:
        hours = hours.unsqueeze(0)
    lead_weight = torch.ones_like(hours, dtype=mask.dtype, device=mask.device)
    late = hours >= float(start_hour)
    if torch.any(late):
        lead_weight = torch.where(
            late,
            torch.full_like(lead_weight, float(multiplier)),
            lead_weight,
        )
    lead_weight = lead_weight[:, None, :, None, None]
    if lead_weight.shape[2] != mask.shape[2]:
        lead_weight = lead_weight[:, :, : mask.shape[2]]
    return mask * lead_weight


def b13_cold_core_reconstruction_loss(
    pred,
    target,
    target_k,
    valid_mask,
    *,
    primary_threshold_k: float,
    fallback_threshold_k: float | None = None,
    min_pixels: int = 1,
    mode: str = "smooth_l1",
    sample_weight=None,
):
    if mode == "mse":
        element_loss = (pred - target) ** 2
    elif mode == "l1":
        element_loss = torch.abs(pred - target)
    else:
        element_loss = F.smooth_l1_loss(pred, target, reduction="none")

    valid_bool = valid_mask > 0
    core_primary = valid_bool & (target_k <= float(primary_threshold_k))
    core_mask = core_primary
    if fallback_threshold_k is not None and float(fallback_threshold_k) > float(primary_threshold_k):
        fallback_mask = valid_bool & (target_k <= float(fallback_threshold_k))
        primary_pixels = core_primary.sum(dim=(-2, -1), keepdim=True)
        use_fallback = primary_pixels < int(max(1, min_pixels))
        core_mask = torch.where(use_fallback, fallback_mask, core_primary)
    return _masked_elementwise_mean(element_loss, core_mask.to(valid_mask.dtype), sample_weight=sample_weight)


def train_from_config(config: dict[str, Any]) -> Path:
    _require_torch()
    predicted_history_cfg = config.get("predicted_history", {})
    if bool(predicted_history_cfg.get("online_rollout_enabled", False)):
        raise NotImplementedError(
            "Formal online-rollout training is intentionally gated; run scripts/smoke_b13_online_no_grad_rollout.py first"
        )
    ctx = distributed_context()
    is_main = ctx["is_main"]
    device = ctx["device"]
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    output_dir = Path(config.get("output", {}).get("output_dir", "outputs/b13cascade_debug"))
    ckpt_dir = output_dir / "checkpoints"

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "config_used.yaml")
        save_config(config, output_dir / "resolved_config.yaml")

    train_ds, val_ds = _make_datasets(config)
    if is_main:
        _write_split_manifest(output_dir, train_ds, val_ds)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ctx["distributed"] else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ctx["distributed"] else None
    loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, sampler=val_sampler, num_workers=0)
    first_batch = next(iter(loader))
    if is_main:
        print("First B13 cascade batch shapes:")
        for key in ["condition", "wrf_cloud_condition", "aerosol_3d_condition", "target", "target_valid_mask", "valid_mask", "loss_mask", "target_k"]:
            if key not in first_batch:
                continue
            print(f"  {key}: {tuple(first_batch[key].shape)}")
        print(f"  cascade_block: {first_batch['cascade_block']}")
        print(f"  future_hours: {first_batch['future_hours']}")
        print(f"  path: {first_batch['path']}")
        print(f"DDP: distributed={ctx['distributed']} world_size={ctx['world_size']} device={device}")

    condition_channels = int(first_batch["condition"].shape[1])
    wrf_cloud_channels = int(first_batch["wrf_cloud_condition"].shape[1]) if "wrf_cloud_condition" in first_batch else 0
    aerosol_3d_shape = (
        list(first_batch["aerosol_3d_condition"].shape[1:])
        if "aerosol_3d_condition" in first_batch
        else None
    )
    target_channels = int(first_batch["target"].shape[1])
    model = build_b13_model(model_cfg, target_channels, condition_channels, wrf_cloud_channels).to(device)
    diffusion_steps = int(model_cfg.get("diffusion_steps", 100))
    if ctx["distributed"]:
        ddp_kwargs = {"device_ids": [ctx["local_rank"]]} if torch.cuda.is_available() else {}
        model = DistributedDataParallel(model, **ddp_kwargs)
        model.module.diffusion_steps = diffusion_steps

    scheduler = DDPMScheduler(steps=diffusion_steps, device=str(device))
    learning_rate = float(train_cfg.get("lr", 1.0e-4))
    optimizer_weight_decay = float(train_cfg.get("weight_decay", 0.01))
    optimizer_betas = tuple(float(value) for value in train_cfg.get("adam_betas", [0.9, 0.999]))
    if len(optimizer_betas) != 2:
        raise ValueError("training.adam_betas must contain exactly two values")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=optimizer_betas,
        weight_decay=optimizer_weight_decay,
    )
    use_amp = bool(train_cfg.get("amp", True)) and torch.cuda.is_available()
    amp_dtype_name = str(train_cfg.get("amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    epochs = int(train_cfg.get("epochs", 2))
    stop_after_epoch = int(train_cfg.get("stop_after_epoch", epochs) or epochs)
    stop_after_epoch = min(stop_after_epoch, epochs)
    warmup_epochs = int(train_cfg.get("lr_warmup_epochs", 0))
    min_lr = float(train_cfg.get("min_lr", 0.0))

    def lr_factor(epoch_index: int) -> float:
        current = epoch_index + 1
        if warmup_epochs > 0 and current <= warmup_epochs:
            return current / float(warmup_epochs)
        remaining = max(1, epochs - warmup_epochs)
        progress = min(max((current - warmup_epochs) / float(remaining), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = min_lr / max(learning_rate, 1.0e-12)
        return min_factor + (1.0 - min_factor) * cosine

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)

    lambda_b13_recon = float(train_cfg.get("lambda_b13_recon", 0.0))
    b13_recon_loss_mode = str(train_cfg.get("b13_recon_loss_mode", "smooth_l1"))
    recon_loss_start_epoch = int(train_cfg.get("recon_loss_start_epoch", 1))
    recon_loss_warmup_epochs = int(train_cfg.get("recon_loss_warmup_epochs", 0))
    recon_loss_max_t_fraction = train_cfg.get("recon_loss_max_t_fraction")
    recon_loss_max_t_fraction = None if recon_loss_max_t_fraction is None else float(recon_loss_max_t_fraction)
    lambda_b13_bias = float(train_cfg.get("lambda_b13_bias", 0.0))
    b13_bias_loss_mode = str(train_cfg.get("b13_bias_loss_mode", "mse"))
    bias_loss_start_epoch = int(train_cfg.get("bias_loss_start_epoch", recon_loss_start_epoch))
    bias_loss_warmup_epochs = int(train_cfg.get("bias_loss_warmup_epochs", recon_loss_warmup_epochs))
    bias_loss_max_t_fraction = train_cfg.get("bias_loss_max_t_fraction", recon_loss_max_t_fraction)
    bias_loss_max_t_fraction = None if bias_loss_max_t_fraction is None else float(bias_loss_max_t_fraction)
    cascade_block_loss_weights = train_cfg.get("cascade_block_loss_weights")
    if cascade_block_loss_weights is not None:
        cascade_block_loss_weights = [float(x) for x in cascade_block_loss_weights]
    lambda_daylight_bias = float(train_cfg.get("lambda_daylight_bias", 0.0))
    daylight_bias_loss_mode = str(train_cfg.get("daylight_bias_loss_mode", b13_bias_loss_mode))
    daylight_bias_start_epoch = int(train_cfg.get("daylight_bias_start_epoch", bias_loss_start_epoch))
    daylight_bias_warmup_epochs = int(train_cfg.get("daylight_bias_warmup_epochs", bias_loss_warmup_epochs))
    daylight_bias_max_t_fraction = train_cfg.get("daylight_bias_max_t_fraction", bias_loss_max_t_fraction)
    daylight_bias_max_t_fraction = None if daylight_bias_max_t_fraction is None else float(daylight_bias_max_t_fraction)
    lambda_b13_gradient_recon = float(train_cfg.get("lambda_b13_gradient_recon", 0.0))
    b13_gradient_recon_loss_mode = str(train_cfg.get("b13_gradient_recon_loss_mode", b13_recon_loss_mode))
    gradient_recon_start_epoch = int(train_cfg.get("gradient_recon_start_epoch", recon_loss_start_epoch))
    gradient_recon_warmup_epochs = int(train_cfg.get("gradient_recon_warmup_epochs", recon_loss_warmup_epochs))
    gradient_recon_max_t_fraction = train_cfg.get("gradient_recon_max_t_fraction", recon_loss_max_t_fraction)
    gradient_recon_max_t_fraction = None if gradient_recon_max_t_fraction is None else float(gradient_recon_max_t_fraction)
    lambda_b13_lead_bias = float(train_cfg.get("lambda_b13_lead_bias", 0.0))
    b13_lead_bias_loss_mode = str(train_cfg.get("b13_lead_bias_loss_mode", b13_bias_loss_mode))
    lead_bias_start_epoch = int(train_cfg.get("lead_bias_start_epoch", bias_loss_start_epoch))
    lead_bias_warmup_epochs = int(train_cfg.get("lead_bias_warmup_epochs", bias_loss_warmup_epochs))
    lead_bias_max_t_fraction = train_cfg.get("lead_bias_max_t_fraction", bias_loss_max_t_fraction)
    lead_bias_max_t_fraction = None if lead_bias_max_t_fraction is None else float(lead_bias_max_t_fraction)
    lambda_daylight_lead_bias = float(train_cfg.get("lambda_daylight_lead_bias", 0.0))
    daylight_lead_bias_loss_mode = str(train_cfg.get("daylight_lead_bias_loss_mode", b13_lead_bias_loss_mode))
    daylight_lead_bias_start_epoch = int(train_cfg.get("daylight_lead_bias_start_epoch", lead_bias_start_epoch))
    daylight_lead_bias_warmup_epochs = int(train_cfg.get("daylight_lead_bias_warmup_epochs", lead_bias_warmup_epochs))
    daylight_lead_bias_max_t_fraction = train_cfg.get("daylight_lead_bias_max_t_fraction", daylight_bias_max_t_fraction)
    daylight_lead_bias_max_t_fraction = None if daylight_lead_bias_max_t_fraction is None else float(daylight_lead_bias_max_t_fraction)
    lambda_boundary_consistency = float(train_cfg.get("lambda_boundary_consistency", 0.0))
    boundary_consistency_loss_mode = str(train_cfg.get("boundary_consistency_loss_mode", b13_recon_loss_mode))
    boundary_consistency_start_epoch = int(train_cfg.get("boundary_consistency_start_epoch", recon_loss_start_epoch))
    boundary_consistency_warmup_epochs = int(train_cfg.get("boundary_consistency_warmup_epochs", recon_loss_warmup_epochs))
    boundary_consistency_max_t_fraction = train_cfg.get("boundary_consistency_max_t_fraction", recon_loss_max_t_fraction)
    boundary_consistency_max_t_fraction = None if boundary_consistency_max_t_fraction is None else float(boundary_consistency_max_t_fraction)
    lambda_b13_temporal_gradient = float(train_cfg.get("lambda_b13_temporal_gradient", 0.0))
    lambda_b13_temporal_acceleration = float(train_cfg.get("lambda_b13_temporal_acceleration", 0.0))
    temporal_consistency_loss_mode = str(train_cfg.get("temporal_consistency_loss_mode", boundary_consistency_loss_mode))
    temporal_consistency_start_epoch = int(train_cfg.get("temporal_consistency_start_epoch", boundary_consistency_start_epoch))
    temporal_consistency_warmup_epochs = int(
        train_cfg.get("temporal_consistency_warmup_epochs", boundary_consistency_warmup_epochs)
    )
    temporal_consistency_max_t_fraction = train_cfg.get(
        "temporal_consistency_max_t_fraction", boundary_consistency_max_t_fraction
    )
    temporal_consistency_max_t_fraction = (
        None if temporal_consistency_max_t_fraction is None else float(temporal_consistency_max_t_fraction)
    )
    b13_clip = config.get("dataset", {}).get("b13_clip", [180.0, 330.0])
    b13_clip_min = float(b13_clip[0])
    b13_clip_max = float(b13_clip[1])
    b13_scale_k = float(b13_clip[1]) - float(b13_clip[0])
    cold_cloud_thresholds_k = [float(x) for x in train_cfg.get("cold_cloud_thresholds_k", [])]
    cold_cloud_thresholds_norm = [
        max(0.0, min(1.0, (threshold - b13_clip_min) / max(b13_scale_k, 1.0e-6)))
        for threshold in cold_cloud_thresholds_k
    ]
    cold_cloud_threshold_weights = train_cfg.get("cold_cloud_threshold_weights")
    if cold_cloud_threshold_weights is None:
        cold_cloud_threshold_weights = [1.0] * len(cold_cloud_thresholds_norm)
    else:
        cold_cloud_threshold_weights = [float(x) for x in cold_cloud_threshold_weights]
    cold_cloud_softness_norm = float(train_cfg.get("cold_cloud_softness_k", 5.0)) / max(b13_scale_k, 1.0e-6)
    lambda_cold_cloud_area = float(train_cfg.get("lambda_cold_cloud_area", 0.0))
    lambda_cold_cloud_dice = float(train_cfg.get("lambda_cold_cloud_dice", 0.0))
    lambda_cold_cloud_retention = float(train_cfg.get("lambda_cold_cloud_retention", 0.0))
    lambda_cold_cloud_false_alarm = float(train_cfg.get("lambda_cold_cloud_false_alarm", 0.0))
    cold_cloud_start_epoch = int(train_cfg.get("cold_cloud_start_epoch", recon_loss_start_epoch))
    cold_cloud_warmup_epochs = int(train_cfg.get("cold_cloud_warmup_epochs", recon_loss_warmup_epochs))
    cold_cloud_max_t_fraction = train_cfg.get("cold_cloud_max_t_fraction", recon_loss_max_t_fraction)
    cold_cloud_max_t_fraction = None if cold_cloud_max_t_fraction is None else float(cold_cloud_max_t_fraction)
    lambda_b13_cold_core_recon = float(train_cfg.get("lambda_b13_cold_core_recon", 0.0))
    b13_cold_core_recon_loss_mode = str(train_cfg.get("b13_cold_core_recon_loss_mode", b13_recon_loss_mode))
    cold_core_primary_threshold_k = train_cfg.get("cold_core_primary_threshold_k")
    cold_core_primary_threshold_k = (
        None if cold_core_primary_threshold_k is None else float(cold_core_primary_threshold_k)
    )
    cold_core_fallback_threshold_k = train_cfg.get("cold_core_fallback_threshold_k")
    cold_core_fallback_threshold_k = (
        None if cold_core_fallback_threshold_k is None else float(cold_core_fallback_threshold_k)
    )
    cold_core_min_pixels = int(train_cfg.get("cold_core_min_pixels", 1))
    cold_core_late_lead_start_hour = train_cfg.get("cold_core_late_lead_start_hour")
    cold_core_late_lead_start_hour = (
        None if cold_core_late_lead_start_hour is None else int(cold_core_late_lead_start_hour)
    )
    cold_core_late_lead_multiplier = float(train_cfg.get("cold_core_late_lead_multiplier", 1.0))
    cold_core_start_epoch = int(train_cfg.get("cold_core_start_epoch", cold_cloud_start_epoch))
    cold_core_warmup_epochs = int(train_cfg.get("cold_core_warmup_epochs", cold_cloud_warmup_epochs))
    cold_core_max_t_fraction = train_cfg.get("cold_core_max_t_fraction", recon_loss_max_t_fraction)
    cold_core_max_t_fraction = None if cold_core_max_t_fraction is None else float(cold_core_max_t_fraction)
    if lambda_b13_cold_core_recon > 0.0 and cold_core_primary_threshold_k is None:
        raise ValueError("training.cold_core_primary_threshold_k must be set when lambda_b13_cold_core_recon > 0")

    validation_seed = int(train_cfg.get("validation_seed", train_cfg.get("seed", config.get("training", {}).get("seed", 42))))
    validation_timestep_mode = str(train_cfg.get("validation_timestep_mode", "stratified"))
    validation_timestep_bins = int(train_cfg.get("validation_timestep_bins", 10))
    validation_noise_mode = str(train_cfg.get("validation_noise_mode", "fixed"))
    use_ema = bool(train_cfg.get("ema", False))
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    ema_validate = bool(train_cfg.get("ema_validate", use_ema))
    ema = ExponentialMovingAverage(_unwrap_model(model), decay=ema_decay) if use_ema else None
    cfg_cfg = copy.deepcopy(config.get("cfg", {}))
    diffusion_cfg = copy.deepcopy(config.get("diffusion", {}))
    max_train_batches = train_cfg.get("max_train_batches")
    max_val_batches = train_cfg.get("max_val_batches")
    checkpoint_every = int(train_cfg.get("checkpoint_every", 0) or 0)
    checkpoint_epochs = {int(x) for x in train_cfg.get("checkpoint_epochs", [])}
    rollout_validation_every = int(train_cfg.get("rollout_validation_every", 0) or 0)
    rollout_validation_max_files = int(train_cfg.get("rollout_validation_max_files", 8))
    rollout_validation_sampling_steps = int(train_cfg.get("rollout_validation_sampling_steps", 25))
    rollout_validation_metrics_min_lon = train_cfg.get("rollout_validation_metrics_min_lon", 80.0)
    rollout_validation_metrics_min_lon = (
        None if rollout_validation_metrics_min_lon is None else float(rollout_validation_metrics_min_lon)
    )
    init_checkpoint = train_cfg.get("init_checkpoint")
    init_checkpoint_weights = str(train_cfg.get("init_checkpoint_weights", "auto"))
    init_checkpoint_load_ema = bool(train_cfg.get("init_checkpoint_load_ema", use_ema))
    l2sp_lambda = float(train_cfg.get("l2sp_lambda", 0.0))
    l2sp_reference_checkpoint = train_cfg.get("l2sp_reference_checkpoint", init_checkpoint)
    l2sp_reference_weights = str(train_cfg.get("l2sp_reference_weights", init_checkpoint_weights))
    rollout_guard_enabled = bool(train_cfg.get("rollout_guard_enabled", False))
    rollout_guard_start_epoch = max(1, int(train_cfg.get("rollout_guard_start_epoch", 1)))
    rollout_best_start_epoch = max(1, int(train_cfg.get("rollout_best_start_epoch", 1)))
    rollout_guard_action = str(train_cfg.get("rollout_guard_action", "stop")).strip().lower()
    # Validate early, even if the guard is disabled, so a typo cannot silently
    # change the behavior when it is enabled in a resumed phase.
    rollout_guard_should_stop(rollout_guard_action, 0, 1)
    rollout_guard_max_abs_bias_k = float(train_cfg.get("rollout_guard_max_abs_bias_k", float("inf")))
    rollout_guard_max_rmse_delta_k = float(train_cfg.get("rollout_guard_max_rmse_delta_k", float("inf")))
    rollout_guard_max_best_rmse_delta_k = float(
        train_cfg.get("rollout_guard_max_best_rmse_delta_k", float("inf"))
    )
    rollout_best_max_abs_bias_k = float(
        train_cfg.get("rollout_best_max_abs_bias_k", float("inf"))
    )
    rollout_best_min_pearson_r = float(
        train_cfg.get("rollout_best_min_pearson_r", -float("inf"))
    )
    rollout_best_max_late_cold_core_rmse_k = float(
        train_cfg.get("rollout_best_max_late_cold_core_rmse_k", float("inf"))
    )
    rollout_guard_patience = max(1, int(train_cfg.get("rollout_guard_patience", 1)))
    rollout_early_stopping_enabled = bool(
        train_cfg.get("rollout_early_stopping_enabled", False)
    )
    rollout_early_stopping_start_epoch = max(
        1, int(train_cfg.get("rollout_early_stopping_start_epoch", rollout_best_start_epoch))
    )
    rollout_early_stopping_patience = max(
        1, int(train_cfg.get("rollout_early_stopping_patience", 3))
    )
    rollout_early_stopping_min_delta_k = max(
        0.0, float(train_cfg.get("rollout_early_stopping_min_delta_k", 0.0))
    )
    if rollout_early_stopping_enabled and rollout_validation_every <= 0:
        raise ValueError(
            "training.rollout_early_stopping_enabled requires rollout_validation_every > 0"
        )
    rollout_baseline_before_training = bool(train_cfg.get("rollout_baseline_before_training", rollout_guard_enabled))
    resume = train_cfg.get("resume")
    start_epoch = 1
    global_step = 0
    best_val = float("inf")
    best_rollout_rmse = float(train_cfg.get("rollout_initial_best_rmse_k", float("inf")))
    rollout_baseline_rmse_k = float(train_cfg.get("rollout_reference_rmse_k", float("nan")))
    rollout_guard_bad_count = 0
    rollout_early_stop_best_rmse = best_rollout_rmse
    rollout_early_stop_bad_count = 0
    init_load_info = None

    if resume and init_checkpoint and is_main:
        print(f"Resume requested; ignoring training.init_checkpoint={init_checkpoint}")
    if init_checkpoint and not resume:
        # Warm-start needs weights only. Loading the optimizer payload directly
        # onto a GPU wastes several GB for full training checkpoints.
        ckpt = torch.load(init_checkpoint, map_location="cpu")
        model_to_load = model.module if hasattr(model, "module") else model
        init_load_info = load_checkpoint_weights(
            model_to_load,
            ckpt,
            weights=init_checkpoint_weights,
            strict=True,
            allow_fallback=init_checkpoint_weights == "auto",
        )
        if ema is not None and init_checkpoint_load_ema:
            if "ema_state" in ckpt:
                ema.load_state_dict(ckpt["ema_state"], device=device)
            elif "ema_model_state" in ckpt:
                ema.load_state_dict({"shadow": ckpt["ema_model_state"]}, device=device)
        if is_main:
            print(
                "Warm-started model from "
                f"{init_checkpoint} (requested={init_checkpoint_weights}, loaded={init_load_info['weights_loaded']})"
            )
        del ckpt

    if resume:
        ckpt = torch.load(resume, map_location=device)
        state = ckpt["model_state"]
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(ckpt.get("raw_model_state", state))
        optimizer.load_state_dict(ckpt.get("optimizer_state", optimizer.state_dict()))
        if "lr_scheduler_state" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        if ema is not None:
            if "ema_state" in ckpt:
                ema.load_state_dict(ckpt["ema_state"], device=device)
            elif "ema_model_state" in ckpt:
                ema.load_state_dict({"shadow": ckpt["ema_model_state"]}, device=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", ckpt.get("optimizer_steps", 0)))
        best_val = float(ckpt.get("best_diffusion_val_loss", ckpt.get("val_loss", best_val)))
        best_rollout_rmse = float(ckpt.get("best_rollout_rmse_k", best_rollout_rmse))
        rollout_baseline_rmse_k = float(ckpt.get("rollout_baseline_rmse_k", rollout_baseline_rmse_k))
        rollout_guard_bad_count = int(ckpt.get("rollout_guard_bad_count", 0))
        rollout_early_stop_best_rmse = float(
            ckpt.get("rollout_early_stop_best_rmse_k", rollout_early_stop_best_rmse)
        )
        rollout_early_stop_bad_count = int(ckpt.get("rollout_early_stop_bad_count", 0))

    l2sp_reference = None
    l2sp_reference_weights_loaded = None
    if l2sp_lambda > 0.0:
        if not l2sp_reference_checkpoint:
            raise ValueError("training.l2sp_reference_checkpoint is required when l2sp_lambda > 0")
        reference_checkpoint = torch.load(l2sp_reference_checkpoint, map_location="cpu")
        l2sp_reference, l2sp_reference_weights_loaded = build_l2sp_reference(
            model,
            reference_checkpoint,
            weights=l2sp_reference_weights,
        )
        del reference_checkpoint
        if is_main:
            print(
                f"L2-SP enabled: lambda={l2sp_lambda:g} reference={l2sp_reference_checkpoint} "
                f"weights={l2sp_reference_weights_loaded}"
            )

    log_path = output_dir / "train_log.csv"
    if is_main:
        if not resume or not log_path.is_file():
            with log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_train_log_fieldnames())
                writer.writeheader()
        else:
            print(f"Resume mode: appending to existing training log {log_path}")
        print(f"Trainable parameters: {_count_trainable_parameters(model):,}")
        if ema is not None:
            print(f"EMA enabled: decay={ema_decay} validate={ema_validate}")

    if rollout_baseline_before_training and not np.isfinite(rollout_baseline_rmse_k):
        baseline_values = {"rmse_k": float("nan")}
        if ctx["distributed"]:
            dist.barrier()
        if is_main:
            if ema is not None and ema_validate:
                with ema.average_parameters(_unwrap_model(model)):
                    baseline_values = rollout_validate(
                        config,
                        _unwrap_model(model),
                        scheduler,
                        device,
                        max_files=rollout_validation_max_files,
                        sampling_steps=rollout_validation_sampling_steps,
                        metrics_min_lon=rollout_validation_metrics_min_lon,
                    )
            else:
                baseline_values = rollout_validate(
                    config,
                    _unwrap_model(model),
                    scheduler,
                    device,
                    max_files=rollout_validation_max_files,
                    sampling_steps=rollout_validation_sampling_steps,
                    metrics_min_lon=rollout_validation_metrics_min_lon,
                )
            rollout_baseline_rmse_k = float(baseline_values["rmse_k"])
            (output_dir / "rollout_baseline.json").write_text(
                json.dumps(baseline_values, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Rollout guard baseline RMSE={rollout_baseline_rmse_k:.4f} K")
        if ctx["distributed"]:
            baseline_tensor = torch.tensor(
                [rollout_baseline_rmse_k if is_main else 0.0], device=device, dtype=torch.float64
            )
            dist.broadcast(baseline_tensor, src=0)
            rollout_baseline_rmse_k = float(baseline_tensor.item())
            dist.barrier()

    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"
    best_diffusion_path = ckpt_dir / "best_diffusion.pt"
    progress_log_every_steps = max(0, int(train_cfg.get("progress_log_every_steps", 0)))
    for epoch in range(start_epoch, stop_after_epoch + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        _set_dataset_epoch(train_ds, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        diffusion_loss_sum = 0.0
        recon_loss_sum = 0.0
        recon_scale_sum = 0.0
        bias_loss_sum = 0.0
        bias_scale_sum = 0.0
        bias_k_sum = 0.0
        bias_k_count = 0
        daylight_bias_loss_sum = 0.0
        daylight_bias_scale_sum = 0.0
        daylight_bias_k_sum = 0.0
        daylight_bias_k_count = 0
        gradient_loss_sum = 0.0
        gradient_scale_sum = 0.0
        lead_bias_loss_sum = 0.0
        lead_bias_scale_sum = 0.0
        lead_bias_k_sum = 0.0
        lead_bias_k_count = 0
        daylight_lead_bias_loss_sum = 0.0
        daylight_lead_bias_scale_sum = 0.0
        daylight_lead_bias_k_sum = 0.0
        daylight_lead_bias_k_count = 0
        boundary_consistency_loss_sum = 0.0
        boundary_consistency_scale_sum = 0.0
        temporal_gradient_loss_sum = 0.0
        temporal_acceleration_loss_sum = 0.0
        temporal_consistency_scale_sum = 0.0
        cold_cloud_area_loss_sum = 0.0
        cold_cloud_dice_loss_sum = 0.0
        cold_cloud_retention_loss_sum = 0.0
        cold_cloud_false_alarm_loss_sum = 0.0
        cold_cloud_scale_sum = 0.0
        cold_core_recon_loss_sum = 0.0
        cold_core_scale_sum = 0.0
        l2sp_loss_sum = 0.0
        predicted_history_sum = 0.0
        predicted_history_count = 0
        west_edge_loss_weight_sum = 0.0
        block_weight_sum = 0.0
        mean_t_sum = 0.0
        grad_norm_sum = 0.0
        optim_steps = 0
        train_steps = 0
        for step, batch in enumerate(loader, start=1):
            if max_train_batches is not None and step > int(max_train_batches):
                break
            batch = move_batch(batch, device)
            x0 = batch["target"]
            condition = batch["condition"]
            wrf_cloud_condition = batch.get("wrf_cloud_condition")
            aerosol_3d_condition = batch.get("aerosol_3d_condition")
            condition, wrf_cloud_condition, cfg_dropped = apply_cfg_condition_dropout(
                condition,
                wrf_cloud_condition,
                batch["condition_group_ids"],
                cfg_cfg,
            )
            aerosol_3d_condition = apply_cfg_aerosol_3d_dropout(
                aerosol_3d_condition, cfg_dropped, cfg_cfg
            )
            loss_mask = batch["loss_mask"]
            block_sample_weight = _block_loss_weights(batch["cascade_block"], cascade_block_loss_weights, device)
            block_weight_scalar = 1.0
            if block_sample_weight is not None:
                block_weight_scalar = float(block_sample_weight.detach().mean().cpu())
            t = torch.randint(0, diffusion_steps, (x0.shape[0],), device=device, dtype=torch.long)
            noise = make_training_noise(x0, diffusion_cfg)
            xt, noise = scheduler.q_sample(x0, t, noise=noise)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_noise = model(
                    xt,
                    condition,
                    t,
                    **(b13_model_kwargs(wrf_cloud_condition, aerosol_3d_condition) or {}),
                )
                diffusion_loss = _masked_elementwise_mean((pred_noise - noise) ** 2, loss_mask, sample_weight=block_sample_weight)
                recon_loss = torch.zeros((), device=device)
                bias_loss = torch.zeros((), device=device)
                bias_norm = torch.zeros((), device=device)
                bias_active = torch.zeros((), device=device)
                daylight_bias_loss = torch.zeros((), device=device)
                daylight_bias_norm = torch.zeros((), device=device)
                daylight_bias_active = torch.zeros((), device=device)
                gradient_loss = torch.zeros((), device=device)
                lead_bias_loss = torch.zeros((), device=device)
                lead_bias_norm = torch.zeros((), device=device)
                lead_bias_active = torch.zeros((), device=device)
                daylight_lead_bias_loss = torch.zeros((), device=device)
                daylight_lead_bias_norm = torch.zeros((), device=device)
                daylight_lead_bias_active = torch.zeros((), device=device)
                boundary_consistency_loss = torch.zeros((), device=device)
                temporal_gradient_loss = torch.zeros((), device=device)
                temporal_acceleration_loss = torch.zeros((), device=device)
                cold_cloud_area_loss = torch.zeros((), device=device)
                cold_cloud_dice_loss = torch.zeros((), device=device)
                cold_cloud_retention_loss = torch.zeros((), device=device)
                cold_cloud_false_alarm_loss = torch.zeros((), device=device)
                cold_core_recon_loss = torch.zeros((), device=device)
                l2sp_loss = torch.zeros((), device=device)
                recon_scale = _loss_scale(epoch, recon_loss_start_epoch, recon_loss_warmup_epochs)
                bias_scale = _loss_scale(epoch, bias_loss_start_epoch, bias_loss_warmup_epochs)
                daylight_bias_scale = _loss_scale(epoch, daylight_bias_start_epoch, daylight_bias_warmup_epochs)
                gradient_scale = _loss_scale(epoch, gradient_recon_start_epoch, gradient_recon_warmup_epochs)
                lead_bias_scale = _loss_scale(epoch, lead_bias_start_epoch, lead_bias_warmup_epochs)
                daylight_lead_bias_scale = _loss_scale(epoch, daylight_lead_bias_start_epoch, daylight_lead_bias_warmup_epochs)
                boundary_consistency_scale = _loss_scale(epoch, boundary_consistency_start_epoch, boundary_consistency_warmup_epochs)
                temporal_consistency_scale = _loss_scale(
                    epoch, temporal_consistency_start_epoch, temporal_consistency_warmup_epochs
                )
                cold_cloud_scale = _loss_scale(epoch, cold_cloud_start_epoch, cold_cloud_warmup_epochs)
                cold_core_scale = _loss_scale(epoch, cold_core_start_epoch, cold_core_warmup_epochs)
                loss = diffusion_loss
                pred_x0 = None
                if (
                    lambda_b13_recon > 0.0
                    or lambda_b13_bias > 0.0
                    or lambda_daylight_bias > 0.0
                    or lambda_b13_gradient_recon > 0.0
                    or lambda_b13_lead_bias > 0.0
                    or lambda_daylight_lead_bias > 0.0
                    or lambda_boundary_consistency > 0.0
                    or lambda_b13_temporal_gradient > 0.0
                    or lambda_b13_temporal_acceleration > 0.0
                    or lambda_cold_cloud_area > 0.0
                    or lambda_cold_cloud_dice > 0.0
                    or lambda_cold_cloud_retention > 0.0
                    or lambda_cold_cloud_false_alarm > 0.0
                    or lambda_b13_cold_core_recon > 0.0
                ):
                    pred_x0 = torch.clamp(scheduler.predict_x0(xt, t, pred_noise), 0.0, 1.0)
                if lambda_b13_recon > 0.0:
                    recon_mask = _apply_west_edge_weight(batch["valid_mask"], batch)
                    if recon_loss_max_t_fraction is not None:
                        t_gate = (t.float() <= float(diffusion_steps) * recon_loss_max_t_fraction).float()
                        recon_mask = recon_mask * t_gate.view(-1, 1, 1, 1, 1)
                    recon_loss = b13_reconstruction_loss(
                        pred_x0,
                        x0,
                        recon_mask,
                        mode=b13_recon_loss_mode,
                        sample_weight=block_sample_weight,
                    )
                    loss = loss + recon_scale * lambda_b13_recon * recon_loss
                if lambda_b13_bias > 0.0:
                    bias_mask = _apply_west_edge_weight(batch["valid_mask"], batch)
                    if bias_loss_max_t_fraction is not None:
                        t_gate = (t.float() <= float(diffusion_steps) * bias_loss_max_t_fraction).float()
                        bias_mask = bias_mask * t_gate.view(-1, 1, 1, 1, 1)
                    bias_loss, bias_norm = b13_mean_bias_loss(
                        pred_x0,
                        x0,
                        bias_mask,
                        mode=b13_bias_loss_mode,
                        sample_weight=block_sample_weight,
                    )
                    bias_active = (bias_mask.sum() > 0).float()
                    loss = loss + bias_scale * lambda_b13_bias * bias_loss
                if lambda_daylight_bias > 0.0:
                    daylight_mask = batch["daylight_mask_future"]
                    if daylight_mask.ndim == 4:
                        daylight_mask = daylight_mask.unsqueeze(1)
                    daylight_mask = daylight_mask.to(dtype=batch["valid_mask"].dtype)
                    daylight_bias_mask = _apply_west_edge_weight(batch["valid_mask"] * daylight_mask, batch)
                    if daylight_bias_max_t_fraction is not None:
                        t_gate = (t.float() <= float(diffusion_steps) * daylight_bias_max_t_fraction).float()
                        daylight_bias_mask = daylight_bias_mask * t_gate.view(-1, 1, 1, 1, 1)
                    daylight_bias_loss, daylight_bias_norm = b13_mean_bias_loss(
                        pred_x0,
                        x0,
                        daylight_bias_mask,
                        mode=daylight_bias_loss_mode,
                        sample_weight=block_sample_weight,
                    )
                    daylight_bias_active = (daylight_bias_mask.sum() > 0).float()
                    loss = loss + daylight_bias_scale * lambda_daylight_bias * daylight_bias_loss
                if lambda_b13_gradient_recon > 0.0:
                    gradient_mask = _apply_west_edge_weight(batch["valid_mask"], batch)
                    if gradient_recon_max_t_fraction is not None:
                        t_gate = (t.float() <= float(diffusion_steps) * gradient_recon_max_t_fraction).float()
                        gradient_mask = gradient_mask * t_gate.view(-1, 1, 1, 1, 1)
                    gradient_loss = b13_gradient_reconstruction_loss(
                        pred_x0,
                        x0,
                        gradient_mask,
                        mode=b13_gradient_recon_loss_mode,
                        sample_weight=block_sample_weight,
                    )
                    loss = loss + gradient_scale * lambda_b13_gradient_recon * gradient_loss
                if lambda_b13_lead_bias > 0.0:
                    lead_bias_mask = _apply_west_edge_weight(batch["valid_mask"], batch)
                    if lead_bias_max_t_fraction is not None:
                        t_gate = (t.float() <= float(diffusion_steps) * lead_bias_max_t_fraction).float()
                        lead_bias_mask = lead_bias_mask * t_gate.view(-1, 1, 1, 1, 1)
                    lead_bias_loss, lead_bias_norm = b13_lead_bias_loss(
                        pred_x0,
                        x0,
                        lead_bias_mask,
                        mode=b13_lead_bias_loss_mode,
                        sample_weight=block_sample_weight,
                    )
                    lead_bias_active = (lead_bias_mask.sum() > 0).float()
                    loss = loss + lead_bias_scale * lambda_b13_lead_bias * lead_bias_loss
                if lambda_daylight_lead_bias > 0.0:
                    daylight_mask = batch["daylight_mask_future"]
                    if daylight_mask.ndim == 4:
                        daylight_mask = daylight_mask.unsqueeze(1)
                    daylight_mask = daylight_mask.to(dtype=batch["valid_mask"].dtype)
                    daylight_lead_bias_mask = _apply_west_edge_weight(batch["valid_mask"] * daylight_mask, batch)
                    if daylight_lead_bias_max_t_fraction is not None:
                        t_gate = (t.float() <= float(diffusion_steps) * daylight_lead_bias_max_t_fraction).float()
                        daylight_lead_bias_mask = daylight_lead_bias_mask * t_gate.view(-1, 1, 1, 1, 1)
                    daylight_lead_bias_loss, daylight_lead_bias_norm = b13_lead_bias_loss(
                        pred_x0,
                        x0,
                        daylight_lead_bias_mask,
                        mode=daylight_lead_bias_loss_mode,
                        sample_weight=block_sample_weight,
                    )
                    daylight_lead_bias_active = (daylight_lead_bias_mask.sum() > 0).float()
                    loss = loss + daylight_lead_bias_scale * lambda_daylight_lead_bias * daylight_lead_bias_loss
                if lambda_boundary_consistency > 0.0:
                    boundary_mask = _apply_west_edge_weight(batch["boundary_valid_mask"], batch)
                    if boundary_consistency_max_t_fraction is not None:
                        t_gate = (t.float() <= float(diffusion_steps) * boundary_consistency_max_t_fraction).float()
                        boundary_mask = boundary_mask * t_gate.view(-1, 1, 1, 1, 1)
                    history_last_k = batch["bt13_history_k"][:, -1]
                    history_last_valid = torch.isfinite(history_last_k).to(dtype=boundary_mask.dtype).unsqueeze(1).unsqueeze(2)
                    boundary_mask = boundary_mask * history_last_valid
                    history_last_k = torch.nan_to_num(history_last_k, nan=b13_clip_min, posinf=b13_clip_max, neginf=b13_clip_min)
                    history_last = _normalize_bt13_tensor(history_last_k, b13_clip_min, b13_clip_max)
                    history_last = history_last.unsqueeze(1).unsqueeze(2)
                    boundary_consistency_loss = b13_boundary_consistency_loss(
                        pred_x0,
                        x0,
                        history_last,
                        batch["boundary_prev_target"],
                        boundary_mask,
                        mode=boundary_consistency_loss_mode,
                        sample_weight=block_sample_weight,
                    )
                    loss = loss + boundary_consistency_scale * lambda_boundary_consistency * boundary_consistency_loss
                if lambda_b13_temporal_gradient > 0.0 or lambda_b13_temporal_acceleration > 0.0:
                    temporal_valid_mask = _apply_west_edge_weight(batch["valid_mask"], batch)
                    temporal_boundary_mask = _apply_west_edge_weight(batch["boundary_valid_mask"], batch)
                    if temporal_consistency_max_t_fraction is not None:
                        t_gate = (t.float() <= float(diffusion_steps) * temporal_consistency_max_t_fraction).float()
                        gate = t_gate.view(-1, 1, 1, 1, 1)
                        temporal_valid_mask = temporal_valid_mask * gate
                        temporal_boundary_mask = temporal_boundary_mask * gate
                    history_last_k = batch["bt13_history_k"][:, -1]
                    history_last_valid = torch.isfinite(history_last_k).to(dtype=temporal_boundary_mask.dtype)
                    temporal_boundary_mask = temporal_boundary_mask * history_last_valid.unsqueeze(1).unsqueeze(2)
                    history_last_k = torch.nan_to_num(
                        history_last_k, nan=b13_clip_min, posinf=b13_clip_max, neginf=b13_clip_min
                    )
                    history_last = _normalize_bt13_tensor(history_last_k, b13_clip_min, b13_clip_max)
                    history_last = history_last.unsqueeze(1).unsqueeze(2)
                    temporal_gradient_loss, temporal_acceleration_loss = b13_temporal_consistency_losses(
                        pred_x0,
                        x0,
                        history_last,
                        batch["boundary_prev_target"],
                        temporal_valid_mask,
                        temporal_boundary_mask,
                        mode=temporal_consistency_loss_mode,
                        sample_weight=block_sample_weight,
                    )
                    loss = loss + temporal_consistency_scale * (
                        lambda_b13_temporal_gradient * temporal_gradient_loss
                        + lambda_b13_temporal_acceleration * temporal_acceleration_loss
                    )
                if (
                    (
                        lambda_cold_cloud_area > 0.0
                        or lambda_cold_cloud_dice > 0.0
                        or lambda_cold_cloud_retention > 0.0
                        or lambda_cold_cloud_false_alarm > 0.0
                    )
                    and cold_cloud_thresholds_norm
                ):
                    cold_cloud_mask = _apply_west_edge_weight(batch["valid_mask"], batch)
                    if cold_cloud_max_t_fraction is not None:
                        t_gate = (t.float() <= float(diffusion_steps) * cold_cloud_max_t_fraction).float()
                        cold_cloud_mask = cold_cloud_mask * t_gate.view(-1, 1, 1, 1, 1)
                    cold_cloud_area_loss, cold_cloud_dice_loss, cold_cloud_retention_loss = b13_cold_cloud_losses(
                        pred_x0,
                        x0,
                        cold_cloud_mask,
                        cold_cloud_thresholds_norm,
                        cold_cloud_softness_norm,
                        threshold_weights=cold_cloud_threshold_weights,
                        sample_weight=block_sample_weight,
                    )
                    cold_cloud_false_alarm_loss = b13_cold_cloud_false_alarm_loss(
                        pred_x0,
                        x0,
                        cold_cloud_mask,
                        cold_cloud_thresholds_norm,
                        cold_cloud_softness_norm,
                        threshold_weights=cold_cloud_threshold_weights,
                        sample_weight=block_sample_weight,
                    )
                    loss = loss + cold_cloud_scale * (
                        lambda_cold_cloud_area * cold_cloud_area_loss
                        + lambda_cold_cloud_dice * cold_cloud_dice_loss
                        + lambda_cold_cloud_retention * cold_cloud_retention_loss
                        + lambda_cold_cloud_false_alarm * cold_cloud_false_alarm_loss
                    )
                if lambda_b13_cold_core_recon > 0.0:
                    cold_core_mask = _apply_west_edge_weight(batch["valid_mask"], batch)
                    if cold_core_max_t_fraction is not None:
                        t_gate = (t.float() <= float(diffusion_steps) * cold_core_max_t_fraction).float()
                        cold_core_mask = cold_core_mask * t_gate.view(-1, 1, 1, 1, 1)
                    cold_core_mask = _apply_future_hour_multiplier(
                        cold_core_mask,
                        batch["future_hours"],
                        cold_core_late_lead_start_hour,
                        cold_core_late_lead_multiplier,
                    )
                    cold_core_recon_loss = b13_cold_core_reconstruction_loss(
                        pred_x0,
                        x0,
                        batch["target_k"],
                        cold_core_mask,
                        primary_threshold_k=cold_core_primary_threshold_k,
                        fallback_threshold_k=cold_core_fallback_threshold_k,
                        min_pixels=cold_core_min_pixels,
                        mode=b13_cold_core_recon_loss_mode,
                        sample_weight=block_sample_weight,
                    )
                    loss = loss + cold_core_scale * lambda_b13_cold_core_recon * cold_core_recon_loss
                if l2sp_reference is not None:
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        l2sp_loss = l2sp_parameter_loss(model, l2sp_reference)
                    loss = loss + l2sp_lambda * l2sp_loss
                backward_loss = loss / grad_accum
            scaler.scale(backward_loss).backward()
            if step % grad_accum == 0:
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(_unwrap_model(model).parameters(), grad_clip_norm)
                    grad_norm_sum += float(grad_norm.detach().cpu())
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = (not scaler.is_enabled()) or scaler.get_scale() >= previous_scale
                optimizer.zero_grad(set_to_none=True)
                if ema is not None and optimizer_stepped:
                    ema.update(_unwrap_model(model))
                if optimizer_stepped:
                    optim_steps += 1
                    global_step += 1
            train_loss_sum += float(loss.detach().cpu())
            diffusion_loss_sum += float(diffusion_loss.detach().cpu())
            recon_loss_sum += float(recon_loss.detach().cpu())
            recon_scale_sum += float(recon_scale)
            bias_loss_sum += float(bias_loss.detach().cpu())
            bias_scale_sum += float(bias_scale)
            bias_active_value = float(bias_active.detach().cpu())
            bias_k_sum += float(bias_norm.detach().cpu()) * b13_scale_k * bias_active_value
            bias_k_count += int(bias_active_value > 0.0)
            daylight_bias_loss_sum += float(daylight_bias_loss.detach().cpu())
            daylight_bias_scale_sum += float(daylight_bias_scale)
            daylight_bias_active_value = float(daylight_bias_active.detach().cpu())
            daylight_bias_k_sum += float(daylight_bias_norm.detach().cpu()) * b13_scale_k * daylight_bias_active_value
            daylight_bias_k_count += int(daylight_bias_active_value > 0.0)
            gradient_loss_sum += float(gradient_loss.detach().cpu())
            gradient_scale_sum += float(gradient_scale)
            lead_bias_loss_sum += float(lead_bias_loss.detach().cpu())
            lead_bias_scale_sum += float(lead_bias_scale)
            lead_bias_active_value = float(lead_bias_active.detach().cpu())
            lead_bias_k_sum += float(lead_bias_norm.detach().cpu()) * b13_scale_k * lead_bias_active_value
            lead_bias_k_count += int(lead_bias_active_value > 0.0)
            daylight_lead_bias_loss_sum += float(daylight_lead_bias_loss.detach().cpu())
            daylight_lead_bias_scale_sum += float(daylight_lead_bias_scale)
            daylight_lead_bias_active_value = float(daylight_lead_bias_active.detach().cpu())
            daylight_lead_bias_k_sum += float(daylight_lead_bias_norm.detach().cpu()) * b13_scale_k * daylight_lead_bias_active_value
            daylight_lead_bias_k_count += int(daylight_lead_bias_active_value > 0.0)
            boundary_consistency_loss_sum += float(boundary_consistency_loss.detach().cpu())
            boundary_consistency_scale_sum += float(boundary_consistency_scale)
            temporal_gradient_loss_sum += float(temporal_gradient_loss.detach().cpu())
            temporal_acceleration_loss_sum += float(temporal_acceleration_loss.detach().cpu())
            temporal_consistency_scale_sum += float(temporal_consistency_scale)
            cold_cloud_area_loss_sum += float(cold_cloud_area_loss.detach().cpu())
            cold_cloud_dice_loss_sum += float(cold_cloud_dice_loss.detach().cpu())
            cold_cloud_retention_loss_sum += float(cold_cloud_retention_loss.detach().cpu())
            cold_cloud_false_alarm_loss_sum += float(cold_cloud_false_alarm_loss.detach().cpu())
            cold_cloud_scale_sum += float(cold_cloud_scale)
            cold_core_recon_loss_sum += float(cold_core_recon_loss.detach().cpu())
            cold_core_scale_sum += float(cold_core_scale)
            l2sp_loss_sum += float(l2sp_loss.detach().cpu())
            history_source = batch.get("history_source")
            if history_source is not None:
                eligible_history = batch["cascade_block"] > 0
                predicted_history_sum += float(((history_source == 1) & eligible_history).float().sum().detach().cpu())
                predicted_history_count += int(eligible_history.sum().detach().cpu())
            west_edge_weight = batch.get("west_edge_weight_mask")
            if west_edge_weight is not None:
                west_edge_loss_weight_sum += float(west_edge_weight.detach().mean().cpu())
            else:
                west_edge_loss_weight_sum += 1.0
            block_weight_sum += block_weight_scalar
            mean_t_sum += float(t.float().mean().detach().cpu())
            train_steps += 1
            if is_main and progress_log_every_steps and step % progress_log_every_steps == 0:
                elapsed = time.perf_counter() - epoch_started
                mean_step_seconds = elapsed / max(step, 1)
                total_steps = len(loader)
                print(
                    f"epoch={epoch} progress={step}/{total_steps} "
                    f"mean_step_s={mean_step_seconds:.2f} elapsed_min={elapsed / 60.0:.1f} "
                    f"local_train_loss={float(loss.detach().cpu()):.6f}",
                    flush=True,
                )

        if train_steps % grad_accum != 0:
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(_unwrap_model(model).parameters(), grad_clip_norm)
                grad_norm_sum += float(grad_norm.detach().cpu())
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = (not scaler.is_enabled()) or scaler.get_scale() >= previous_scale
            optimizer.zero_grad(set_to_none=True)
            if ema is not None and optimizer_stepped:
                ema.update(_unwrap_model(model))
            if optimizer_stepped:
                optim_steps += 1
                global_step += 1

        train_loss = _distributed_average(train_loss_sum, train_steps, device)
        diffusion_loss_avg = _distributed_average(diffusion_loss_sum, train_steps, device)
        recon_loss_avg = _distributed_average(recon_loss_sum, train_steps, device)
        recon_scale_avg = _distributed_average(recon_scale_sum, train_steps, device)
        bias_loss_avg = _distributed_average(bias_loss_sum, train_steps, device)
        bias_scale_avg = _distributed_average(bias_scale_sum, train_steps, device)
        bias_k_avg = _distributed_average(bias_k_sum, bias_k_count, device)
        daylight_bias_loss_avg = _distributed_average(daylight_bias_loss_sum, train_steps, device)
        daylight_bias_scale_avg = _distributed_average(daylight_bias_scale_sum, train_steps, device)
        daylight_bias_k_avg = _distributed_average(daylight_bias_k_sum, daylight_bias_k_count, device)
        gradient_loss_avg = _distributed_average(gradient_loss_sum, train_steps, device)
        gradient_scale_avg = _distributed_average(gradient_scale_sum, train_steps, device)
        lead_bias_loss_avg = _distributed_average(lead_bias_loss_sum, train_steps, device)
        lead_bias_scale_avg = _distributed_average(lead_bias_scale_sum, train_steps, device)
        lead_bias_k_avg = _distributed_average(lead_bias_k_sum, lead_bias_k_count, device)
        daylight_lead_bias_loss_avg = _distributed_average(daylight_lead_bias_loss_sum, train_steps, device)
        daylight_lead_bias_scale_avg = _distributed_average(daylight_lead_bias_scale_sum, train_steps, device)
        daylight_lead_bias_k_avg = _distributed_average(daylight_lead_bias_k_sum, daylight_lead_bias_k_count, device)
        boundary_consistency_loss_avg = _distributed_average(boundary_consistency_loss_sum, train_steps, device)
        boundary_consistency_scale_avg = _distributed_average(boundary_consistency_scale_sum, train_steps, device)
        temporal_gradient_loss_avg = _distributed_average(temporal_gradient_loss_sum, train_steps, device)
        temporal_acceleration_loss_avg = _distributed_average(temporal_acceleration_loss_sum, train_steps, device)
        temporal_consistency_scale_avg = _distributed_average(temporal_consistency_scale_sum, train_steps, device)
        cold_cloud_area_loss_avg = _distributed_average(cold_cloud_area_loss_sum, train_steps, device)
        cold_cloud_dice_loss_avg = _distributed_average(cold_cloud_dice_loss_sum, train_steps, device)
        cold_cloud_retention_loss_avg = _distributed_average(cold_cloud_retention_loss_sum, train_steps, device)
        cold_cloud_false_alarm_loss_avg = _distributed_average(cold_cloud_false_alarm_loss_sum, train_steps, device)
        cold_cloud_scale_avg = _distributed_average(cold_cloud_scale_sum, train_steps, device)
        cold_core_recon_loss_avg = _distributed_average(cold_core_recon_loss_sum, train_steps, device)
        cold_core_scale_avg = _distributed_average(cold_core_scale_sum, train_steps, device)
        l2sp_loss_avg = _distributed_average(l2sp_loss_sum, train_steps, device)
        predicted_history_fraction = _distributed_average(predicted_history_sum, predicted_history_count, device)
        west_edge_loss_weight_avg = _distributed_average(west_edge_loss_weight_sum, train_steps, device)
        block_weight_avg = _distributed_average(block_weight_sum, train_steps, device)
        mean_t_avg = _distributed_average(mean_t_sum, train_steps, device)
        grad_norm_avg = _distributed_average(grad_norm_sum, optim_steps, device) if optim_steps else 0.0
        if ema is not None and ema_validate:
            with ema.average_parameters(_unwrap_model(model)):
                val_loss = validate(
                    model,
                    scheduler,
                    val_loader,
                    device,
                    diffusion_steps,
                    max_batches=max_val_batches,
                    timestep_mode=validation_timestep_mode,
                    timestep_bins=validation_timestep_bins,
                    noise_mode=validation_noise_mode,
                    seed=validation_seed,
                )
        else:
            val_loss = validate(
                model,
                scheduler,
                val_loader,
                device,
                diffusion_steps,
                max_batches=max_val_batches,
                timestep_mode=validation_timestep_mode,
                timestep_bins=validation_timestep_bins,
                noise_mode=validation_noise_mode,
                seed=validation_seed,
            )
        rollout_values = {
            "rmse_k": float("nan"),
            "bias_k": float("nan"),
            "pearson_r": float("nan"),
            "boundary_jump_ratio": float("nan"),
            "late_cold_core_rmse_k": float("nan"),
            "late_cold_core_bias_k": float("nan"),
        }
        run_rollout_validation = rollout_validation_every > 0 and epoch % rollout_validation_every == 0
        if ctx["distributed"]:
            dist.barrier()
        if is_main and run_rollout_validation:
            if ema is not None and ema_validate:
                with ema.average_parameters(_unwrap_model(model)):
                    rollout_values = rollout_validate(
                        config,
                        _unwrap_model(model),
                        scheduler,
                        device,
                        max_files=rollout_validation_max_files,
                        sampling_steps=rollout_validation_sampling_steps,
                        metrics_min_lon=rollout_validation_metrics_min_lon,
                    )
            else:
                rollout_values = rollout_validate(
                    config,
                    _unwrap_model(model),
                    scheduler,
                    device,
                    max_files=rollout_validation_max_files,
                    sampling_steps=rollout_validation_sampling_steps,
                    metrics_min_lon=rollout_validation_metrics_min_lon,
                )
        if ctx["distributed"]:
            dist.barrier()
        lr_scheduler.step()
        stop_training = False
        stop_reason = ""
        if is_main:
            rollout_guard_passed = True
            rollout_guard_reason = ""
            rollout_best_gate_passed, rollout_best_gate_reason = evaluate_rollout_best_gate(
                rollout_values,
                max_abs_bias_k=rollout_best_max_abs_bias_k,
                min_pearson_r=rollout_best_min_pearson_r,
                max_late_cold_core_rmse_k=rollout_best_max_late_cold_core_rmse_k,
            )
            rollout_guard_active = rollout_guard_enabled and epoch >= rollout_guard_start_epoch
            if rollout_guard_active and np.isfinite(rollout_values["rmse_k"]):
                reasons = []
                if abs(rollout_values["bias_k"]) > rollout_guard_max_abs_bias_k:
                    reasons.append(
                        f"abs_bias={abs(rollout_values['bias_k']):.3f}>{rollout_guard_max_abs_bias_k:.3f}K"
                    )
                if (
                    np.isfinite(rollout_baseline_rmse_k)
                    and rollout_values["rmse_k"] > rollout_baseline_rmse_k + rollout_guard_max_rmse_delta_k
                ):
                    reasons.append(
                        f"rmse={rollout_values['rmse_k']:.3f}>baseline+delta="
                        f"{rollout_baseline_rmse_k + rollout_guard_max_rmse_delta_k:.3f}K"
                    )
                if (
                    np.isfinite(best_rollout_rmse)
                    and rollout_values["rmse_k"] > best_rollout_rmse + rollout_guard_max_best_rmse_delta_k
                ):
                    reasons.append(
                        f"rmse={rollout_values['rmse_k']:.3f}>best+delta="
                        f"{best_rollout_rmse + rollout_guard_max_best_rmse_delta_k:.3f}K"
                    )
                rollout_guard_passed = not reasons
                rollout_guard_reason = "; ".join(reasons)
                rollout_guard_bad_count = 0 if rollout_guard_passed else rollout_guard_bad_count + 1
                stop_training = rollout_guard_should_stop(
                    rollout_guard_action,
                    rollout_guard_bad_count,
                    rollout_guard_patience,
                )
            rollout_best_candidate = is_rollout_best_candidate(
                epoch=epoch,
                start_epoch=rollout_best_start_epoch,
                rmse_k=rollout_values["rmse_k"],
                best_rmse_k=best_rollout_rmse,
                guard_passed=rollout_guard_passed,
                best_gate_passed=rollout_best_gate_passed,
            )
            next_best_rollout_rmse = (
                float(rollout_values["rmse_k"])
                if rollout_best_candidate
                else best_rollout_rmse
            )
            rollout_early_stop_improved = False
            rollout_early_stop_reason = ""
            early_stop_active = (
                rollout_early_stopping_enabled
                and run_rollout_validation
                and epoch >= rollout_early_stopping_start_epoch
            )
            if early_stop_active:
                (
                    rollout_early_stop_best_rmse,
                    rollout_early_stop_bad_count,
                    early_stop_triggered,
                    rollout_early_stop_improved,
                    rollout_early_stop_reason,
                ) = update_rollout_early_stopping(
                    rmse_k=rollout_values["rmse_k"],
                    gate_passed=rollout_best_gate_passed and rollout_guard_passed,
                    best_rmse_k=rollout_early_stop_best_rmse,
                    bad_count=rollout_early_stop_bad_count,
                    min_delta_k=rollout_early_stopping_min_delta_k,
                    patience=rollout_early_stopping_patience,
                )
                if early_stop_triggered:
                    stop_training = True
                    stop_reason = "rollout_plateau"
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "total_loss": train_loss,
                "diffusion_loss": diffusion_loss_avg,
                "b13_recon_loss": recon_loss_avg,
                "recon_scale": recon_scale_avg,
                "b13_bias_loss": bias_loss_avg,
                "b13_bias_scale": bias_scale_avg,
                "b13_bias_k_estimate": bias_k_avg,
                "daylight_bias_loss": daylight_bias_loss_avg,
                "daylight_bias_scale": daylight_bias_scale_avg,
                "daylight_bias_k_estimate": daylight_bias_k_avg,
                "b13_gradient_loss": gradient_loss_avg,
                "b13_gradient_scale": gradient_scale_avg,
                "lead_bias_loss": lead_bias_loss_avg,
                "lead_bias_scale": lead_bias_scale_avg,
                "lead_bias_k_estimate": lead_bias_k_avg,
                "daylight_lead_bias_loss": daylight_lead_bias_loss_avg,
                "daylight_lead_bias_scale": daylight_lead_bias_scale_avg,
                "daylight_lead_bias_k_estimate": daylight_lead_bias_k_avg,
                "boundary_consistency_loss": boundary_consistency_loss_avg,
                "boundary_consistency_scale": boundary_consistency_scale_avg,
                "temporal_gradient_loss": temporal_gradient_loss_avg,
                "temporal_acceleration_loss": temporal_acceleration_loss_avg,
                "temporal_consistency_scale": temporal_consistency_scale_avg,
                "cold_cloud_area_loss": cold_cloud_area_loss_avg,
                "cold_cloud_dice_loss": cold_cloud_dice_loss_avg,
                "cold_cloud_retention_loss": cold_cloud_retention_loss_avg,
                "cold_cloud_false_alarm_loss": cold_cloud_false_alarm_loss_avg,
                "cold_cloud_scale": cold_cloud_scale_avg,
                "cold_core_recon_loss": cold_core_recon_loss_avg,
                "cold_core_scale": cold_core_scale_avg,
                "l2sp_loss": l2sp_loss_avg,
                "predicted_history_fraction": predicted_history_fraction,
                "west_edge_loss_weight": west_edge_loss_weight_avg,
                "block_loss_weight": block_weight_avg,
                "mean_t": mean_t_avg,
                "val_loss": val_loss,
                "rollout_val_rmse_k": rollout_values["rmse_k"],
                "rollout_val_bias_k": rollout_values["bias_k"],
                "rollout_val_pearson_r": rollout_values["pearson_r"],
                "rollout_boundary_jump_ratio": rollout_values["boundary_jump_ratio"],
                "rollout_late_cold_core_rmse_k": rollout_values["late_cold_core_rmse_k"],
                "rollout_late_cold_core_bias_k": rollout_values["late_cold_core_bias_k"],
                "rollout_guard_passed": int(rollout_guard_passed),
                "rollout_early_stop_best_rmse_k": rollout_early_stop_best_rmse,
                "rollout_early_stop_bad_count": rollout_early_stop_bad_count,
                "rollout_early_stop_improved": int(rollout_early_stop_improved),
                "world_size": ctx["world_size"],
                "train_steps": train_steps,
                "optim_steps": optim_steps,
                "grad_norm": grad_norm_avg,
                "lr": epoch_learning_rate,
                "ema": int(ema is not None),
                "gpu_peak_memory_gib": (
                    float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
            }
            with log_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_train_log_fieldnames())
                writer.writerow(row)
            model_state = _unwrap_model(model).state_dict()
            checkpoint = {
                "checkpoint_schema_version": 2,
                "model_state": model_state,
                "raw_model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "val_loss": val_loss,
                "config": config,
                "cfg": cfg_cfg,
                "diffusion": diffusion_cfg,
                "condition_channels": condition_channels,
                "wrf_cloud_channels": wrf_cloud_channels,
                "aerosol_3d_condition_shape": aerosol_3d_shape,
                "target_channels": target_channels,
                "init_checkpoint": init_checkpoint,
                "init_checkpoint_weights_requested": init_checkpoint_weights if init_checkpoint else None,
                "init_checkpoint_weights_loaded": None if init_load_info is None else init_load_info["weights_loaded"],
                "l2sp_lambda": l2sp_lambda,
                "l2sp_reference_checkpoint": l2sp_reference_checkpoint if l2sp_lambda > 0.0 else None,
                "l2sp_reference_weights_loaded": l2sp_reference_weights_loaded,
                "b13_clip": config.get("dataset", {}).get("b13_clip", [180.0, 330.0]),
                "normalize_b13": bool(config.get("dataset", {}).get("normalize_b13", True)),
                "lr_scheduler_state": lr_scheduler.state_dict(),
                "rollout_val_rmse_k": rollout_values["rmse_k"],
                "rollout_val_bias_k": rollout_values["bias_k"],
                "rollout_val_pearson_r": rollout_values["pearson_r"],
                "rollout_boundary_jump_ratio": rollout_values["boundary_jump_ratio"],
                "rollout_late_cold_core_rmse_k": rollout_values["late_cold_core_rmse_k"],
                "rollout_late_cold_core_bias_k": rollout_values["late_cold_core_bias_k"],
                "rollout_baseline_rmse_k": rollout_baseline_rmse_k,
                "rollout_guard_bad_count": rollout_guard_bad_count,
                "rollout_guard_passed": rollout_guard_passed,
                "rollout_guard_reason": rollout_guard_reason,
                "rollout_guard_action": rollout_guard_action,
                "rollout_guard_start_epoch": rollout_guard_start_epoch,
                "rollout_best_start_epoch": rollout_best_start_epoch,
                "rollout_guard_max_best_rmse_delta_k": rollout_guard_max_best_rmse_delta_k,
                "rollout_best_gate_passed": rollout_best_gate_passed,
                "rollout_best_gate_reason": rollout_best_gate_reason,
                "rollout_best_max_abs_bias_k": rollout_best_max_abs_bias_k,
                "rollout_best_min_pearson_r": rollout_best_min_pearson_r,
                "rollout_best_max_late_cold_core_rmse_k": rollout_best_max_late_cold_core_rmse_k,
                "rollout_early_stopping_enabled": rollout_early_stopping_enabled,
                "rollout_early_stopping_start_epoch": rollout_early_stopping_start_epoch,
                "rollout_early_stopping_patience": rollout_early_stopping_patience,
                "rollout_early_stopping_min_delta_k": rollout_early_stopping_min_delta_k,
                "rollout_early_stop_best_rmse_k": rollout_early_stop_best_rmse,
                "rollout_early_stop_bad_count": rollout_early_stop_bad_count,
                "rollout_early_stop_reason": rollout_early_stop_reason,
                "optimizer_weight_decay": optimizer_weight_decay,
                "optimizer_betas": list(optimizer_betas),
                "best_diffusion_val_loss": min(best_val, val_loss),
                "best_rollout_rmse_k": next_best_rollout_rmse,
                "gpu_peak_memory_gib": row["gpu_peak_memory_gib"],
            }
            if ema is not None:
                checkpoint["ema_state"] = ema.state_dict()
                checkpoint["ema_model_state"] = _ema_inference_state(_unwrap_model(model), ema)
            torch.save(checkpoint, latest_path)
            torch.save(checkpoint, output_dir / "latest.pt")
            if (checkpoint_every > 0 and epoch % checkpoint_every == 0) or epoch in checkpoint_epochs:
                epoch_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
                torch.save(checkpoint, epoch_path)
                torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
            if val_loss < best_val:
                best_val = val_loss
                checkpoint["best_diffusion_val_loss"] = best_val
                torch.save(checkpoint, best_diffusion_path)
                torch.save(checkpoint, output_dir / "best_diffusion.pt")
                if rollout_validation_every <= 0:
                    torch.save(checkpoint, best_path)
                    torch.save(checkpoint, output_dir / "best.pt")
            if rollout_best_candidate:
                best_rollout_rmse = float(rollout_values["rmse_k"])
                checkpoint["best_rollout_rmse_k"] = best_rollout_rmse
                torch.save(checkpoint, best_path)
                torch.save(checkpoint, output_dir / "best.pt")
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} diffusion_loss={diffusion_loss_avg:.6f} "
                f"bias_k={bias_k_avg:.3f} daylight_bias_k={daylight_bias_k_avg:.3f} "
                f"lead_bias_k={lead_bias_k_avg:.3f} boundary_loss={boundary_consistency_loss_avg:.6f} "
                f"cold_retention={cold_cloud_retention_loss_avg:.6f} cold_core_loss={cold_core_recon_loss_avg:.6f} "
                f"grad_loss={gradient_loss_avg:.6f} "
                f"val_loss={val_loss:.6f} rollout_rmse={rollout_values['rmse_k']:.3f} "
                f"rollout_bias={rollout_values['bias_k']:.3f} l2sp={l2sp_loss_avg:.6f}"
            )
            rollout_guard_triggered = (
                not rollout_guard_passed
                and rollout_guard_bad_count >= rollout_guard_patience
            )
            if rollout_guard_triggered:
                action_text = "stopping" if rollout_guard_action == "stop" else "warning; continuing"
                print(
                    f"Rollout guard triggered ({action_text}) after "
                    f"{rollout_guard_bad_count} consecutive checks: "
                    f"{rollout_guard_reason}"
                )
                if rollout_guard_action == "stop":
                    stop_reason = "rollout_guard"
            if early_stop_active:
                print(
                    "Rollout early stopping: "
                    f"best={rollout_early_stop_best_rmse:.4f}K "
                    f"bad_checks={rollout_early_stop_bad_count}/{rollout_early_stopping_patience} "
                    f"status={rollout_early_stop_reason}"
                )
            if stop_training:
                early_stop_payload = {
                    "schema": "diffcloud_rollout_early_stop_v1",
                    "epoch": epoch,
                    "reason": stop_reason,
                    "rollout_rmse_k": rollout_values["rmse_k"],
                    "best_rollout_rmse_k": next_best_rollout_rmse,
                    "plateau_best_rmse_k": rollout_early_stop_best_rmse,
                    "bad_checks": rollout_early_stop_bad_count,
                    "patience": rollout_early_stopping_patience,
                    "min_delta_k": rollout_early_stopping_min_delta_k,
                }
                (output_dir / "early_stop.json").write_text(
                    json.dumps(early_stop_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            if (
                run_rollout_validation
                and epoch >= rollout_best_start_epoch
                and not rollout_best_gate_passed
            ):
                print(f"Rollout best gate rejected epoch {epoch}: {rollout_best_gate_reason}")

        if ctx["distributed"]:
            stop_tensor = torch.tensor([int(stop_training)], device=device, dtype=torch.int32)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())
        if stop_training:
            break

    if ctx["distributed"]:
        dist.barrier()
    if str(model_cfg.get("architecture", "")) == "grouped_aerosol_residual_v1":
        memory_report = {
            "rank": ctx["rank"], "world_size": ctx["world_size"],
            "allocated_gib": float(torch.cuda.max_memory_allocated(device)) / 1024**3 if torch.cuda.is_available() else 0.0,
            "reserved_gib": float(torch.cuda.max_memory_reserved(device)) / 1024**3 if torch.cuda.is_available() else 0.0,
        }
        (output_dir / f"gpu_memory_rank_{ctx['rank']}.json").write_text(
            json.dumps(memory_report, indent=2), encoding="utf-8"
        )
    cleanup_distributed()
    return best_path


def _ema_inference_state(model, ema) -> dict:
    """Export frozen backbone + trainable EMA as a strict-loadable full state."""
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    state.update({name: tensor.detach().cpu() for name, tensor in ema.shadow.items()})
    return state


def _train_log_fieldnames() -> list[str]:
    return [
        "epoch",
        "train_loss",
        "total_loss",
        "diffusion_loss",
        "b13_recon_loss",
        "recon_scale",
        "b13_bias_loss",
        "b13_bias_scale",
        "b13_bias_k_estimate",
        "daylight_bias_loss",
        "daylight_bias_scale",
        "daylight_bias_k_estimate",
        "b13_gradient_loss",
        "b13_gradient_scale",
        "lead_bias_loss",
        "lead_bias_scale",
        "lead_bias_k_estimate",
        "daylight_lead_bias_loss",
        "daylight_lead_bias_scale",
        "daylight_lead_bias_k_estimate",
        "boundary_consistency_loss",
        "boundary_consistency_scale",
        "temporal_gradient_loss",
        "temporal_acceleration_loss",
        "temporal_consistency_scale",
        "cold_cloud_area_loss",
        "cold_cloud_dice_loss",
        "cold_cloud_retention_loss",
        "cold_cloud_false_alarm_loss",
        "cold_cloud_scale",
        "cold_core_recon_loss",
        "cold_core_scale",
        "l2sp_loss",
        "predicted_history_fraction",
        "west_edge_loss_weight",
        "block_loss_weight",
        "mean_t",
        "val_loss",
        "rollout_val_rmse_k",
        "rollout_val_bias_k",
        "rollout_val_pearson_r",
        "rollout_boundary_jump_ratio",
        "rollout_late_cold_core_rmse_k",
        "rollout_late_cold_core_bias_k",
        "rollout_guard_passed",
        "rollout_early_stop_best_rmse_k",
        "rollout_early_stop_bad_count",
        "rollout_early_stop_improved",
        "world_size",
        "train_steps",
        "optim_steps",
        "grad_norm",
        "lr",
        "ema",
        "gpu_peak_memory_gib",
    ]


def _loss_scale(epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    progress = (epoch - start_epoch + 1) / float(warmup_epochs)
    return float(max(0.0, min(1.0, progress)))


@torch.no_grad() if torch is not None else (lambda fn: fn)
def validate(
    model,
    scheduler,
    loader,
    device,
    diffusion_steps: int,
    max_batches: int | None = None,
    timestep_mode: str = "stratified",
    timestep_bins: int = 10,
    noise_mode: str = "fixed",
    seed: int = 42,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    rank = int(os.environ.get("RANK", "0"))
    for step, batch in enumerate(loader, start=1):
        if max_batches is not None and step > int(max_batches):
            break
        batch = move_batch(batch, device)
        x0 = batch["target"]
        condition = batch["condition"]
        wrf_cloud_condition = batch.get("wrf_cloud_condition")
        aerosol_3d_condition = batch.get("aerosol_3d_condition")
        loss_mask = batch["loss_mask"]
        generator = _make_generator(device, int(seed) + rank * 1000003 + step)
        t = _validation_timesteps(
            batch_size=x0.shape[0],
            diffusion_steps=diffusion_steps,
            step=step,
            device=device,
            mode=timestep_mode,
            bins=timestep_bins,
            generator=generator,
        )
        noise = None
        if str(noise_mode).lower() in {"fixed", "deterministic"}:
            noise = torch.randn(x0.shape, device=device, dtype=x0.dtype, generator=generator)
        xt, noise = scheduler.q_sample(x0, t, noise=noise)
        pred_noise = model(
            xt,
            condition,
            t,
            **(b13_model_kwargs(wrf_cloud_condition, aerosol_3d_condition) or {}),
        )
        loss = scheduler.noise_loss(pred_noise, noise, loss_mask=loss_mask)
        total += float(loss.detach().cpu())
        count += 1
    return _distributed_average(total, count, device)


@torch.no_grad() if torch is not None else (lambda fn: fn)
def rollout_validate(
    config: dict[str, Any],
    model,
    scheduler,
    device,
    max_files: int = 8,
    sampling_steps: int = 25,
    metrics_min_lon: float | None = 80.0,
) -> dict[str, float]:
    from scripts.eval_b13cascade import continuity_metric_rows, continuity_metrics, predict_file
    from src.evaluation.b13_stability import cold_core_intensity_metrics

    rollout_config = copy.deepcopy(config)
    dataset_cfg = rollout_config.setdefault("dataset", {})
    dataset_cfg["patch_size"] = None
    dataset_cfg["patch_sampling"] = "center"
    dataset_cfg["patches_per_file"] = 1
    dataset_cfg["max_files"] = int(max_files)
    dataset_cfg["teacher_forcing"] = False
    model_cfg = copy.deepcopy(rollout_config.get("model", {}))
    model_cfg["sampling_steps"] = int(sampling_steps)
    sampling = SamplingConfig.from_config(rollout_config, steps=int(sampling_steps), ensemble_members=1)
    dataset = B13CascadeDataset(rollout_config, split="val")
    dataset.files = dataset.files[: int(max_files)]

    sum_squared_error = 0.0
    count = 0
    sum_pred = 0.0
    sum_truth = 0.0
    sum_pred_squared = 0.0
    sum_truth_squared = 0.0
    sum_product = 0.0
    late_core_squared_error = 0.0
    late_core_bias = 0.0
    late_core_count = 0
    continuity_rows: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    for file_index, path in enumerate(dataset.files):
        result = predict_file(
            dataset,
            file_index,
            model,
            scheduler,
            device,
            model_cfg,
            mode="autoregressive",
            sampling_config=sampling,
            cfg_cfg=rollout_config.get("cfg", {}),
        )
        domain = np.ones(result["valid"].shape[-2:], dtype=bool)
        lon = result.get("lon")
        if metrics_min_lon is not None and lon is not None:
            domain &= lon >= float(metrics_min_lon)
        valid = result["valid"] & domain[None, :, :]
        valid &= np.isfinite(result["pred_k"]) & np.isfinite(result["truth_k"])
        pred = result["pred_k"][valid].astype(np.float64)
        truth = result["truth_k"][valid].astype(np.float64)
        if pred.size:
            error = pred - truth
            sum_squared_error += float(np.sum(error**2))
            count += int(pred.size)
            sum_pred += float(np.sum(pred))
            sum_truth += float(np.sum(truth))
            sum_pred_squared += float(np.sum(pred**2))
            sum_truth_squared += float(np.sum(truth**2))
            sum_product += float(np.sum(pred * truth))
        for lead_index in range(12, result["pred_k"].shape[0]):
            core = cold_core_intensity_metrics(
                result["pred_k"][lead_index],
                result["truth_k"][lead_index],
                valid[lead_index],
                primary_threshold_k=230.0,
                fallback_threshold_k=240.0,
                min_pixels=32,
            )
            core_count = int(core["cold_core_pixel_count"])
            if core_count > 0 and np.isfinite(core["cold_core_rmse_k"]):
                late_core_squared_error += float(core["cold_core_rmse_k"] ** 2) * core_count
                late_core_bias += float(core["cold_core_mean_bias_k"]) * core_count
                late_core_count += core_count
        continuity_rows.extend(
            continuity_metric_rows(result, path=str(path), metrics_min_lon=metrics_min_lon)
        )
    if was_training:
        model.train()

    rmse = math.sqrt(sum_squared_error / count) if count else float("nan")
    covariance = count * sum_product - sum_pred * sum_truth
    pred_variance = count * sum_pred_squared - sum_pred**2
    truth_variance = count * sum_truth_squared - sum_truth**2
    denominator = math.sqrt(max(pred_variance, 0.0) * max(truth_variance, 0.0))
    pearson_r = covariance / denominator if denominator > 0.0 else float("nan")
    continuity = continuity_metrics(continuity_rows)
    result = {
        "rmse_k": float(rmse),
        "bias_k": float((sum_pred - sum_truth) / count) if count else float("nan"),
        "pearson_r": float(pearson_r),
        "boundary_jump_ratio": float(continuity["boundary_jump_ratio"]),
        "temporal_gradient_rmse_k": float(continuity["temporal_gradient_rmse_k"]),
        "num_files": float(len(dataset.files)),
        "valid_count": float(count),
        "late_cold_core_rmse_k": float(math.sqrt(late_core_squared_error / late_core_count))
        if late_core_count
        else float("nan"),
        "late_cold_core_bias_k": float(late_core_bias / late_core_count) if late_core_count else float("nan"),
        "late_cold_core_pixel_count": float(late_core_count),
    }
    print(
        "rollout_validation "
        f"files={len(dataset.files)} steps={sampling_steps} rmse_k={result['rmse_k']:.4f} "
        f"bias_k={result['bias_k']:.4f} r={result['pearson_r']:.4f} "
        f"late_core_rmse_k={result['late_cold_core_rmse_k']:.4f} "
        f"boundary_jump_ratio={result['boundary_jump_ratio']:.4f}"
    )
    return result
