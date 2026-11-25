"""Window-weighted B13 losses for adapter training."""
from __future__ import annotations

import torch

from src.training.trainer_b13cascade import (
    _apply_west_edge_weight,
    _block_loss_weights,
    _loss_scale,
    _masked_elementwise_mean,
    _normalize_bt13_tensor,
    b13_boundary_consistency_loss,
    b13_cold_cloud_false_alarm_loss,
    b13_cold_cloud_losses,
    b13_gradient_reconstruction_loss,
    b13_lead_bias_loss,
    b13_mean_bias_loss,
    b13_reconstruction_loss,
    b13_temporal_consistency_losses,
)

LOSS_RECIPE = "b13_window_weighted"
AUXILIARY_TERMS = {
    "recon_loss": ("lambda_b13_recon", 0.08),
    "bias_loss": ("lambda_b13_bias", 0.04),
    "daylight_bias_loss": ("lambda_daylight_bias", 0.05),
    "lead_bias_loss": ("lambda_b13_lead_bias", 0.06),
    "daylight_lead_bias_loss": ("lambda_daylight_lead_bias", 0.06),
    "gradient_loss": ("lambda_b13_gradient_recon", 0.02),
    "boundary_consistency_loss": ("lambda_boundary_consistency", 0.05),
    "temporal_gradient_loss": ("lambda_b13_temporal_gradient", 0.05),
    "temporal_acceleration_loss": ("lambda_b13_temporal_acceleration", 0.015),
    "cold_cloud_area_loss": ("lambda_cold_cloud_area", 0.02),
    "cold_cloud_dice_loss": ("lambda_cold_cloud_dice", 0.04),
    "cold_cloud_retention_loss": ("lambda_cold_cloud_retention", 0.02),
    "cold_cloud_false_alarm_loss": ("lambda_cold_cloud_false_alarm", 0.02),
}


def validate_3daer_loss_config(training: dict) -> None:
    expected = {
        "loss_recipe": LOSS_RECIPE,
        "cascade_block_loss_weights": [1.0, 1.15, 1.30, 1.45, 1.60],
        "b13_recon_loss_mode": "smooth_l1",
        "b13_bias_loss_mode": "mse",
        "daylight_bias_loss_mode": "mse",
        "b13_lead_bias_loss_mode": "mse",
        "daylight_lead_bias_loss_mode": "mse",
        "b13_gradient_recon_loss_mode": "smooth_l1",
        "boundary_consistency_loss_mode": "smooth_l1",
        "temporal_consistency_loss_mode": "smooth_l1",
        "cold_cloud_thresholds_k": [241.0, 225.0, 210.0],
        "cold_cloud_threshold_weights": [1.0, 1.2, 1.4],
        "cold_cloud_softness_k": 4.0,
        **{key: value for key, value in AUXILIARY_TERMS.values()},
    }
    for prefix in ("recon_loss", "bias_loss", "daylight_bias", "lead_bias", "daylight_lead_bias",
                   "gradient_recon", "boundary_consistency", "temporal_consistency", "cold_cloud"):
        expected.update({f"{prefix}_start_epoch": 1, f"{prefix}_warmup_epochs": 5,
                         f"{prefix}_max_t_fraction": 0.35})
    for key, value in expected.items():
        if training.get(key) != value:
            raise ValueError(f"3DAer B13 loss setting changed: training.{key}")
    for key, value in training.items():
        if key.startswith("lambda_") and key not in expected and float(value or 0) != 0.0:
            raise ValueError(f"Unexpected 3DAer loss coefficient: training.{key}")


def _window_mask(batch, blocks):
    mask = batch["loss_mask"].float()
    if mask.ndim != 5 or mask.shape[2] != 8:
        raise ValueError("3DAer training requires an eight-hour observation mask")
    hours = batch["future_hours"]
    expected_hours = 4 * blocks[:, None] + torch.arange(1, 9, device=blocks.device)
    if blocks.shape != (mask.shape[0],) or torch.any((blocks < 0) | (blocks > 4)):
        raise ValueError("Each sample requires a cascade block from 0 to 4")
    if hours.shape != expected_hours.shape or not torch.equal(hours, expected_hours):
        raise ValueError("Forecast hours differ from the sample's cascade block")
    has_pixels = (mask > 0).any(dim=(1, 3, 4))
    missing = batch.get("known_missing_target")
    if missing is not None:
        if missing.dtype != torch.bool or missing.shape != has_pixels.shape:
            raise ValueError("known_missing_target must be a boolean B,T tensor")
        if torch.any(has_pixels == missing):
            raise ValueError("The target mask differs from the declared observation availability")
    return mask


def b13_window_loss(pred_noise, true_noise, xt, batch, timesteps, scheduler,
                    epoch: int, training: dict, b13_clip) -> dict[str, torch.Tensor]:
    """Use the 2DAer term-specific reductions on the valid pixels of each window."""
    with torch.autocast(device_type=pred_noise.device.type, enabled=False):
        blocks = batch["cascade_block"].reshape(-1)
        mask = _window_mask(batch, blocks)
        valid = mask > 0
        weights = _block_loss_weights(blocks, training["cascade_block_loss_weights"], pred_noise.device)
        prediction = torch.where(valid, pred_noise.float(), 0.0)
        noise = torch.where(valid, true_noise.float(), 0.0)
        epsilon = _masked_elementwise_mean((prediction - noise).square(), mask, sample_weight=weights)

        gate = (timesteps <= scheduler.steps * training["recon_loss_max_t_fraction"])
        gate_mask = gate.view(-1, 1, 1, 1, 1)
        pred = scheduler.predict_x0(torch.where(valid, xt.float(), 0.0), timesteps, prediction).clamp(0.0, 1.0)
        target = torch.where(valid, batch["target"].float(), 0.0)
        aux_mask = _apply_west_edge_weight(mask, batch) * gate_mask
        daylight = batch["daylight_mask_future"]
        if daylight.ndim == 4:
            daylight = daylight.unsqueeze(1)
        day_mask = aux_mask * daylight
        low, high = b13_clip
        history_k = batch["bt13_history_k"][:, -1]
        boundary_mask = _apply_west_edge_weight(batch["boundary_valid_mask"].float(), batch)
        boundary_mask = boundary_mask * gate_mask * valid[:, :, :1] * torch.isfinite(history_k)[:, None, None]
        history_k = torch.nan_to_num(history_k, nan=low, posinf=high, neginf=low)
        history = _normalize_bt13_tensor(history_k, low, high)[:, None, None]
        previous = batch["boundary_prev_target"].float()

        weighted = {"sample_weight": weights}
        losses = {
            "recon_loss": b13_reconstruction_loss(pred, target, aux_mask, **weighted),
            "bias_loss": b13_mean_bias_loss(pred, target, aux_mask, **weighted)[0],
            "daylight_bias_loss": b13_mean_bias_loss(pred, target, day_mask, **weighted)[0],
            "lead_bias_loss": b13_lead_bias_loss(pred, target, aux_mask, **weighted)[0],
            "daylight_lead_bias_loss": b13_lead_bias_loss(pred, target, day_mask, **weighted)[0],
            "gradient_loss": b13_gradient_reconstruction_loss(pred, target, aux_mask, **weighted),
            "boundary_consistency_loss": b13_boundary_consistency_loss(
                pred, target, history, previous, boundary_mask, **weighted),
        }
        losses["temporal_gradient_loss"], losses["temporal_acceleration_loss"] = (
            b13_temporal_consistency_losses(pred, target, history, previous, aux_mask, boundary_mask, **weighted)
        )
        thresholds = [min(1.0, max(0.0, (value - low) / (high - low)))
                      for value in training["cold_cloud_thresholds_k"]]
        cold_args = (pred, target, aux_mask, thresholds, training["cold_cloud_softness_k"] / (high - low))
        cold_kwargs = dict(weighted, threshold_weights=training["cold_cloud_threshold_weights"])
        (losses["cold_cloud_area_loss"], losses["cold_cloud_dice_loss"],
         losses["cold_cloud_retention_loss"]) = b13_cold_cloud_losses(*cold_args, **cold_kwargs)
        losses["cold_cloud_false_alarm_loss"] = b13_cold_cloud_false_alarm_loss(*cold_args, **cold_kwargs)

        warmup = _loss_scale(epoch, training["recon_loss_start_epoch"], training["recon_loss_warmup_epochs"])
        auxiliary = warmup * sum(training[key] * losses[name] for name, (key, _) in AUXILIARY_TERMS.items())
        return {"loss": epsilon + auxiliary, "epsilon_loss": epsilon, "auxiliary_loss": auxiliary,
                "gate_fraction": gate.float().mean(), **losses}
