from __future__ import annotations

from contextlib import nullcontext

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def masked_mean(loss, mask):
    loss = loss * mask
    return loss.sum() / torch.clamp(mask.sum(), min=1.0)


def cloud_probability_reconstruction_loss(pred, target, valid_mask, mode: str = "mse"):
    pred_cloud = torch.clamp(pred[:, 0:1], 0.0, 1.0)
    target_cloud = torch.clamp(target[:, 0:1], 0.0, 1.0)
    if mode == "bce":
        eps = 1.0e-4
        if hasattr(torch, "amp"):
            ctx = torch.amp.autocast(device_type=pred_cloud.device.type, enabled=False)
        elif torch.cuda.is_available():
            ctx = torch.cuda.amp.autocast(enabled=False)
        else:
            ctx = nullcontext()
        with ctx:
            pred_prob = torch.clamp(pred_cloud.float(), eps, 1.0 - eps)
            loss = F.binary_cross_entropy(pred_prob, target_cloud.float(), reduction="none")
    else:
        loss = (pred_cloud - target_cloud) ** 2
    return masked_mean(loss, valid_mask)


def log_cot_reconstruction_loss(pred, target, cloudy_mask, mode: str = "smooth_l1"):
    pred_cot = pred[:, 1:2]
    target_cot = target[:, 1:2]
    if mode == "mse":
        loss = (pred_cot - target_cot) ** 2
    else:
        loss = F.smooth_l1_loss(pred_cot, target_cot, reduction="none")
    return masked_mean(loss, cloudy_mask)
